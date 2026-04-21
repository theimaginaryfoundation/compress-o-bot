package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/theimaginaryfoundation/compress-o-bot/migration"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/fileutils"
	"github.com/theimaginaryfoundation/compress-o-bot/pipeline"
)

func main() {
	cfg, err := parseFlags(flag.CommandLine, os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}
	if err := cfg.Validate(); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "missing OPENAI_API_KEY (or pass -api-key)")
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := os.MkdirAll(cfg.OutDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, fmt.Errorf("mkdir -out: %w", err).Error())
		os.Exit(2)
	}
	if cfg.SentimentOutDir != "" {
		if err := os.MkdirAll(cfg.SentimentOutDir, 0o755); err != nil {
			fmt.Fprintln(os.Stderr, fmt.Errorf("mkdir -sentiment-out: %w", err).Error())
			os.Exit(2)
		}
	}

	summaryFiles, err := collectChunkSummaryFiles(cfg.InPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}
	if len(summaryFiles) == 0 {
		fmt.Fprintln(os.Stderr, "no *.summary.json files found")
		os.Exit(2)
	}

	glossaryPath := cfg.GlossaryPath
	if glossaryPath == "" {
		glossaryPath = filepath.Join(cfg.InPath, "glossary.json")
	}
	glossary, err := migration.LoadGlossary(glossaryPath)
	if err != nil {
		glossary = migration.Glossary{Version: 1, Entries: []migration.GlossaryEntry{}}
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	rolluper := &pipeline.OpenAIThreadRolluper{Client: &client, Model: cfg.Model}
	sentRolluper := &pipeline.OpenAIThreadSentimentRolluper{Client: &client, Model: cfg.SentimentModel}

	if cfg.Concurrency == 0 {
		cfg.Concurrency = 1
	}

	indexPath := cfg.IndexPath
	if indexPath == "" {
		indexPath = filepath.Join(cfg.OutDir, "thread_index.json")
	}
	sentimentIndexPath := cfg.SentimentIndexPath
	if sentimentIndexPath == "" && cfg.SentimentOutDir != "" {
		sentimentIndexPath = filepath.Join(cfg.SentimentOutDir, "sentiment_thread_index.json")
	}

	byThread, err := groupChunkSummaries(summaryFiles)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}

	sentimentFiles, err := collectChunkSentimentSummaryFiles(cfg.InPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}
	byThreadSent, err := groupChunkSentimentSummaries(sentimentFiles)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}

	glossaryExcerpt := pipeline.GlossaryForPrompt(glossary, cfg.GlossaryMaxTerms)

	threadIDs := make([]string, 0, len(byThread))
	for id := range byThread {
		threadIDs = append(threadIDs, id)
	}
	sort.Strings(threadIDs)

	start := time.Now()
	totalThreads := int64(len(threadIDs))

	var processed int64
	if err := forEachThreadIDConcurrent(ctx, cfg.Concurrency, threadIDs, func(ctx context.Context, threadID string) error {
		if err := processThreadRollup(ctx, cfg, threadID, byThread, byThreadSent, rolluper, sentRolluper, glossaryExcerpt); err != nil {
			return err
		}
		n := atomic.AddInt64(&processed, 1)
		fmt.Fprintf(os.Stderr, "progress thread-rollup: %d/%d threads rolled up (last=%s elapsed=%s)\n",
			n, totalThreads, threadID, time.Since(start).Round(time.Second))
		return nil
	}); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	if cfg.Reindex {
		if err := rebuildThreadIndices(cfg, indexPath, sentimentIndexPath); err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}
	}

	if cfg.SentimentOutDir != "" {
		fmt.Fprintf(os.Stdout, "threads_processed=%d out_dir=%s index=%s sentiment_out_dir=%s sentiment_index=%s\n", processed, cfg.OutDir, indexPath, cfg.SentimentOutDir, sentimentIndexPath)
	} else {
		fmt.Fprintf(os.Stdout, "threads_processed=%d out_dir=%s index=%s\n", processed, cfg.OutDir, indexPath)
	}
}

func processThreadRollup(
	ctx context.Context,
	cfg Config,
	threadID string,
	byThread map[string][]migration.ChunkSummary,
	byThreadSent map[string][]migration.ChunkSentimentSummary,
	rolluper pipeline.ThreadRolluper,
	sentRolluper pipeline.ThreadSentimentRolluper,
	glossaryExcerpt string,
) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	outPath := filepath.Join(cfg.OutDir, threadID+".thread.summary.json")
	needSemantic := cfg.Overwrite || !fileExists(outPath)
	if !needSemantic && !cfg.Resume && !cfg.Overwrite {
		return fmt.Errorf("thread summary exists: %s", outPath)
	}

	if needSemantic {
		chunks := byThread[threadID]
		if err := writeThreadSummaryWithOptionalSplit(ctx, cfg, threadID, chunks, rolluper, glossaryExcerpt, outPath); err != nil {
			return err
		}
	}

	if cfg.SentimentOutDir != "" {
		if sentChunks, ok := byThreadSent[threadID]; ok && len(sentChunks) > 0 {
			sentOutPath := filepath.Join(cfg.SentimentOutDir, threadID+".thread.sentiment.summary.json")
			needSentiment := cfg.Overwrite || !fileExists(sentOutPath)
			if !needSentiment && !cfg.Resume && !cfg.Overwrite {
				return fmt.Errorf("thread sentiment summary exists: %s", sentOutPath)
			}
			if needSentiment {
				if err := writeThreadSentimentSummaryWithOptionalSplit(ctx, cfg, threadID, sentChunks, sentRolluper, glossaryExcerpt, sentOutPath); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func writeThreadSummaryWithOptionalSplit(
	ctx context.Context,
	cfg Config,
	threadID string,
	chunks []migration.ChunkSummary,
	rolluper pipeline.ThreadRolluper,
	glossaryExcerpt string,
	finalOutPath string,
) error {
	if cfg.MaxChunksPerThread <= 0 || len(chunks) <= cfg.MaxChunksPerThread {
		roll, err := rolluper.Rollup(ctx, threadID, chunks, glossaryExcerpt)
		if err != nil {
			return fmt.Errorf("failed rollup %s: %w", threadID, err)
		}
		return fileutils.WriteJSONFileAtomic(finalOutPath, roll, cfg.Pretty)
	}

	parts := pipeline.SplitChunksIntoWindows(chunks, cfg.MaxChunksPerThread)
	partSummaries := make([]migration.ThreadSummary, 0, len(parts))
	for i, win := range parts {
		partPath := semanticPartOutPath(cfg.OutDir, threadID, i+1, len(parts))
		needPart := cfg.Overwrite || !fileExists(partPath)
		if !needPart && !cfg.Resume && !cfg.Overwrite {
			return fmt.Errorf("thread summary part exists: %s", partPath)
		}

		if needPart {
			partRoll, err := rolluper.Rollup(ctx, threadID, win, glossaryExcerpt)
			if err != nil {
				return fmt.Errorf("failed rollup part %s part=%d/%d: %w", threadID, i+1, len(parts), err)
			}
			if err := fileutils.WriteJSONFileAtomic(partPath, partRoll, cfg.Pretty); err != nil {
				return err
			}
			partSummaries = append(partSummaries, partRoll)
		} else {
			ts, err := readThreadSummaryFile(partPath)
			if err != nil {
				return err
			}
			partSummaries = append(partSummaries, ts)
		}
	}

	merged, err := rolluper.RollupFromThreadSummaries(ctx, threadID, partSummaries, glossaryExcerpt)
	if err != nil {
		return fmt.Errorf("failed rollup merge %s: %w", threadID, err)
	}
	return fileutils.WriteJSONFileAtomic(finalOutPath, merged, cfg.Pretty)
}

func writeThreadSentimentSummaryWithOptionalSplit(
	ctx context.Context,
	cfg Config,
	threadID string,
	chunks []migration.ChunkSentimentSummary,
	rolluper pipeline.ThreadSentimentRolluper,
	glossaryExcerpt string,
	finalOutPath string,
) error {
	if cfg.MaxChunksPerThread <= 0 || len(chunks) <= cfg.MaxChunksPerThread {
		roll, err := rolluper.Rollup(ctx, threadID, chunks, glossaryExcerpt)
		if err != nil {
			return fmt.Errorf("failed sentiment rollup %s: %w", threadID, err)
		}
		return fileutils.WriteJSONFileAtomic(finalOutPath, roll, cfg.Pretty)
	}

	parts := pipeline.SplitChunksIntoWindows(chunks, cfg.MaxChunksPerThread)
	partSummaries := make([]migration.ThreadSentimentSummary, 0, len(parts))
	for i, win := range parts {
		partPath := sentimentPartOutPath(cfg.SentimentOutDir, threadID, i+1, len(parts))
		needPart := cfg.Overwrite || !fileExists(partPath)
		if !needPart && !cfg.Resume && !cfg.Overwrite {
			return fmt.Errorf("thread sentiment summary part exists: %s", partPath)
		}

		if needPart {
			partRoll, err := rolluper.Rollup(ctx, threadID, win, glossaryExcerpt)
			if err != nil {
				return fmt.Errorf("failed sentiment rollup part %s part=%d/%d: %w", threadID, i+1, len(parts), err)
			}
			if err := fileutils.WriteJSONFileAtomic(partPath, partRoll, cfg.Pretty); err != nil {
				return err
			}
			partSummaries = append(partSummaries, partRoll)
		} else {
			ts, err := readThreadSentimentSummaryFile(partPath)
			if err != nil {
				return err
			}
			partSummaries = append(partSummaries, ts)
		}
	}

	merged, err := rolluper.RollupFromThreadSentimentSummaries(ctx, threadID, partSummaries, glossaryExcerpt)
	if err != nil {
		return fmt.Errorf("failed sentiment rollup merge %s: %w", threadID, err)
	}
	return fileutils.WriteJSONFileAtomic(finalOutPath, merged, cfg.Pretty)
}

func semanticPartOutPath(outDir, threadID string, partNum int, total int) string {
	return filepath.Join(outDir, fmt.Sprintf("%s.thread.summary.part%02dof%02d.json", threadID, partNum, total))
}

func sentimentPartOutPath(outDir, threadID string, partNum int, total int) string {
	return filepath.Join(outDir, fmt.Sprintf("%s.thread.sentiment.summary.part%02dof%02d.json", threadID, partNum, total))
}

func readThreadSummaryFile(path string) (migration.ThreadSummary, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return migration.ThreadSummary{}, fmt.Errorf("read thread summary %s: %w", path, err)
	}
	var ts migration.ThreadSummary
	if err := json.Unmarshal(b, &ts); err != nil {
		return migration.ThreadSummary{}, fmt.Errorf("unmarshal thread summary %s: %w", path, err)
	}
	return ts, nil
}

func readThreadSentimentSummaryFile(path string) (migration.ThreadSentimentSummary, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return migration.ThreadSentimentSummary{}, fmt.Errorf("read thread sentiment summary %s: %w", path, err)
	}
	var ts migration.ThreadSentimentSummary
	if err := json.Unmarshal(b, &ts); err != nil {
		return migration.ThreadSentimentSummary{}, fmt.Errorf("unmarshal thread sentiment summary %s: %w", path, err)
	}
	return ts, nil
}

func forEachThreadIDConcurrent(ctx context.Context, concurrency int, threadIDs []string, fn func(context.Context, string) error) error {
	if concurrency <= 0 {
		concurrency = 1
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sem := make(chan struct{}, concurrency)
	errCh := make(chan error, len(threadIDs))

	var wg sync.WaitGroup
	for _, threadID := range threadIDs {
		threadID := threadID
		wg.Add(1)
		go func() {
			defer wg.Done()

			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return
			}
			defer func() { <-sem }()

			if err := fn(ctx, threadID); err != nil {
				errCh <- err
				cancel()
				return
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			return err
		}
	}
	if ctx.Err() != nil && !errors.Is(ctx.Err(), context.Canceled) {
		return ctx.Err()
	}
	return nil
}

func rebuildThreadIndices(cfg Config, indexPath string, sentimentIndexPath string) error {
	if err := rebuildSemanticThreadIndex(cfg, indexPath); err != nil {
		return err
	}
	if cfg.SentimentOutDir != "" {
		if err := rebuildSentimentThreadIndex(cfg, sentimentIndexPath); err != nil {
			return err
		}
	}
	return nil
}

func rebuildSemanticThreadIndex(cfg Config, indexPath string) error {
	var paths []string
	if err := filepath.WalkDir(cfg.OutDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(path), ".thread.summary.json") {
			paths = append(paths, path)
		}
		return nil
	}); err != nil {
		return fmt.Errorf("reindex semantic: walk thread summaries: %w", err)
	}
	sort.Strings(paths)

	f, err := os.OpenFile(indexPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("reindex semantic: open index: %w", err)
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	defer w.Flush()

	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return fmt.Errorf("reindex semantic: read %s: %w", p, err)
		}
		var ts migration.ThreadSummary
		if err := json.Unmarshal(b, &ts); err != nil {
			return fmt.Errorf("reindex semantic: unmarshal %s: %w", p, err)
		}
		if ts.ConversationID == "" {
			continue
		}
		rec := migration.BuildThreadIndexRecord(ts, p)
		rec.Summary = fileutils.Truncate(rec.Summary, cfg.IndexSummaryMaxChars)
		rec.Tags = limitSlice(rec.Tags, cfg.IndexTagsMax)
		rec.Terms = limitSlice(rec.Terms, cfg.IndexTermsMax)
		line, err := json.Marshal(rec)
		if err != nil {
			return fmt.Errorf("reindex semantic: marshal: %w", err)
		}
		if _, err := w.Write(append(line, '\n')); err != nil {
			return fmt.Errorf("reindex semantic: write: %w", err)
		}
	}
	return w.Flush()
}

func rebuildSentimentThreadIndex(cfg Config, sentimentIndexPath string) error {
	var paths []string
	if err := filepath.WalkDir(cfg.SentimentOutDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(path), ".thread.sentiment.summary.json") {
			paths = append(paths, path)
		}
		return nil
	}); err != nil {
		return fmt.Errorf("reindex sentiment: walk thread sentiment summaries: %w", err)
	}
	sort.Strings(paths)

	f, err := os.OpenFile(sentimentIndexPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("reindex sentiment: open index: %w", err)
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	defer w.Flush()

	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return fmt.Errorf("reindex sentiment: read %s: %w", p, err)
		}
		var ts migration.ThreadSentimentSummary
		if err := json.Unmarshal(b, &ts); err != nil {
			return fmt.Errorf("reindex sentiment: unmarshal %s: %w", p, err)
		}
		if ts.ConversationID == "" {
			continue
		}
		rec := migration.BuildThreadSentimentIndexRecord(ts, p)
		rec.EmotionalSummary = fileutils.Truncate(rec.EmotionalSummary, cfg.IndexSummaryMaxChars)
		rec.DominantEmotions = limitSlice(rec.DominantEmotions, cfg.IndexTermsMax)
		rec.RememberedEmotions = limitSlice(rec.RememberedEmotions, cfg.IndexTermsMax)
		rec.PresentEmotions = limitSlice(rec.PresentEmotions, cfg.IndexTermsMax)
		rec.EmotionalTensions = limitSlice(rec.EmotionalTensions, cfg.IndexTermsMax)
		rec.Themes = limitSlice(rec.Themes, cfg.IndexTagsMax)
		line, err := json.Marshal(rec)
		if err != nil {
			return fmt.Errorf("reindex sentiment: marshal: %w", err)
		}
		if _, err := w.Write(append(line, '\n')); err != nil {
			return fmt.Errorf("reindex sentiment: write: %w", err)
		}
	}
	return w.Flush()
}

func limitSlice(in []string, max int) []string {
	if max <= 0 || len(in) <= max {
		return in
	}
	return in[:max]
}

func parseFlags(fs *flag.FlagSet, args []string) (Config, error) {
	cfg := defaultConfig()
	fs.SetOutput(os.Stderr)
	fs.StringVar(&cfg.InPath, "in", cfg.InPath, "Path to summaries directory containing *.summary.json files (recursively)")
	fs.StringVar(&cfg.OutDir, "out", cfg.OutDir, "Output directory for per-thread summary JSON files")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "OpenAI model to use (e.g. gpt-5-mini)")
	fs.BoolVar(&cfg.Pretty, "pretty", false, "Pretty-print thread summary JSON files")
	fs.BoolVar(&cfg.Overwrite, "overwrite", false, "Overwrite existing thread summary JSON files")
	fs.StringVar(&cfg.IndexPath, "index", "", "Optional path for thread_index.json (default: <out>/thread_index.json)")
	fs.StringVar(&cfg.GlossaryPath, "glossary", "", "Optional glossary.json path (default: <in>/glossary.json)")
	fs.IntVar(&cfg.GlossaryMaxTerms, "glossary-max-terms", cfg.GlossaryMaxTerms, "Max glossary terms to include in the prompt (0 disables)")
	fs.StringVar(&cfg.SentimentOutDir, "sentiment-out", cfg.SentimentOutDir, "Output directory for per-thread sentiment summary JSON files (empty disables sentiment rollup)")
	fs.StringVar(&cfg.SentimentIndexPath, "sentiment-index", "", "Optional path for sentiment_thread_index.json (default: <sentiment-out>/sentiment_thread_index.json)")
	fs.StringVar(&cfg.SentimentModel, "sentiment-model", cfg.SentimentModel, "OpenAI model to use for sentiment rollup (e.g. gpt-5-mini)")
	fs.BoolVar(&cfg.Resume, "resume", cfg.Resume, "Skip thread rollups that already have output files")
	fs.BoolVar(&cfg.Reindex, "reindex", cfg.Reindex, "Rebuild thread index files from existing outputs at end of run")
	fs.IntVar(&cfg.Concurrency, "concurrency", cfg.Concurrency, "Max concurrent thread rollups")
	fs.IntVar(&cfg.MaxChunksPerThread, "max-chunks-per-thread", cfg.MaxChunksPerThread, "Max chunk summaries per thread rollup before splitting into parts (0 disables)")
	fs.IntVar(&cfg.IndexSummaryMaxChars, "index-summary-max-chars", cfg.IndexSummaryMaxChars, "Max chars in index summary fields (0 disables truncation)")
	fs.IntVar(&cfg.IndexTagsMax, "index-tags-max", cfg.IndexTagsMax, "Max tag/emotion/theme labels stored in index rows (0 disables limiting)")
	fs.IntVar(&cfg.IndexTermsMax, "index-terms-max", cfg.IndexTermsMax, "Max terms stored in index rows (0 disables limiting)")
	fs.StringVar(&cfg.APIKey, "api-key", "", "OpenAI API key (overrides OPENAI_API_KEY env var)")

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	cfg.InPath = filepath.Clean(cfg.InPath)
	cfg.OutDir = filepath.Clean(cfg.OutDir)
	if cfg.IndexPath != "" {
		cfg.IndexPath = filepath.Clean(cfg.IndexPath)
	}
	if cfg.GlossaryPath != "" {
		cfg.GlossaryPath = filepath.Clean(cfg.GlossaryPath)
	}
	if cfg.SentimentOutDir != "" {
		cfg.SentimentOutDir = filepath.Clean(cfg.SentimentOutDir)
	}
	if cfg.SentimentIndexPath != "" {
		cfg.SentimentIndexPath = filepath.Clean(cfg.SentimentIndexPath)
	}
	return cfg, nil
}

func collectChunkSummaryFiles(inPath string) ([]string, error) {
	fi, err := os.Stat(inPath)
	if err != nil {
		return nil, fmt.Errorf("stat -in: %w", err)
	}
	if !fi.IsDir() {
		return nil, errors.New("-in must be a directory containing summaries")
	}

	var files []string
	err = filepath.WalkDir(inPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		lp := strings.ToLower(path)
		if strings.HasSuffix(lp, ".sentiment.summary.json") {
			return nil
		}
		if strings.HasSuffix(lp, ".summary.json") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk summaries dir: %w", err)
	}
	sort.Strings(files)
	return files, nil
}

func collectChunkSentimentSummaryFiles(inPath string) ([]string, error) {
	fi, err := os.Stat(inPath)
	if err != nil {
		return nil, fmt.Errorf("stat -in: %w", err)
	}
	if !fi.IsDir() {
		return nil, errors.New("-in must be a directory containing summaries")
	}

	var files []string
	err = filepath.WalkDir(inPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(path), ".sentiment.summary.json") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk summaries dir: %w", err)
	}
	sort.Strings(files)
	return files, nil
}

func groupChunkSummaries(paths []string) (map[string][]migration.ChunkSummary, error) {
	out := make(map[string][]migration.ChunkSummary)
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		var s migration.ChunkSummary
		if err := json.Unmarshal(b, &s); err != nil {
			return nil, fmt.Errorf("unmarshal %s: %w", p, err)
		}
		if s.ConversationID == "" {
			return nil, fmt.Errorf("missing conversation_id in %s", p)
		}
		out[s.ConversationID] = append(out[s.ConversationID], s)
	}
	for k := range out {
		sort.Slice(out[k], func(i, j int) bool {
			if out[k][i].ChunkNumber != out[k][j].ChunkNumber {
				return out[k][i].ChunkNumber < out[k][j].ChunkNumber
			}
			return out[k][i].TurnStart < out[k][j].TurnStart
		})
	}
	return out, nil
}

func groupChunkSentimentSummaries(paths []string) (map[string][]migration.ChunkSentimentSummary, error) {
	out := make(map[string][]migration.ChunkSentimentSummary)
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		var s migration.ChunkSentimentSummary
		if err := json.Unmarshal(b, &s); err != nil {
			return nil, fmt.Errorf("unmarshal %s: %w", p, err)
		}
		if s.ConversationID == "" {
			return nil, fmt.Errorf("missing conversation_id in %s", p)
		}
		out[s.ConversationID] = append(out[s.ConversationID], s)
	}
	for k := range out {
		sort.Slice(out[k], func(i, j int) bool {
			if out[k][i].ChunkNumber != out[k][j].ChunkNumber {
				return out[k][i].ChunkNumber < out[k][j].ChunkNumber
			}
			return out[k][i].TurnStart < out[k][j].TurnStart
		})
	}
	return out, nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
