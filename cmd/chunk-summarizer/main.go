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
	"github.com/openai/openai-go/responses"
	"github.com/theimaginaryfoundation/compress-o-bot/migration"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/fileutils"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/provider"
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

	chunkFiles, err := collectChunkFiles(cfg.InPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}
	if len(chunkFiles) == 0 {
		fmt.Fprintln(os.Stderr, "no chunk .json files found")
		os.Exit(2)
	}
	if cfg.MaxChunks > 0 && len(chunkFiles) > cfg.MaxChunks {
		chunkFiles = chunkFiles[:cfg.MaxChunks]
	}

	start := time.Now()
	totalChunks := int64(len(chunkFiles))

	glossaryPath := cfg.GlossaryPath
	if glossaryPath == "" {
		glossaryPath = filepath.Join(cfg.OutDir, "glossary.json")
	}
	indexPath := cfg.IndexPath
	if indexPath == "" {
		indexPath = filepath.Join(cfg.OutDir, "index.json")
	}
	sentimentIndexPath := cfg.SentimentIndexPath
	if sentimentIndexPath == "" {
		sentimentIndexPath = filepath.Join(cfg.OutDir, "sentiment_index.json")
	}

	glossary, err := migration.LoadGlossary(glossaryPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}

	sentimentHeader := defaultSentimentPromptHeader
	if cfg.SentimentPromptFile != "" {
		h, err := loadPromptHeaderFromFile(cfg.SentimentPromptFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(2)
		}
		sentimentHeader = h
	}
	sentimentInstructions := composeSentimentInstructions(sentimentHeader)

	client := openai.NewClient(option.WithAPIKey(apiKey))
	summarizer := openAISummarizer{
		client:                &client,
		model:                 cfg.Model,
		sentimentModel:        cfg.SentimentModel,
		sentimentInstructions: sentimentInstructions,
	}

	if cfg.BatchSize == 0 {
		cfg.BatchSize = len(chunkFiles)
	}
	if cfg.Concurrency == 0 {
		cfg.Concurrency = 1
	}

	type glossaryUpdate struct {
		additions []migration.GlossaryAddition
		seenAt    *float64
	}

	var processed int64
	for bstart := 0; bstart < len(chunkFiles); bstart += cfg.BatchSize {
		bend := bstart + cfg.BatchSize
		if bend > len(chunkFiles) {
			bend = len(chunkFiles)
		}
		batch := chunkFiles[bstart:bend]
		glossaryExcerpt := glossaryForPrompt(glossary, cfg.GlossaryMaxTerms)

		sem := make(chan struct{}, cfg.Concurrency)
		errCh := make(chan error, len(batch))
		updatesCh := make(chan glossaryUpdate, len(batch))

		wg := sync.WaitGroup{}
		for _, chunkPath := range batch {
			wg.Add(1)
			go func(chunkPath string) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()

				select {
				case <-ctx.Done():
					errCh <- ctx.Err()
					return
				default:
				}

				semanticOut := semanticSummaryOutPath(cfg.InPath, cfg.OutDir, chunkPath)
				sentOut := sentimentSummaryOutPath(cfg.InPath, cfg.OutDir, chunkPath)
				if cfg.Resume && fileutils.FileExists(semanticOut) && fileutils.FileExists(sentOut) {
					return
				}

				chunk, err := readChunkFile(chunkPath)
				if err != nil {
					return
				}

				sumResp, err := summarizer.SummarizeChunkWithOptions(ctx, chunk, glossaryExcerpt, promptOptions{MaxTranscriptChars: 80_000, IncludeToolText: true})
				if err != nil {
					sumResp, err = summarizer.SummarizeChunkWithOptions(ctx, chunk, glossaryExcerpt, promptOptions{MaxTranscriptChars: 40_000, IncludeToolText: false})
					if err != nil {
						errCh <- fmt.Errorf("semantic summarize %s: %w", chunkPath, err)
						return
					}
				}

				sentResp, err := summarizer.SummarizeChunkSentimentWithOptions(ctx, chunk, glossaryExcerpt, promptOptions{MaxTranscriptChars: 80_000, IncludeToolText: true})
				if err != nil {
					sentResp, err = summarizer.SummarizeChunkSentimentWithOptions(ctx, chunk, glossaryExcerpt, promptOptions{MaxTranscriptChars: 40_000, IncludeToolText: false})
					if err != nil {
						errCh <- fmt.Errorf("sentiment summarize %s: %w", chunkPath, err)
						return
					}
				}

				semantic := migration.ChunkSummary{
					ConversationID: chunk.ConversationID,
					ThreadStart:    chunk.ThreadStart,
					ChunkNumber:    chunk.ChunkNumber,
					TurnStart:      chunk.TurnStart,
					TurnEnd:        chunk.TurnEnd,
					Summary:        sumResp.Summary,
					KeyPoints:      sumResp.KeyPoints,
					Tags:           sumResp.Tags,
					Terms:          sumResp.Terms,
				}
				if _, err := writeSummaryFile(cfg.InPath, cfg.OutDir, chunkPath, semantic, cfg.Pretty, cfg.Overwrite); err != nil {
					if !(cfg.Resume && strings.Contains(err.Error(), "already exists")) {
						errCh <- err
						return
					}
				}

				sentiment := migrationChunkSentimentSummary{
					ConversationID:     chunk.ConversationID,
					ThreadStart:        chunk.ThreadStart,
					ChunkNumber:        chunk.ChunkNumber,
					TurnStart:          chunk.TurnStart,
					TurnEnd:            chunk.TurnEnd,
					EmotionalSummary:   sentResp.EmotionalSummary,
					DominantEmotions:   sentResp.DominantEmotions,
					RememberedEmotions: sentResp.RememberedEmotions,
					PresentEmotions:    sentResp.PresentEmotions,
					EmotionalTensions:  sentResp.EmotionalTensions,
					RelationalShift:    sentResp.RelationalShift,
					EmotionalArc:       sentResp.EmotionalArc,
					Themes:             sentResp.Themes,
					SymbolsOrMetaphors: sentResp.SymbolsOrMetaphors,
					ResonanceNotes:     sentResp.ResonanceNotes,
					ToneMarkers:        sentResp.ToneMarkers,
				}
				if _, err := writeSentimentSummaryFile(cfg.InPath, cfg.OutDir, chunkPath, sentiment, cfg.Pretty, cfg.Overwrite); err != nil {
					if !(cfg.Resume && strings.Contains(err.Error(), "already exists")) {
						errCh <- err
						return
					}
				}

				additions := append([]migration.GlossaryAddition(nil), sumResp.GlossaryAdditions...)
				for _, t := range sumResp.Terms {
					additions = append(additions, migration.GlossaryAddition{Term: t})
				}
				updatesCh <- glossaryUpdate{additions: additions, seenAt: chunk.ThreadStart}

				n := atomic.AddInt64(&processed, 1)
				fmt.Fprintf(os.Stderr, "progress chunk-summarizer: %d/%d chunks summarized (last=%s elapsed=%s)\n",
					n, totalChunks, filepath.Base(chunkPath), time.Since(start).Round(time.Second))
			}(chunkPath)
		}

		wg.Wait()
		close(errCh)
		close(updatesCh)

		for err := range errCh {
			if err != nil {
				fmt.Fprintln(os.Stderr, err.Error())
				os.Exit(1)
			}
		}

		for u := range updatesCh {
			migration.MergeGlossary(&glossary, u.additions, u.seenAt)
		}

		if err := migration.SaveGlossary(glossaryPath, glossary); err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}
	}

	if cfg.GlossaryMinCount > 1 {
		migration.CullGlossary(&glossary, cfg.GlossaryMinCount)
	}
	if err := migration.SaveGlossary(glossaryPath, glossary); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	if cfg.Reindex {
		if err := rebuildIndices(cfg, indexPath, sentimentIndexPath); err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}
	} else {
		fmt.Fprintln(os.Stderr, "warning: -reindex=false may produce incomplete indices when -resume=true")
	}

	fmt.Fprintf(os.Stdout, "chunks_processed=%d summaries_out=%s index=%s sentiment_index=%s glossary=%s\n", processed, cfg.OutDir, indexPath, sentimentIndexPath, glossaryPath)
}

func parseFlags(fs *flag.FlagSet, args []string) (Config, error) {
	cfg := defaultConfig()
	fs.SetOutput(os.Stderr)

	fs.StringVar(&cfg.InPath, "in", cfg.InPath, "Path to chunk JSON file OR directory of chunk JSON files (recursively)")
	fs.StringVar(&cfg.OutDir, "out", cfg.OutDir, "Output directory for summary files + index/glossary")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "OpenAI model to use (e.g. gpt-5-mini)")
	fs.StringVar(&cfg.SentimentModel, "sentiment-model", cfg.SentimentModel, "OpenAI model override for sentiment chunk summaries (default: -model)")
	fs.StringVar(&cfg.SentimentPromptFile, "sentiment-prompt-file", "", "Optional path to a file containing a custom sentiment prompt header (prepended before required SECURITY+schema tail)")
	fs.BoolVar(&cfg.Pretty, "pretty", false, "Pretty-print summary JSON files")
	fs.BoolVar(&cfg.Overwrite, "overwrite", false, "Overwrite existing summary JSON files")
	fs.StringVar(&cfg.IndexPath, "index", "", "Optional path for index.json (default: <out>/index.json)")
	fs.StringVar(&cfg.SentimentIndexPath, "sentiment-index", "", "Optional path for sentiment_index.json (default: <out>/sentiment_index.json)")
	fs.StringVar(&cfg.GlossaryPath, "glossary", "", "Optional path for glossary.json (default: <out>/glossary.json)")
	fs.IntVar(&cfg.GlossaryMaxTerms, "glossary-max-terms", cfg.GlossaryMaxTerms, "Max glossary terms to include in the prompt (0 disables)")
	fs.IntVar(&cfg.GlossaryMinCount, "glossary-min-count", cfg.GlossaryMinCount, "Cull glossary terms with count < N at end of run (0 disables)")
	fs.IntVar(&cfg.MaxChunks, "max-chunks", 0, "Process only the first N chunks (0 = all)")
	fs.BoolVar(&cfg.Resume, "resume", cfg.Resume, "Skip chunks that already have both semantic+sentiment summary outputs")
	fs.BoolVar(&cfg.Reindex, "reindex", cfg.Reindex, "Rebuild index files from existing outputs at end of run (recommended with -resume)")
	fs.IntVar(&cfg.Concurrency, "concurrency", cfg.Concurrency, "Max concurrent chunk inferences within a batch")
	fs.IntVar(&cfg.BatchSize, "batch-size", cfg.BatchSize, "Batch size for glossary chaining/merging (0 = all)")
	fs.IntVar(&cfg.IndexSummaryMaxChars, "index-summary-max-chars", cfg.IndexSummaryMaxChars, "Max chars to keep in index summary fields (0 disables truncation)")
	fs.IntVar(&cfg.IndexTagsMax, "index-tags-max", cfg.IndexTagsMax, "Max tags/emotion/theme labels stored in index rows (0 disables limiting)")
	fs.IntVar(&cfg.IndexTermsMax, "index-terms-max", cfg.IndexTermsMax, "Max terms stored in index rows (0 disables limiting)")
	fs.StringVar(&cfg.APIKey, "api-key", "", "OpenAI API key (overrides OPENAI_API_KEY env var)")

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}

	if cfg.SentimentModel == "" {
		cfg.SentimentModel = cfg.Model
	}
	cfg.InPath = filepath.Clean(cfg.InPath)
	cfg.OutDir = filepath.Clean(cfg.OutDir)
	if cfg.SentimentPromptFile != "" {
		cfg.SentimentPromptFile = filepath.Clean(cfg.SentimentPromptFile)
	}
	if cfg.IndexPath != "" {
		cfg.IndexPath = filepath.Clean(cfg.IndexPath)
	}
	if cfg.SentimentIndexPath != "" {
		cfg.SentimentIndexPath = filepath.Clean(cfg.SentimentIndexPath)
	}
	if cfg.GlossaryPath != "" {
		cfg.GlossaryPath = filepath.Clean(cfg.GlossaryPath)
	}
	return cfg, nil
}

func semanticSummaryOutPath(inRoot, outRoot, chunkPath string) string {
	rel := chunkPath
	if fi, err := os.Stat(inRoot); err == nil && fi.IsDir() {
		if r, err := filepath.Rel(inRoot, chunkPath); err == nil {
			rel = r
		}
	}
	base := strings.TrimSuffix(rel, filepath.Ext(rel)) + ".summary.json"
	return filepath.Join(outRoot, base)
}

func sentimentSummaryOutPath(inRoot, outRoot, chunkPath string) string {
	rel := chunkPath
	if fi, err := os.Stat(inRoot); err == nil && fi.IsDir() {
		if r, err := filepath.Rel(inRoot, chunkPath); err == nil {
			rel = r
		}
	}
	base := strings.TrimSuffix(rel, filepath.Ext(rel)) + ".sentiment.summary.json"
	return filepath.Join(outRoot, base)
}

func limitStrings(in []string, max int) []string {
	if max <= 0 || len(in) <= max {
		return in
	}
	return in[:max]
}

func rebuildIndices(cfg Config, indexPath string, sentimentIndexPath string) error {
	var semanticPaths []string
	var sentimentPaths []string

	err := filepath.WalkDir(cfg.OutDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		lp := strings.ToLower(path)
		if strings.HasSuffix(lp, ".sentiment.summary.json") {
			sentimentPaths = append(sentimentPaths, path)
			return nil
		}
		if strings.HasSuffix(lp, ".summary.json") {
			semanticPaths = append(semanticPaths, path)
			return nil
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("reindex: walk summaries: %w", err)
	}
	sort.Strings(semanticPaths)
	sort.Strings(sentimentPaths)

	if err := os.MkdirAll(filepath.Dir(indexPath), 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(sentimentIndexPath), 0o755); err != nil {
		return err
	}

	indexFile, err := os.OpenFile(indexPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer indexFile.Close()
	indexW := bufio.NewWriterSize(indexFile, 1<<20)
	defer indexW.Flush()

	sentFile, err := os.OpenFile(sentimentIndexPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer sentFile.Close()
	sentW := bufio.NewWriterSize(sentFile, 1<<20)
	defer sentW.Flush()

	for _, sumPath := range semanticPaths {
		rel, err := filepath.Rel(cfg.OutDir, sumPath)
		if err != nil {
			continue
		}
		chunkRel := strings.TrimSuffix(rel, ".summary.json") + ".json"
		chunkPath := filepath.Join(cfg.InPath, chunkRel)

		chunk, err := readChunkFile(chunkPath)
		if err != nil {
			continue
		}
		b, err := os.ReadFile(sumPath)
		if err != nil {
			continue
		}
		var summary migration.ChunkSummary
		if err := json.Unmarshal(b, &summary); err != nil {
			continue
		}

		rec := migration.BuildIndexRecord(chunk, chunkPath, summary, sumPath)
		if cfg.IndexSummaryMaxChars > 0 {
			rec.Summary = fileutils.Truncate(rec.Summary, cfg.IndexSummaryMaxChars)
		}
		rec.Tags = limitStrings(rec.Tags, cfg.IndexTagsMax)
		rec.Terms = limitStrings(rec.Terms, cfg.IndexTermsMax)

		line, err := json.Marshal(rec)
		if err != nil {
			continue
		}
		if _, err := indexW.Write(append(line, '\n')); err != nil {
			return err
		}
	}

	for _, sumPath := range sentimentPaths {
		rel, err := filepath.Rel(cfg.OutDir, sumPath)
		if err != nil {
			continue
		}
		chunkRel := strings.TrimSuffix(rel, ".sentiment.summary.json") + ".json"
		chunkPath := filepath.Join(cfg.InPath, chunkRel)

		chunk, err := readChunkFile(chunkPath)
		if err != nil {
			continue
		}
		b, err := os.ReadFile(sumPath)
		if err != nil {
			continue
		}
		var summary migrationChunkSentimentSummary
		if err := json.Unmarshal(b, &summary); err != nil {
			continue
		}

		rec := sentimentIndexRecordFrom(chunk, chunkPath, sumPath, summary)
		if cfg.IndexSummaryMaxChars > 0 {
			rec.EmotionalSummary = fileutils.Truncate(rec.EmotionalSummary, cfg.IndexSummaryMaxChars)
		}
		rec.DominantEmotions = limitStrings(rec.DominantEmotions, cfg.IndexTagsMax)
		rec.Themes = limitStrings(rec.Themes, cfg.IndexTagsMax)

		line, err := json.Marshal(rec)
		if err != nil {
			continue
		}
		if _, err := sentW.Write(append(line, '\n')); err != nil {
			return err
		}
	}

	return nil
}

type SentimentIndexRecord struct {
	ConversationID string   `json:"conversation_id"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`
	ChunkNumber    int      `json:"chunk_number"`
	TurnStart      int      `json:"turn_start"`
	TurnEnd        int      `json:"turn_end"`

	ChunkPath            string `json:"chunk_path"`
	SentimentSummaryPath string `json:"sentiment_summary_path"`

	EmotionalSummary   string   `json:"emotional_summary"`
	DominantEmotions   []string `json:"dominant_emotions"`
	RememberedEmotions []string `json:"remembered_emotions"`
	PresentEmotions    []string `json:"present_emotions"`
	EmotionalTensions  []string `json:"emotional_tensions"`
	EmotionalArc       string   `json:"emotional_arc"`
	Themes             []string `json:"themes"`
	SymbolsOrMetaphors []string `json:"symbols_or_metaphors"`
	RelationalShift    string   `json:"relational_shift"`
	ResonanceNotes     string   `json:"resonance_notes,omitempty"`
	ToneMarkers        []string `json:"tone_markers,omitempty"`
}

func sentimentIndexRecordFrom(chunk migration.Chunk, chunkPath string, sentimentSummaryPath string, summary migrationChunkSentimentSummary) SentimentIndexRecord {
	return SentimentIndexRecord{
		ConversationID:       chunk.ConversationID,
		ThreadStart:          chunk.ThreadStart,
		ChunkNumber:          chunk.ChunkNumber,
		TurnStart:            chunk.TurnStart,
		TurnEnd:              chunk.TurnEnd,
		ChunkPath:            chunkPath,
		SentimentSummaryPath: sentimentSummaryPath,
		EmotionalSummary:     strings.TrimSpace(summary.EmotionalSummary),
		DominantEmotions:     summary.DominantEmotions,
		RememberedEmotions:   summary.RememberedEmotions,
		PresentEmotions:      summary.PresentEmotions,
		EmotionalTensions:    summary.EmotionalTensions,
		EmotionalArc:         strings.TrimSpace(summary.EmotionalArc),
		Themes:               summary.Themes,
		SymbolsOrMetaphors:   summary.SymbolsOrMetaphors,
		RelationalShift:      strings.TrimSpace(summary.RelationalShift),
		ResonanceNotes:       strings.TrimSpace(summary.ResonanceNotes),
		ToneMarkers:          summary.ToneMarkers,
	}
}

func collectChunkFiles(inPath string) ([]string, error) {
	fi, err := os.Stat(inPath)
	if err != nil {
		return nil, fmt.Errorf("stat -in: %w", err)
	}
	if !fi.IsDir() {
		if strings.ToLower(filepath.Ext(inPath)) != ".json" {
			return nil, fmt.Errorf("input file must be .json: %s", inPath)
		}
		return []string{inPath}, nil
	}

	var files []string
	err = filepath.WalkDir(inPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skip any nested summaries/index dirs if user points at a broad tree.
			name := d.Name()
			if strings.EqualFold(name, "summaries") || strings.EqualFold(name, "summary") || strings.EqualFold(name, "index") {
				return fs.SkipDir
			}
			return nil
		}
		if strings.ToLower(filepath.Ext(path)) != ".json" {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(path), ".summary.json") {
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk input dir: %w", err)
	}
	sort.Strings(files)
	return files, nil
}

func readChunkFile(path string) (migration.Chunk, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return migration.Chunk{}, err
	}
	var c migration.Chunk
	if err := json.Unmarshal(b, &c); err != nil {
		return migration.Chunk{}, err
	}
	if c.ConversationID == "" || c.ChunkNumber == 0 {
		return migration.Chunk{}, errors.New("not a chunk file (missing conversation_id/chunk_number)")
	}
	return c, nil
}

func writeSummaryFile(inRoot, outRoot, chunkPath string, summary migration.ChunkSummary, pretty bool, overwrite bool) (string, error) {
	rel := chunkPath
	if fi, err := os.Stat(inRoot); err == nil && fi.IsDir() {
		if r, err := filepath.Rel(inRoot, chunkPath); err == nil {
			rel = r
		}
	}

	base := strings.TrimSuffix(rel, filepath.Ext(rel)) + ".summary.json"
	outPath := filepath.Join(outRoot, base)

	if !overwrite {
		if _, err := os.Stat(outPath); err == nil {
			return "", fmt.Errorf("summary already exists: %s", outPath)
		} else if !errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("stat summary file: %w", err)
		}
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return "", fmt.Errorf("mkdir summary dir: %w", err)
	}

	var b []byte
	var err error
	if pretty {
		b, err = json.MarshalIndent(summary, "", "  ")
	} else {
		b, err = json.Marshal(summary)
	}
	if err != nil {
		return "", fmt.Errorf("marshal summary: %w", err)
	}

	if err := fileutils.WriteFileAtomicSameDir(outPath, b, 0o644); err != nil {
		return "", fmt.Errorf("write summary: %w", err)
	}
	return outPath, nil
}

type migrationChunkSentimentSummary struct {
	ConversationID string   `json:"conversation_id"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`
	ChunkNumber    int      `json:"chunk_number"`
	TurnStart      int      `json:"turn_start"`
	TurnEnd        int      `json:"turn_end"`

	// EmotionalSummary is "how it felt" in this chunk.
	EmotionalSummary string `json:"emotional_summary"`

	// RememberedEmotions are emotions recalled about past events discussed in the chunk
	// (retrospective, past-tense, memory-oriented), not emotions in the current interaction.
	RememberedEmotions []string `json:"remembered_emotions"`

	// PresentEmotions are emotions expressed/enacted in the current interaction (tone, pacing, humor, affirmation).
	PresentEmotions []string `json:"present_emotions"`

	// EmotionalTensions are contrasts between coexisting emotional states, expressed as "X vs Y" (2–5 max).
	EmotionalTensions []string `json:"emotional_tensions"`

	// RelationalShift describes how the relationship/framing changed because of this interaction.
	// If no shift occurred, say so explicitly.
	RelationalShift string `json:"relational_shift"`

	// DominantEmotions are 3–7 emotion labels clearly present or implied in the chunk.
	DominantEmotions []string `json:"dominant_emotions"`

	// EmotionalArc describes any change in emotions/stance across the chunk.
	EmotionalArc string `json:"emotional_arc"`

	// Themes are 3–8 recurring emotional/narrative themes.
	Themes []string `json:"themes"`

	// SymbolsOrMetaphors are 0–5 symbolic/metaphoric motifs meaningfully used.
	SymbolsOrMetaphors []string `json:"symbols_or_metaphors"`

	// ResonanceNotes are optional short notes on why this felt significant/memorable.
	ResonanceNotes string `json:"resonance_notes,omitempty"`

	// ToneMarkers are optional compact indicators of tone; emojis allowed.
	ToneMarkers []string `json:"tone_markers,omitempty"`
}

func writeSentimentSummaryFile(inRoot, outRoot, chunkPath string, summary migrationChunkSentimentSummary, pretty bool, overwrite bool) (string, error) {
	rel := chunkPath
	if fi, err := os.Stat(inRoot); err == nil && fi.IsDir() {
		if r, err := filepath.Rel(inRoot, chunkPath); err == nil {
			rel = r
		}
	}

	base := strings.TrimSuffix(rel, filepath.Ext(rel)) + ".sentiment.summary.json"
	outPath := filepath.Join(outRoot, base)

	if !overwrite {
		if _, err := os.Stat(outPath); err == nil {
			return "", fmt.Errorf("sentiment summary already exists: %s", outPath)
		} else if !errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("stat sentiment summary file: %w", err)
		}
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return "", fmt.Errorf("mkdir sentiment summary dir: %w", err)
	}

	var b []byte
	var err error
	if pretty {
		b, err = json.MarshalIndent(summary, "", "  ")
	} else {
		b, err = json.Marshal(summary)
	}
	if err != nil {
		return "", fmt.Errorf("marshal sentiment summary: %w", err)
	}

	if err := fileutils.WriteFileAtomicSameDir(outPath, b, 0o644); err != nil {
		return "", fmt.Errorf("write sentiment summary: %w", err)
	}
	return outPath, nil
}

func glossaryForPrompt(g migration.Glossary, maxTerms int) string {
	if maxTerms == 0 || len(g.Entries) == 0 {
		return ""
	}
	entries := g.Entries
	if maxTerms > 0 && len(entries) > maxTerms {
		entries = entries[:maxTerms]
	}
	var b strings.Builder
	for _, e := range entries {
		term := strings.TrimSpace(e.Term)
		if term == "" {
			continue
		}
		def := strings.TrimSpace(e.Definition)
		if def == "" {
			continue
		}
		fmt.Fprintf(&b, "- %s: %s\n", term, def)
	}
	return b.String()
}

type summarizeResponse struct {
	Summary           string                       `json:"summary"`
	KeyPoints         []string                     `json:"key_points"`
	Tags              []string                     `json:"tags"`
	Terms             []string                     `json:"terms"`
	GlossaryAdditions []migration.GlossaryAddition `json:"glossary_additions"`
}

type summarizeSentimentResponse struct {
	EmotionalSummary string   `json:"emotional_summary"`
	DominantEmotions []string `json:"dominant_emotions"`

	// New required fields:
	RememberedEmotions []string `json:"remembered_emotions"`
	PresentEmotions    []string `json:"present_emotions"`
	EmotionalTensions  []string `json:"emotional_tensions"`
	RelationalShift    string   `json:"relational_shift"`

	EmotionalArc       string   `json:"emotional_arc"`
	Themes             []string `json:"themes"`
	SymbolsOrMetaphors []string `json:"symbols_or_metaphors"`
	ResonanceNotes     string   `json:"resonance_notes"`
	ToneMarkers        []string `json:"tone_markers"`
}

type openAISummarizer struct {
	client                *openai.Client
	model                 string
	sentimentModel        string
	sentimentInstructions string
}

var summarizeSchema = provider.GenerateSchema[summarizeResponse]()
var summarizeSentimentSchema = provider.GenerateSchema[summarizeSentimentResponse]()

type promptOptions struct {
	MaxTranscriptChars int
	IncludeToolText    bool
}

func (s openAISummarizer) SummarizeChunk(ctx context.Context, chunk migration.Chunk, glossaryExcerpt string) (summarizeResponse, error) {
	return s.SummarizeChunkWithOptions(ctx, chunk, glossaryExcerpt, promptOptions{MaxTranscriptChars: 80_000, IncludeToolText: true})
}

func (s openAISummarizer) SummarizeChunkWithOptions(ctx context.Context, chunk migration.Chunk, glossaryExcerpt string, opt promptOptions) (summarizeResponse, error) {
	if s.client == nil {
		return summarizeResponse{}, errors.New("openAISummarizer: client is nil")
	}
	if s.model == "" {
		return summarizeResponse{}, errors.New("openAISummarizer: model is empty")
	}

	input := buildChunkPromptInputWithOptions(chunk, glossaryExcerpt, opt)
	format := responses.ResponseFormatTextConfigUnionParam{
		OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name:        "ChunkSummary",
			Schema:      summarizeSchema,
			Strict:      openai.Bool(true),
			Description: openai.String("Chunk summary JSON"),
			Type:        "json_schema",
		},
	}

	params := responses.ResponseNewParams{
		Model:           s.model,
		MaxOutputTokens: openai.Int(2500),
		Instructions:    openai.String(chunkSummarizerPrompt),
		ServiceTier:     responses.ResponseNewParamsServiceTierFlex,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: []responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(input, responses.EasyInputMessageRoleUser),
			},
		},
		Text: responses.ResponseTextConfigParam{
			Format: format,
		},
	}

	resp, err := provider.CallWithRetry(ctx, s.client, params)
	if err != nil {
		return summarizeResponse{}, err
	}

	var out summarizeResponse
	if err := fileutils.DecodeModelJSON(resp.OutputText(), &out); err != nil {
		return summarizeResponse{}, fmt.Errorf("unmarshal summary: %w", err)
	}
	out.Summary = strings.TrimSpace(out.Summary)
	return out, nil
}

func (s openAISummarizer) SummarizeChunkSentiment(ctx context.Context, chunk migration.Chunk, glossaryExcerpt string) (summarizeSentimentResponse, error) {
	return s.SummarizeChunkSentimentWithOptions(ctx, chunk, glossaryExcerpt, promptOptions{MaxTranscriptChars: 80_000, IncludeToolText: true})
}

func (s openAISummarizer) SummarizeChunkSentimentWithOptions(ctx context.Context, chunk migration.Chunk, glossaryExcerpt string, opt promptOptions) (summarizeSentimentResponse, error) {
	if s.client == nil {
		return summarizeSentimentResponse{}, errors.New("openAISummarizer: client is nil")
	}
	if s.sentimentModel == "" {
		return summarizeSentimentResponse{}, errors.New("openAISummarizer: sentiment model is empty")
	}
	if strings.TrimSpace(s.sentimentInstructions) == "" {
		return summarizeSentimentResponse{}, errors.New("openAISummarizer: sentiment instructions are empty")
	}

	input := buildChunkPromptInputWithOptions(chunk, glossaryExcerpt, opt)
	format := responses.ResponseFormatTextConfigUnionParam{
		OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name:        "ChunkSentimentSummary",
			Schema:      summarizeSentimentSchema,
			Strict:      openai.Bool(true),
			Description: openai.String("Chunk sentiment summary JSON"),
			Type:        "json_schema",
		},
	}

	params := responses.ResponseNewParams{
		Model:           s.sentimentModel,
		MaxOutputTokens: openai.Int(2500),
		Instructions:    openai.String(s.sentimentInstructions),
		ServiceTier:     responses.ResponseNewParamsServiceTierFlex,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: []responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(chunkSentimentSystemTurnStub, responses.EasyInputMessageRoleDeveloper),
				responses.ResponseInputItemParamOfMessage(input, responses.EasyInputMessageRoleUser),
			},
		},
		Text: responses.ResponseTextConfigParam{
			Format: format,
		},
	}

	resp, err := provider.CallWithRetry(ctx, s.client, params)
	if err != nil {
		return summarizeSentimentResponse{}, err
	}

	var out summarizeSentimentResponse
	if err := fileutils.DecodeModelJSON(resp.OutputText(), &out); err != nil {
		return summarizeSentimentResponse{}, fmt.Errorf("unmarshal sentiment summary: %w", err)
	}
	out.EmotionalSummary = strings.TrimSpace(out.EmotionalSummary)
	out.EmotionalArc = strings.TrimSpace(out.EmotionalArc)
	out.RelationalShift = strings.TrimSpace(out.RelationalShift)
	out.ResonanceNotes = strings.TrimSpace(out.ResonanceNotes)
	return out, nil
}

func composeSentimentInstructions(header string) string {
	header = strings.TrimSpace(header)
	if header == "" {
		header = strings.TrimSpace(defaultSentimentPromptHeader)
	}
	tail := strings.TrimSpace(sentimentPromptRequiredTail)
	return header + "\n\n" + tail
}

func buildChunkPromptInputWithOptions(chunk migration.Chunk, glossaryExcerpt string, opt promptOptions) string {
	var b strings.Builder
	fmt.Fprintf(&b, "chunk_metadata:\nconversation_id=%s\nchunk_number=%d\nturn_range=%d..%d\n\n",
		chunk.ConversationID, chunk.ChunkNumber, chunk.TurnStart, chunk.TurnEnd)

	if glossaryExcerpt != "" {
		b.WriteString("glossary:\n")
		b.WriteString(glossaryExcerpt)
		b.WriteString("\n")
	}

	b.WriteString("transcript:\n")
	maxTranscriptChars := opt.MaxTranscriptChars
	if maxTranscriptChars <= 0 {
		maxTranscriptChars = 80_000
	}
	total := 0
	for _, m := range chunk.Messages {
		role := m.Role
		if role == "" {
			role = "unknown"
		}
		name := ""
		if m.Name != "" {
			name = ":" + m.Name
		}

		line := ""
		if !opt.IncludeToolText && role == "tool" {
			// For retries / size pressure, keep tool outputs as compact references.
			desc := strings.TrimSpace(m.ContentType)
			if desc == "" {
				desc = "tool"
			}
			parts := []string{"[tool", m.Name, desc, m.Title, m.URL}
			line = strings.TrimSpace(strings.Join(parts, " "))
		} else if strings.TrimSpace(m.Text) != "" {
			line = m.Text
		} else if m.URL != "" || m.Title != "" {
			line = strings.TrimSpace(strings.Join([]string{m.Title, m.URL}, " "))
		} else {
			line = "[" + strings.TrimSpace(m.ContentType) + "]"
		}
		line = fileutils.Truncate(line, 2000)
		row := fmt.Sprintf("- %s%s: %s\n", role, name, fileutils.SanitizeNewlines(line))
		if total+len(row) > maxTranscriptChars {
			b.WriteString("... [transcript truncated]\n")
			break
		}
		b.WriteString(row)
		total += len(row)
	}
	return b.String()
}
func loadPromptHeaderFromFile(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", errors.New("sentiment-prompt-file is empty")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read sentiment-prompt-file: %w", err)
	}
	s := strings.TrimSpace(string(b))
	if s == "" {
		return "", errors.New("sentiment-prompt-file is empty after trimming whitespace")
	}
	return s, nil
}
