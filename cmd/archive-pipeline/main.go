package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/theimaginaryfoundation/compress-o-bot/migration/fileutils"
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

	ctx := context.Background()

	stages := []string{"split", "chunk", "summarize", "rollup", "pack"}
	if cfg.OnlyStage != "" {
		stages = []string{cfg.OnlyStage}
	} else if cfg.FromStage != "" {
		stages = stagesFrom(stages, cfg.FromStage)
	}

	base := filepath.Clean(cfg.BaseDir)
	conversations := filepath.Clean(cfg.ConversationsPath)

	threadsDir := filepath.Join(base, "threads")
	chunksDir := filepath.Join(threadsDir, "chunks")
	summariesDir := filepath.Join(threadsDir, "summaries")
	threadSummariesDir := filepath.Join(threadsDir, "thread_summaries")
	threadSentimentSummariesDir := filepath.Join(threadsDir, "thread_sentiment_summaries")
	semanticShardsDir := filepath.Join(threadsDir, "memory_shards")
	sentimentShardsDir := filepath.Join(threadsDir, "memory_shards_sentiment")

	for _, stage := range stages {
		switch stage {
		case "split":
			// If threads already exist and we're not overwriting, skip.
			if !cfg.Overwrite && dirHasJSON(threadsDir) {
				fmt.Fprintln(os.Stdout, "skip split: threads already exist")
				continue
			}
			args := []string{
				"run", "./cmd/archive-splitter",
				"-in", conversations,
				"-out", threadsDir,
			}
			if cfg.Pretty {
				args = append(args, "-pretty")
			}
			if cfg.Overwrite {
				args = append(args, "-overwrite")
			}
			if err := runGo(ctx, args...); err != nil {
				os.Exit(1)
			}
		case "chunk":
			if !cfg.Overwrite && dirHasAny(chunksDir) {
				fmt.Fprintln(os.Stdout, "skip chunk: chunks already exist")
				continue
			}
			args := []string{
				"run", "./cmd/thread-chunker",
				"-in", threadsDir,
				"-out", chunksDir,
				"-model", cfg.Model,
				"-target-turns", fmt.Sprintf("%d", cfg.TargetTurns),
			}
			if cfg.Pretty {
				args = append(args, "-pretty")
			}
			if cfg.Overwrite {
				args = append(args, "-overwrite")
			}
			if err := runGo(ctx, args...); err != nil {
				os.Exit(1)
			}
		case "summarize":
			args := []string{
				"run", "./cmd/chunk-summarizer",
				"-in", chunksDir,
				"-out", summariesDir,
				"-model", cfg.Model,
				"-sentiment-model", cfg.SentimentModel,
				"-resume=true",
				"-reindex=true",
				"-concurrency", fmt.Sprintf("%d", cfg.Concurrency),
				"-batch-size", fmt.Sprintf("%d", cfg.BatchSize),
				"-max-chunks", fmt.Sprintf("%d", cfg.MaxChunks),
				"-index-summary-max-chars", fmt.Sprintf("%d", cfg.IndexSummaryMaxChars),
				"-index-tags-max", fmt.Sprintf("%d", cfg.IndexTagsMax),
				"-index-terms-max", fmt.Sprintf("%d", cfg.IndexTermsMax),
			}
			if cfg.Pretty {
				args = append(args, "-pretty")
			}
			if cfg.Overwrite {
				args = append(args, "-overwrite")
			}
			if cfg.SentimentPromptFile != "" {
				args = append(args, "-sentiment-prompt-file", cfg.SentimentPromptFile)
			}
			if err := runGo(ctx, args...); err != nil {
				os.Exit(1)
			}
		case "rollup":
			args := []string{
				"run", "./cmd/thread-rollup",
				"-in", summariesDir,
				"-out", threadSummariesDir,
				"-sentiment-out", threadSentimentSummariesDir,
				"-model", cfg.Model,
				"-sentiment-model", cfg.SentimentModel,
				"-resume=true",
				"-reindex=true",
				"-concurrency", fmt.Sprintf("%d", cfg.Concurrency),
				"-index-summary-max-chars", fmt.Sprintf("%d", cfg.IndexSummaryMaxChars),
				"-index-tags-max", fmt.Sprintf("%d", cfg.IndexTagsMax),
				"-index-terms-max", fmt.Sprintf("%d", cfg.IndexTermsMax),
			}
			if cfg.Pretty {
				args = append(args, "-pretty")
			}
			if cfg.Overwrite {
				args = append(args, "-overwrite")
			}
			if err := runGo(ctx, args...); err != nil {
				os.Exit(1)
			}
		case "pack":
			// Semantic
			{
				args := []string{
					"run", "./cmd/memory-pack",
					"-mode", "semantic",
					"-in", threadSummariesDir,
					"-out", semanticShardsDir,
					"-max-bytes", fmt.Sprintf("%d", cfg.MaxShardBytes),
					"-index-summary-max-chars", fmt.Sprintf("%d", cfg.IndexSummaryMaxChars),
					"-index-tags-max", fmt.Sprintf("%d", cfg.IndexTagsMax),
					"-index-terms-max", fmt.Sprintf("%d", cfg.IndexTermsMax),
				}
				if cfg.Overwrite {
					args = append(args, "-overwrite")
				}
				if err := runGo(ctx, args...); err != nil {
					os.Exit(1)
				}
			}
			// Sentiment
			{
				args := []string{
					"run", "./cmd/memory-pack",
					"-mode", "sentiment",
					"-in", threadSentimentSummariesDir,
					"-out", sentimentShardsDir,
					"-max-bytes", fmt.Sprintf("%d", cfg.MaxShardBytes),
					"-index-summary-max-chars", fmt.Sprintf("%d", cfg.IndexSummaryMaxChars),
					"-index-tags-max", fmt.Sprintf("%d", cfg.IndexTagsMax),
					"-index-terms-max", fmt.Sprintf("%d", cfg.IndexTermsMax),
				}
				if cfg.Overwrite {
					args = append(args, "-overwrite")
				}
				if err := runGo(ctx, args...); err != nil {
					os.Exit(1)
				}
			}

			// Copy glossary.json into the final shard output dirs for convenience.
			// The glossary is produced by chunk-summarizer in the summaries dir by default.
			glossarySrc := filepath.Join(summariesDir, "glossary.json")
			for _, dstDir := range []string{semanticShardsDir, sentimentShardsDir} {
				dst := filepath.Join(dstDir, "glossary.json")
				copied, err := fileutils.CopyFileIfExists(glossarySrc, dst, cfg.Overwrite)
				if err != nil {
					fmt.Fprintln(os.Stderr, "failed copying glossary:", err.Error())
					os.Exit(1)
				}
				if copied {
					fmt.Fprintln(os.Stdout, "copied glossary:", dst)
				}
			}
		default:
			fmt.Fprintln(os.Stderr, "unknown stage:", stage)
			os.Exit(2)
		}
	}
}

type Config struct {
	ConversationsPath string
	BaseDir           string

	Model          string
	SentimentModel string
	TargetTurns    int

	Concurrency int
	BatchSize   int
	MaxChunks   int

	MaxShardBytes int

	IndexSummaryMaxChars int
	IndexTagsMax         int
	IndexTermsMax        int

	FromStage string
	OnlyStage string

	Pretty    bool
	Overwrite bool

	SentimentPromptFile string
}

func parseFlags(fs *flag.FlagSet, args []string) (Config, error) {
	cfg := defaultConfig()
	fs.SetOutput(os.Stderr)

	fs.StringVar(&cfg.ConversationsPath, "conversations", cfg.ConversationsPath, "Path to conversations.json")
	fs.StringVar(&cfg.BaseDir, "base-dir", cfg.BaseDir, "Base output directory (defaults to docs/peanut-gallery)")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "OpenAI model for chunking/summarization/rollups (uses OPENAI_API_KEY)")
	fs.StringVar(&cfg.SentimentModel, "sentiment-model", cfg.SentimentModel, "OpenAI model override for sentiment passes (chunk sentiment + thread sentiment rollup)")
	fs.IntVar(&cfg.TargetTurns, "target-turns", cfg.TargetTurns, "Target turns per chunk for thread chunking")

	fs.IntVar(&cfg.Concurrency, "concurrency", cfg.Concurrency, "Concurrent chunk summarizations per batch")
	fs.IntVar(&cfg.BatchSize, "batch-size", cfg.BatchSize, "Batch size for glossary chaining/merging (0 = all)")
	fs.IntVar(&cfg.MaxChunks, "max-chunks", cfg.MaxChunks, "Limit number of chunks processed (0 = all)")

	fs.IntVar(&cfg.MaxShardBytes, "max-shard-bytes", cfg.MaxShardBytes, "Max UTF-8 bytes per markdown shard file")

	fs.IntVar(&cfg.IndexSummaryMaxChars, "index-summary-max-chars", cfg.IndexSummaryMaxChars, "Max chars in index summary fields (0 disables truncation)")
	fs.IntVar(&cfg.IndexTagsMax, "index-tags-max", cfg.IndexTagsMax, "Max tags/themes stored in index rows (0 disables limiting)")
	fs.IntVar(&cfg.IndexTermsMax, "index-terms-max", cfg.IndexTermsMax, "Max terms/emotions stored in index rows (0 disables limiting)")

	fs.StringVar(&cfg.FromStage, "from-stage", "", "Start at stage: split|chunk|summarize|rollup|pack")
	fs.StringVar(&cfg.OnlyStage, "only-stage", "", "Run only one stage: split|chunk|summarize|rollup|pack")

	fs.BoolVar(&cfg.Pretty, "pretty", cfg.Pretty, "Pretty-print JSON outputs where supported")
	fs.BoolVar(&cfg.Overwrite, "overwrite", cfg.Overwrite, "Overwrite existing outputs (disables resume behavior)")
	fs.StringVar(&cfg.SentimentPromptFile, "sentiment-prompt-file", "", "Optional path to a file containing a custom sentiment prompt header (prepended before required SECURITY+schema tail)")

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	if cfg.SentimentModel == "" {
		cfg.SentimentModel = cfg.Model
	}
	if cfg.SentimentPromptFile != "" {
		cfg.SentimentPromptFile = filepath.Clean(cfg.SentimentPromptFile)
	}
	return cfg, nil
}

func runGo(ctx context.Context, args ...string) error {
	cmd := exec.CommandContext(ctx, "go", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()

	start := time.Now()
	err := cmd.Run()
	if err != nil {
		fmt.Fprintln(os.Stderr, "command failed:", "go "+strings.Join(args, " "))
		fmt.Fprintln(os.Stderr, "error:", err.Error())
		return err
	}
	fmt.Fprintln(os.Stdout, "ok:", "go "+strings.Join(args, " "), "(", time.Since(start).Round(time.Millisecond).String()+")")
	return nil
}

func stagesFrom(stages []string, from string) []string {
	from = strings.ToLower(strings.TrimSpace(from))
	for i, s := range stages {
		if s == from {
			return stages[i:]
		}
	}
	return stages
}

func dirHasJSON(dir string) bool {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		if strings.HasSuffix(strings.ToLower(e.Name()), ".json") {
			return true
		}
	}
	return false
}

func dirHasAny(dir string) bool {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	return len(ents) > 0
}
