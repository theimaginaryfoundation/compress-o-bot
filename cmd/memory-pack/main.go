package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
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

	mode := strings.ToLower(strings.TrimSpace(cfg.Mode))
	if mode == "" {
		mode = "semantic"
	}

	paths, err := collectThreadSummaryFiles(cfg.InPath, mode)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}
	if len(paths) == 0 {
		if mode == "sentiment" {
			fmt.Fprintln(os.Stderr, "no *.thread.sentiment.summary.json files found")
		} else {
			fmt.Fprintln(os.Stderr, "no *.thread.summary.json files found")
		}
		os.Exit(2)
	}

	indexPath := cfg.IndexPath
	if indexPath == "" {
		if mode == "sentiment" {
			indexPath = filepath.Join(cfg.OutDir, "sentiment_memory_index.jsonl")
		} else {
			indexPath = filepath.Join(cfg.OutDir, "memory_index.jsonl")
		}
	}

	switch mode {
	case "sentiment":
		summaries := make([]migration.ThreadSentimentSummary, 0, len(paths))
		for _, p := range paths {
			b, err := os.ReadFile(p)
			if err != nil {
				fmt.Fprintln(os.Stderr, fmt.Errorf("read %s: %w", p, err).Error())
				os.Exit(1)
			}
			var ts migration.ThreadSentimentSummary
			if err := json.Unmarshal(b, &ts); err != nil {
				fmt.Fprintln(os.Stderr, fmt.Errorf("unmarshal %s: %w", p, err).Error())
				os.Exit(1)
			}
			if ts.ConversationID == "" {
				continue
			}
			summaries = append(summaries, ts)
		}

		index, err := migration.WriteSentimentMemoryShards(summaries, migration.MemoryPackOptions{
			OutDir:           cfg.OutDir,
			MaxBytes:         cfg.MaxBytes,
			Overwrite:        cfg.Overwrite,
			IncludeKeyPoints: cfg.IncludeKeyPoints,
			IncludeTags:      cfg.IncludeTags,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}

		for i := range index {
			index[i].EmotionalSummary = truncateLimit(index[i].EmotionalSummary, cfg.IndexSummaryMaxChars)
			if cfg.IndexIncludeTags {
				index[i].Themes = limitSlice(index[i].Themes, cfg.IndexTagsMax)
			} else {
				index[i].Themes = nil
			}

			if cfg.IndexIncludeTerms {
				index[i].DominantEmotions = limitSlice(index[i].DominantEmotions, cfg.IndexTermsMax)
				index[i].RememberedEmotions = limitSlice(index[i].RememberedEmotions, cfg.IndexTermsMax)
				index[i].PresentEmotions = limitSlice(index[i].PresentEmotions, cfg.IndexTermsMax)
				index[i].EmotionalTensions = limitSlice(index[i].EmotionalTensions, cfg.IndexTermsMax)
			} else {
				index[i].DominantEmotions = nil
				index[i].RememberedEmotions = nil
				index[i].PresentEmotions = nil
				index[i].EmotionalTensions = nil
			}
		}

		if err := migration.WriteSentimentMemoryIndex(indexPath, index, cfg.Overwrite); err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}
		fmt.Fprintf(os.Stdout, "threads_packed=%d mode=sentiment out_dir=%s index=%s\n", len(index), cfg.OutDir, indexPath)
	default:
		summaries := make([]migration.ThreadSummary, 0, len(paths))
		for _, p := range paths {
			b, err := os.ReadFile(p)
			if err != nil {
				fmt.Fprintln(os.Stderr, fmt.Errorf("read %s: %w", p, err).Error())
				os.Exit(1)
			}
			var ts migration.ThreadSummary
			if err := json.Unmarshal(b, &ts); err != nil {
				fmt.Fprintln(os.Stderr, fmt.Errorf("unmarshal %s: %w", p, err).Error())
				os.Exit(1)
			}
			if ts.ConversationID == "" {
				continue
			}
			summaries = append(summaries, ts)
		}

		index, err := migration.WriteMemoryShards(summaries, migration.MemoryPackOptions{
			OutDir:           cfg.OutDir,
			MaxBytes:         cfg.MaxBytes,
			Overwrite:        cfg.Overwrite,
			IncludeKeyPoints: cfg.IncludeKeyPoints,
			IncludeTags:      cfg.IncludeTags,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}

		for i := range index {
			index[i].Summary = truncateLimit(index[i].Summary, cfg.IndexSummaryMaxChars)
			if cfg.IndexIncludeTags {
				index[i].Tags = limitSlice(index[i].Tags, cfg.IndexTagsMax)
			} else {
				index[i].Tags = nil
			}
			if cfg.IndexIncludeTerms {
				index[i].Terms = limitSlice(index[i].Terms, cfg.IndexTermsMax)
			} else {
				index[i].Terms = nil
			}
		}

		if err := migration.WriteMemoryIndex(indexPath, index, cfg.Overwrite); err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}
		fmt.Fprintf(os.Stdout, "threads_packed=%d mode=semantic out_dir=%s index=%s\n", len(index), cfg.OutDir, indexPath)
	}
}

func truncateLimit(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func limitSlice(in []string, max int) []string {
	if max <= 0 || len(in) <= max {
		return in
	}
	return in[:max]
}

type Config struct {
	InPath           string
	OutDir           string
	IndexPath        string
	MaxBytes         int
	Overwrite        bool
	IncludeKeyPoints bool
	IncludeTags      bool
	Mode             string

	IndexSummaryMaxChars int
	IndexTagsMax         int
	IndexTermsMax        int
	IndexIncludeTags     bool
	IndexIncludeTerms    bool
}

func (c Config) Validate() error {
	if c.InPath == "" {
		return errors.New("missing -in")
	}
	if c.OutDir == "" {
		return errors.New("missing -out")
	}
	if c.MaxBytes <= 0 {
		return errors.New("max-bytes must be > 0")
	}
	return nil
}

func defaultConfig() Config {
	return Config{
		InPath:               filepath.FromSlash("docs/peanut-gallery/threads/thread_summaries"),
		OutDir:               filepath.FromSlash("docs/peanut-gallery/threads/memory_shards"),
		MaxBytes:             100 * 1024,
		IncludeKeyPoints:     true,
		IncludeTags:          true,
		Mode:                 "semantic",
		IndexSummaryMaxChars: 400,
		IndexTagsMax:         5,
		IndexTermsMax:        15,
		IndexIncludeTags:     true,
		IndexIncludeTerms:    true,
	}
}

func parseFlags(fs *flag.FlagSet, args []string) (Config, error) {
	semanticDefaults := defaultConfig()
	cfg := semanticDefaults
	fs.SetOutput(os.Stderr)
	fs.StringVar(&cfg.InPath, "in", cfg.InPath, "Path to thread summaries directory (mode-dependent, recursively)")
	fs.StringVar(&cfg.OutDir, "out", cfg.OutDir, "Output directory for markdown shard files")
	fs.StringVar(&cfg.IndexPath, "index", "", "Optional path for memory_index.jsonl (default: <out>/memory_index.jsonl)")
	fs.IntVar(&cfg.MaxBytes, "max-bytes", cfg.MaxBytes, "Max UTF-8 bytes per markdown shard file (default ~100KB)")
	fs.BoolVar(&cfg.Overwrite, "overwrite", false, "Overwrite existing shard/index files")
	fs.BoolVar(&cfg.IncludeKeyPoints, "include-keypoints", cfg.IncludeKeyPoints, "Include key points section per thread")
	fs.BoolVar(&cfg.IncludeTags, "include-tags", cfg.IncludeTags, "Include tags/terms lines per thread")
	fs.StringVar(&cfg.Mode, "mode", cfg.Mode, "Packing mode: semantic or sentiment")
	fs.IntVar(&cfg.IndexSummaryMaxChars, "index-summary-max-chars", cfg.IndexSummaryMaxChars, "Max chars in index summary fields (0 disables truncation)")
	fs.IntVar(&cfg.IndexTagsMax, "index-tags-max", cfg.IndexTagsMax, "Max tag/theme labels stored in index rows (0 disables limiting)")
	fs.IntVar(&cfg.IndexTermsMax, "index-terms-max", cfg.IndexTermsMax, "Max term/emotion labels stored in index rows (0 disables limiting)")
	fs.BoolVar(&cfg.IndexIncludeTags, "index-include-tags", cfg.IndexIncludeTags, "Include tag/theme arrays in index rows")
	fs.BoolVar(&cfg.IndexIncludeTerms, "index-include-terms", cfg.IndexIncludeTerms, "Include term/emotion arrays in index rows")

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}

	// If user chose sentiment mode but left in/out at semantic defaults, switch to sentiment defaults.
	mode := strings.ToLower(strings.TrimSpace(cfg.Mode))
	if mode == "sentiment" {
		if cfg.InPath == semanticDefaults.InPath {
			cfg.InPath = filepath.FromSlash("docs/peanut-gallery/threads/thread_sentiment_summaries")
		}
		if cfg.OutDir == semanticDefaults.OutDir {
			cfg.OutDir = filepath.FromSlash("docs/peanut-gallery/threads/memory_shards_sentiment")
		}
	}

	cfg.InPath = filepath.Clean(cfg.InPath)
	cfg.OutDir = filepath.Clean(cfg.OutDir)
	if cfg.IndexPath != "" {
		cfg.IndexPath = filepath.Clean(cfg.IndexPath)
	}
	return cfg, nil
}

func collectThreadSummaryFiles(inPath string, mode string) ([]string, error) {
	fi, err := os.Stat(inPath)
	if err != nil {
		return nil, fmt.Errorf("stat -in: %w", err)
	}
	if !fi.IsDir() {
		return nil, errors.New("-in must be a directory")
	}

	wantSuffix := ".thread.summary.json"
	if mode == "sentiment" {
		wantSuffix = ".thread.sentiment.summary.json"
	}

	var files []string
	err = filepath.WalkDir(inPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(path), wantSuffix) {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk thread summaries: %w", err)
	}
	sort.Strings(files)
	return files, nil
}
