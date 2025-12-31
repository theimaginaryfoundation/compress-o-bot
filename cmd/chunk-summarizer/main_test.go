package main

import (
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseFlags_Overrides(t *testing.T) {
	t.Parallel()

	fs := flag.NewFlagSet("chunk-summarizer", flag.ContinueOnError)
	cfg, err := parseFlags(fs, []string{
		"-in", "docs/peanut-gallery/threads/chunks",
		"-out", "docs/peanut-gallery/threads/summaries",
		"-model", "gpt-5-mini",
		"-pretty",
		"-overwrite",
		"-glossary-max-terms", "10",
		"-glossary-min-count", "3",
		"-max-chunks", "5",
		"-api-key", "k",
	})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.InPath != "docs/peanut-gallery/threads/chunks" {
		t.Fatalf("InPath=%q", cfg.InPath)
	}
	if cfg.OutDir != "docs/peanut-gallery/threads/summaries" {
		t.Fatalf("OutDir=%q", cfg.OutDir)
	}
	if cfg.Model != "gpt-5-mini" {
		t.Fatalf("Model=%q", cfg.Model)
	}
	if cfg.SentimentModel != "gpt-5-mini" {
		t.Fatalf("SentimentModel=%q", cfg.SentimentModel)
	}
	if !cfg.Pretty || !cfg.Overwrite {
		t.Fatalf("Pretty=%v Overwrite=%v", cfg.Pretty, cfg.Overwrite)
	}
	if cfg.GlossaryMaxTerms != 10 || cfg.GlossaryMinCount != 3 || cfg.MaxChunks != 5 {
		t.Fatalf("glossary max=%d min=%d maxchunks=%d", cfg.GlossaryMaxTerms, cfg.GlossaryMinCount, cfg.MaxChunks)
	}
	if cfg.APIKey != "k" {
		t.Fatalf("APIKey=%q", cfg.APIKey)
	}
}

func TestLoadPromptHeaderFromFile(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	p := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(p, []byte("hello world\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := loadPromptHeaderFromFile(p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got != "hello world" {
		t.Fatalf("got=%q", got)
	}
}

func TestComposeSentimentInstructions_AppendsRequiredTail(t *testing.T) {
	t.Parallel()

	got := composeSentimentInstructions("custom header")
	if !strings.HasPrefix(got, "custom header") {
		t.Fatalf("missing header prefix: %q", got[:min(40, len(got))])
	}
	if !strings.Contains(got, "\n\nSECURITY:\n") {
		t.Fatalf("missing SECURITY tail")
	}
	if !strings.Contains(got, "Return only JSON matching the schema.") {
		t.Fatalf("missing schema line")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestCollectChunkFiles_DirRecursiveAndSkipsSummaryFiles(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "t1"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "t1", "a.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "t1", "a.summary.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	files, err := collectChunkFiles(root)
	if err != nil {
		t.Fatalf("collectChunkFiles: %v", err)
	}
	if len(files) != 1 {
		t.Fatalf("files=%v, want 1", files)
	}
	if filepath.Base(files[0]) != "a.json" {
		t.Fatalf("got %s", files[0])
	}
}
