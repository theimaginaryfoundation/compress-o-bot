package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"
)

func TestParseFlags_Overrides(t *testing.T) {
	t.Parallel()

	fs := flag.NewFlagSet("archive-pipeline", flag.ContinueOnError)
	cfg, err := parseFlags(fs, []string{
		"-conversations", "docs/peanut-gallery/conversations.json",
		"-base-dir", "docs/peanut-gallery",
		"-model", "gpt-5-mini",
		"-target-turns", "20",
		"-concurrency", "5",
		"-batch-size", "25",
		"-max-chunks", "10",
		"-max-shard-bytes", "102400",
		"-index-summary-max-chars", "600",
		"-index-tags-max", "5",
		"-index-terms-max", "15",
		"-from-stage", "summarize",
	})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.FromStage != "summarize" {
		t.Fatalf("FromStage=%q", cfg.FromStage)
	}
	if cfg.SentimentModel != cfg.Model {
		t.Fatalf("SentimentModel=%q Model=%q", cfg.SentimentModel, cfg.Model)
	}
	if cfg.Concurrency != 5 || cfg.BatchSize != 25 || cfg.MaxChunks != 10 {
		t.Fatalf("concurrency/batch/max=%d/%d/%d", cfg.Concurrency, cfg.BatchSize, cfg.MaxChunks)
	}
}

func TestCopyFileIfExists(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	src := filepath.Join(dir, "src.txt")
	dst := filepath.Join(dir, "out", "dst.txt")

	// Missing src: no-op.
	copied, err := copyFileIfExists(src, dst, false)
	if err != nil {
		t.Fatalf("copy missing src: %v", err)
	}
	if copied {
		t.Fatalf("expected copied=false for missing src")
	}

	if err := os.WriteFile(src, []byte("hello"), 0o644); err != nil {
		t.Fatalf("write src: %v", err)
	}

	// First copy should create dst.
	copied, err = copyFileIfExists(src, dst, false)
	if err != nil {
		t.Fatalf("copy: %v", err)
	}
	if !copied {
		t.Fatalf("expected copied=true")
	}
	b, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("read dst: %v", err)
	}
	if string(b) != "hello" {
		t.Fatalf("dst=%q", string(b))
	}

	// Without overwrite, should not change dst.
	if err := os.WriteFile(src, []byte("new"), 0o644); err != nil {
		t.Fatalf("write src2: %v", err)
	}
	copied, err = copyFileIfExists(src, dst, false)
	if err != nil {
		t.Fatalf("copy no-overwrite: %v", err)
	}
	if copied {
		t.Fatalf("expected copied=false when dst exists and overwrite=false")
	}
	b, _ = os.ReadFile(dst)
	if string(b) != "hello" {
		t.Fatalf("dst changed unexpectedly: %q", string(b))
	}

	// With overwrite, should update dst.
	copied, err = copyFileIfExists(src, dst, true)
	if err != nil {
		t.Fatalf("copy overwrite: %v", err)
	}
	if !copied {
		t.Fatalf("expected copied=true when overwrite=true")
	}
	b, _ = os.ReadFile(dst)
	if string(b) != "new" {
		t.Fatalf("dst=%q", string(b))
	}
}
