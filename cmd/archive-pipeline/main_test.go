package main

import (
	"flag"
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
