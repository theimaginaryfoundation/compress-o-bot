package main

import (
	"flag"
	"testing"
)

func TestParseFlags_Defaults(t *testing.T) {
	t.Parallel()

	fs := flag.NewFlagSet("archive-splitter", flag.ContinueOnError)
	cfg, err := parseFlags(fs, nil)
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.InputPath == "" {
		t.Fatalf("expected default InputPath")
	}
	if cfg.OutputDir == "" {
		t.Fatalf("expected default OutputDir")
	}
}

func TestParseFlags_Overrides(t *testing.T) {
	t.Parallel()

	fs := flag.NewFlagSet("archive-splitter", flag.ContinueOnError)
	cfg, err := parseFlags(fs, []string{
		"-in", "a/b/c.json",
		"-out", "x/y",
		"-pretty",
		"-overwrite",
		"-array-field", "conversations",
	})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.InputPath != "a/b/c.json" {
		t.Fatalf("InputPath=%q, want %q", cfg.InputPath, "a/b/c.json")
	}
	if cfg.OutputDir != "x/y" {
		t.Fatalf("OutputDir=%q, want %q", cfg.OutputDir, "x/y")
	}
	if !cfg.Pretty {
		t.Fatalf("Pretty=false, want true")
	}
	if !cfg.Overwrite {
		t.Fatalf("Overwrite=false, want true")
	}
	if cfg.ArrayField != "conversations" {
		t.Fatalf("ArrayField=%q, want %q", cfg.ArrayField, "conversations")
	}
}

func TestConfig_Validate(t *testing.T) {
	t.Parallel()

	if err := (Config{}).Validate(); err == nil {
		t.Fatalf("expected error for empty config")
	}
	if err := (Config{InputPath: "in.json"}).Validate(); err == nil {
		t.Fatalf("expected error for missing OutputDir")
	}
	if err := (Config{InputPath: "in.json", OutputDir: "out"}).Validate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
