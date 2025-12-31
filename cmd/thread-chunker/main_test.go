package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"
)

func TestParseFlags_Overrides(t *testing.T) {
	t.Parallel()

	fs := flag.NewFlagSet("thread-chunker", flag.ContinueOnError)
	cfg, err := parseFlags(fs, []string{
		"-in", "docs/peanut-gallery/threads/x.json",
		"-out", "docs/peanut-gallery/threads/chunks",
		"-model", "gpt-5-mini",
		"-target-turns", "20",
		"-pretty",
		"-overwrite",
		"-api-key", "k",
	})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.InputPath != "docs/peanut-gallery/threads/x.json" {
		t.Fatalf("InputPath=%q", cfg.InputPath)
	}
	if cfg.OutputDir != "docs/peanut-gallery/threads/chunks" {
		t.Fatalf("OutputDir=%q", cfg.OutputDir)
	}
	if cfg.Model != "gpt-5-mini" {
		t.Fatalf("Model=%q", cfg.Model)
	}
	if cfg.TargetTurns != 20 {
		t.Fatalf("TargetTurns=%d", cfg.TargetTurns)
	}
	if !cfg.Pretty || !cfg.Overwrite {
		t.Fatalf("Pretty=%v Overwrite=%v", cfg.Pretty, cfg.Overwrite)
	}
	if cfg.APIKey != "k" {
		t.Fatalf("APIKey=%q", cfg.APIKey)
	}
}

func TestConfig_Validate(t *testing.T) {
	t.Parallel()

	if err := (Config{}).Validate(); err == nil {
		t.Fatalf("expected error")
	}
	if err := (Config{InputPath: "in.json", OutputDir: "out", Model: "m", TargetTurns: 20}).Validate(); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
}

func TestCollectInputFiles_File(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	p := filepath.Join(dir, "x.json")
	if err := os.WriteFile(p, []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	files, err := collectInputFiles(p)
	if err != nil {
		t.Fatalf("collectInputFiles: %v", err)
	}
	if len(files) != 1 || files[0] != p {
		t.Fatalf("files=%v, want [%s]", files, p)
	}
}

func TestCollectInputFiles_Directory_SortedAndSkipsNonJSONAndChunksDir(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "chunks"), 0o755); err != nil {
		t.Fatalf("mkdir chunks: %v", err)
	}

	a := filepath.Join(dir, "b.json")
	b := filepath.Join(dir, "a.json")
	if err := os.WriteFile(a, []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write b.json: %v", err)
	}
	if err := os.WriteFile(b, []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write a.json: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte(`x`), 0o644); err != nil {
		t.Fatalf("write note.txt: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "chunks", "c.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write chunks/c.json: %v", err)
	}

	files, err := collectInputFiles(dir)
	if err != nil {
		t.Fatalf("collectInputFiles: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("len(files)=%d, want 2: %v", len(files), files)
	}
	if filepath.Base(files[0]) != "a.json" || filepath.Base(files[1]) != "b.json" {
		t.Fatalf("files=%v, want [a.json b.json] sorted", files)
	}
}
