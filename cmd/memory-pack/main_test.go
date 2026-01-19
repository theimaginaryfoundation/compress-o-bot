package main

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
)

func TestParseFlags_Overrides(t *testing.T) {
	t.Parallel()

	fs := flag.NewFlagSet("memory-pack", flag.ContinueOnError)
	cfg, err := parseFlags(fs, []string{
		"-in", "docs/peanut-gallery/threads/thread_summaries",
		"-out", "docs/peanut-gallery/threads/memory_shards",
		"-index", "x.json",
		"-max-bytes", "12345",
		"-overwrite",
		"-include-keypoints=false",
		"-include-tags=false",
		"-mode", "semantic",
		"-index-summary-max-chars", "99",
		"-index-tags-max", "3",
		"-index-terms-max", "7",
		"-index-include-tags=false",
		"-index-include-terms=false",
	})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.MaxBytes != 12345 {
		t.Fatalf("MaxBytes=%d", cfg.MaxBytes)
	}
	if !cfg.Overwrite {
		t.Fatalf("Overwrite=false")
	}
	if cfg.IncludeKeyPoints || cfg.IncludeTags {
		t.Fatalf("expected include flags false")
	}
	if cfg.Mode != "semantic" {
		t.Fatalf("Mode=%q", cfg.Mode)
	}
	if cfg.IndexSummaryMaxChars != 99 || cfg.IndexTagsMax != 3 || cfg.IndexTermsMax != 7 {
		t.Fatalf("index caps=%d/%d/%d", cfg.IndexSummaryMaxChars, cfg.IndexTagsMax, cfg.IndexTermsMax)
	}
	if cfg.IndexIncludeTags || cfg.IndexIncludeTerms {
		t.Fatalf("expected index include flags false")
	}
}

func TestCollectThreadSummaryFiles_FindsRecursive(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "a"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	p := filepath.Join(root, "a", "x.thread.summary.json")
	if err := os.WriteFile(p, []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	files, err := collectThreadSummaryFiles(root, "semantic")
	if err != nil {
		t.Fatalf("collect: %v", err)
	}
	if len(files) != 1 {
		t.Fatalf("files=%v", files)
	}
}

func TestWriteMemoryShards_SplitsByMaxBytes(t *testing.T) {
	t.Parallel()

	outDir := t.TempDir()
	// Big enough that each summary section is ~200 bytes; set max to force splits.
	max := 300
	t1 := migration.ThreadSummary{ConversationID: "c1", Title: "T1", Summary: "a " + repeat("x", 180)}
	t2 := migration.ThreadSummary{ConversationID: "c2", Title: "T2", Summary: "b " + repeat("y", 180)}
	t3 := migration.ThreadSummary{ConversationID: "c3", Title: "T3", Summary: "c " + repeat("z", 180)}

	index, err := migration.WriteMemoryShards([]migration.ThreadSummary{t1, t2, t3}, migration.MemoryPackOptions{
		OutDir:    outDir,
		MaxBytes:  max,
		Overwrite: true,
	})
	if err != nil {
		t.Fatalf("WriteMemoryShards: %v", err)
	}
	if len(index) != 3 {
		t.Fatalf("len(index)=%d", len(index))
	}

	// Ensure multiple shard files were created.
	entries, err := os.ReadDir(outDir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	var mdCount int
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".md" {
			mdCount++
			b, err := os.ReadFile(filepath.Join(outDir, e.Name()))
			if err != nil {
				t.Fatalf("read shard: %v", err)
			}
			// Allow a bit of overhead for header; but ensure it doesn't explode.
			if len(b) > max+500 {
				t.Fatalf("shard too large: %d bytes", len(b))
			}
		}
	}
	if mdCount < 2 {
		t.Fatalf("mdCount=%d, want >=2", mdCount)
	}

	// Index should be marshalable.
	if _, err := json.Marshal(index); err != nil {
		t.Fatalf("marshal index: %v", err)
	}
}

func repeat(s string, n int) string {
	out := ""
	for i := 0; i < n; i++ {
		out += s
	}
	return out
}
