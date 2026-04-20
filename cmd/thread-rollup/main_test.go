package main

import (
	"context"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/fileutils"
	"github.com/theimaginaryfoundation/compress-o-bot/pipeline"
)

func TestDecodeModelJSON_ExtractsObjectFromWrappedText(t *testing.T) {
	t.Parallel()

	type out struct {
		A int `json:"a"`
	}

	var o out
	if err := fileutils.DecodeModelJSON("here you go:\n\n{\"a\": 2}\n", &o); err != nil {
		t.Fatalf("decodeModelJSON: %v", err)
	}
	if o.A != 2 {
		t.Fatalf("A=%d", o.A)
	}
}

func TestDecodeModelJSON_MissingClosingBrace_ReturnsUnexpectedEOF(t *testing.T) {
	t.Parallel()

	var m map[string]any
	err := fileutils.DecodeModelJSON("{\"a\": 1", &m)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestDecodeModelJSON_NoObjectReturnsError(t *testing.T) {
	t.Parallel()

	var o struct{ A int }
	if err := fileutils.DecodeModelJSON("prefix [1,2,3] suffix", &o); err == nil {
		t.Fatalf("expected error when no JSON object present")
	}
}

func TestParseFlags_Overrides(t *testing.T) {
	t.Parallel()

	fs := flag.NewFlagSet("thread-rollup", flag.ContinueOnError)
	cfg, err := parseFlags(fs, []string{
		"-in", "docs/peanut-gallery/threads/summaries",
		"-out", "docs/peanut-gallery/threads/thread_summaries",
		"-model", "gpt-5-mini",
		"-pretty",
		"-overwrite",
		"-glossary-max-terms", "10",
		"-concurrency", "3",
		"-max-chunks-per-thread", "9",
		"-api-key", "k",
	})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.InPath != "docs/peanut-gallery/threads/summaries" {
		t.Fatalf("InPath=%q", cfg.InPath)
	}
	if cfg.OutDir != "docs/peanut-gallery/threads/thread_summaries" {
		t.Fatalf("OutDir=%q", cfg.OutDir)
	}
	if cfg.Model != "gpt-5-mini" {
		t.Fatalf("Model=%q", cfg.Model)
	}
	if !cfg.Pretty || !cfg.Overwrite {
		t.Fatalf("Pretty=%v Overwrite=%v", cfg.Pretty, cfg.Overwrite)
	}
	if cfg.GlossaryMaxTerms != 10 {
		t.Fatalf("GlossaryMaxTerms=%d", cfg.GlossaryMaxTerms)
	}
	if cfg.Concurrency != 3 {
		t.Fatalf("Concurrency=%d", cfg.Concurrency)
	}
	if cfg.MaxChunksPerThread != 9 {
		t.Fatalf("MaxChunksPerThread=%d", cfg.MaxChunksPerThread)
	}
	if cfg.APIKey != "k" {
		t.Fatalf("APIKey=%q", cfg.APIKey)
	}
}

func TestChunkWindows_SplitsByMax(t *testing.T) {
	t.Parallel()

	in := make([]int, 16)
	for i := range in {
		in[i] = i + 1
	}

	got := pipeline.SplitChunksIntoWindows(in, 10)
	if len(got) != 2 {
		t.Fatalf("len=%d", len(got))
	}
	if len(got[0]) != 10 || len(got[1]) != 6 {
		t.Fatalf("sizes=%d,%d", len(got[0]), len(got[1]))
	}
	if got[0][0] != 1 || got[0][9] != 10 || got[1][0] != 11 || got[1][5] != 16 {
		t.Fatalf("windows=%v", got)
	}
}

func TestPartOutPaths(t *testing.T) {
	t.Parallel()

	sem := semanticPartOutPath("/out", "t", 1, 2)
	if sem != filepath.Join("/out", "t.thread.summary.part01of02.json") {
		t.Fatalf("semantic=%q", sem)
	}
	sent := sentimentPartOutPath("/sout", "t", 2, 12)
	if sent != filepath.Join("/sout", "t.thread.sentiment.summary.part02of12.json") {
		t.Fatalf("sentiment=%q", sent)
	}
}

func TestForEachThreadIDConcurrent_RespectsConcurrencyLimit(t *testing.T) {
	t.Parallel()

	threadIDs := make([]string, 50)
	for i := range threadIDs {
		threadIDs[i] = "t" + strconv.Itoa(i)
	}

	const limit = 3

	var inFlight int64
	var maxInFlight int64
	started := make(chan struct{}, len(threadIDs))
	block := make(chan struct{})

	done := make(chan error, 1)
	go func() {
		done <- forEachThreadIDConcurrent(context.Background(), limit, threadIDs, func(ctx context.Context, threadID string) error {
			n := atomic.AddInt64(&inFlight, 1)
			for {
				m := atomic.LoadInt64(&maxInFlight)
				if n <= m {
					break
				}
				if atomic.CompareAndSwapInt64(&maxInFlight, m, n) {
					break
				}
			}

			started <- struct{}{}
			<-block
			atomic.AddInt64(&inFlight, -1)
			return nil
		})
	}()

	// Let the executor start at least `limit` tasks, then verify it doesn't exceed the cap
	// while those tasks are blocked.
	for i := 0; i < limit; i++ {
		select {
		case <-started:
		case <-time.After(2 * time.Second):
			t.Fatalf("timeout waiting for worker start %d/%d", i+1, limit)
		}
	}

	// Give any extra goroutines a chance to run; they should be blocked on the semaphore.
	time.Sleep(50 * time.Millisecond)
	if got := atomic.LoadInt64(&maxInFlight); got > limit {
		close(block)
		t.Fatalf("maxInFlight=%d > limit=%d", got, limit)
	}

	close(block)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("forEachThreadIDConcurrent: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timeout waiting for executor to finish")
	}
	if got := atomic.LoadInt64(&maxInFlight); got != limit {
		t.Fatalf("maxInFlight=%d want=%d", got, limit)
	}
}

func TestGroupChunkSummaries_GroupsByConversationIDAndSorts(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	a := migration.ChunkSummary{ConversationID: "c1", ChunkNumber: 2, TurnStart: 10, Summary: "b"}
	b := migration.ChunkSummary{ConversationID: "c1", ChunkNumber: 1, TurnStart: 0, Summary: "a"}
	c := migration.ChunkSummary{ConversationID: "c2", ChunkNumber: 1, TurnStart: 0, Summary: "x"}

	paths := []string{
		writeJSON(t, dir, "a.summary.json", a),
		writeJSON(t, dir, "b.summary.json", b),
		writeJSON(t, dir, "c.summary.json", c),
	}

	m, err := groupChunkSummaries(paths)
	if err != nil {
		t.Fatalf("groupChunkSummaries: %v", err)
	}
	if len(m) != 2 {
		t.Fatalf("len=%d", len(m))
	}
	if len(m["c1"]) != 2 || m["c1"][0].ChunkNumber != 1 || m["c1"][1].ChunkNumber != 2 {
		t.Fatalf("c1=%v", m["c1"])
	}
}

func TestCollectChunkSummaryFiles_ExcludesSentiment(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	// semantic
	_ = writeJSON(t, dir, "a.summary.json", migration.ChunkSummary{ConversationID: "c1", ChunkNumber: 1})
	// sentiment (should be excluded from semantic collector)
	_ = writeJSON(t, dir, "a.sentiment.summary.json", migration.ChunkSentimentSummary{ConversationID: "c1", ChunkNumber: 1})

	files, err := collectChunkSummaryFiles(dir)
	if err != nil {
		t.Fatalf("collectChunkSummaryFiles: %v", err)
	}
	if len(files) != 1 || filepath.Base(files[0]) != "a.summary.json" {
		t.Fatalf("files=%v", files)
	}

	sfiles, err := collectChunkSentimentSummaryFiles(dir)
	if err != nil {
		t.Fatalf("collectChunkSentimentSummaryFiles: %v", err)
	}
	if len(sfiles) != 1 || filepath.Base(sfiles[0]) != "a.sentiment.summary.json" {
		t.Fatalf("sfiles=%v", sfiles)
	}
}

func writeJSON(t *testing.T, dir, name string, v any) string {
	t.Helper()
	p := filepath.Join(dir, name)
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return p
}
