package migration

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteSentimentMemoryShards_SplitsByMaxBytes(t *testing.T) {
	t.Parallel()

	outDir := t.TempDir()
	max := 300
	ts := 1735689600.0 // 2025-01-01T00:00:00Z

	t1 := ThreadSentimentSummary{ConversationID: "c1", Title: "T1", ThreadStart: &ts, EmotionalSummary: "a " + repeat("x", 180)}
	t2 := ThreadSentimentSummary{ConversationID: "c2", Title: "T2", EmotionalSummary: "b " + repeat("y", 180)}
	t3 := ThreadSentimentSummary{ConversationID: "c3", Title: "T3", EmotionalSummary: "c " + repeat("z", 180)}

	index, err := WriteSentimentMemoryShards([]ThreadSentimentSummary{t1, t2, t3}, MemoryPackOptions{
		OutDir:    outDir,
		MaxBytes:  max,
		Overwrite: true,
	})
	if err != nil {
		t.Fatalf("WriteSentimentMemoryShards: %v", err)
	}
	if len(index) != 3 {
		t.Fatalf("len(index)=%d", len(index))
	}
	var found bool
	for _, r := range index {
		if r.ConversationID == "c1" {
			found = true
			if r.ThreadStartISO != "2025-01-01T00:00:00Z" {
				t.Fatalf("ThreadStartISO=%q", r.ThreadStartISO)
			}
		}
	}
	if !found {
		t.Fatalf("missing c1 record")
	}

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
			if strings.Contains(string(b), "conversation_id: `c1`") &&
				!strings.Contains(string(b), "thread_start_time: `1735689600.000` (`2025-01-01T00:00:00Z`)") {
				t.Fatalf("missing ISO thread_start_time line:\n%s", string(b))
			}
			if len(b) > max+500 {
				t.Fatalf("shard too large: %d bytes", len(b))
			}
		}
	}
	if mdCount < 2 {
		t.Fatalf("mdCount=%d, want >=2", mdCount)
	}
}

func repeat(s string, n int) string {
	out := ""
	for i := 0; i < n; i++ {
		out += s
	}
	return out
}
