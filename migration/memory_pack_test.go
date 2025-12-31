package migration

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteMemoryShards_IncludesThreadStartISO8601(t *testing.T) {
	t.Parallel()

	outDir := t.TempDir()
	ts := 1735689600.0 // 2025-01-01T00:00:00Z

	index, err := WriteMemoryShards([]ThreadSummary{
		{ConversationID: "c1", Title: "T1", ThreadStart: &ts, Summary: "hello"},
	}, MemoryPackOptions{
		OutDir:    outDir,
		MaxBytes:  100 * 1024,
		Overwrite: true,
	})
	if err != nil {
		t.Fatalf("WriteMemoryShards: %v", err)
	}
	if len(index) != 1 {
		t.Fatalf("len(index)=%d", len(index))
	}
	if index[0].ThreadStartISO != "2025-01-01T00:00:00Z" {
		t.Fatalf("ThreadStartISO=%q", index[0].ThreadStartISO)
	}

	b, err := os.ReadFile(filepath.Join(outDir, index[0].ShardFile))
	if err != nil {
		t.Fatalf("read shard: %v", err)
	}
	if !strings.Contains(string(b), "thread_start_time: `1735689600.000` (`2025-01-01T00:00:00Z`)") {
		t.Fatalf("missing ISO thread_start_time line:\n%s", string(b))
	}
}


