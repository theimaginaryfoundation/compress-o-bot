package migration

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type fakeDecider struct {
	breakpoints []int
}

func (f fakeDecider) DecideBreakpoints(ctx context.Context, thread SimplifiedConversation, turns []Turn, targetTurnsPerChunk int) ([]int, error) {
	return append([]int(nil), f.breakpoints...), nil
}

func TestBuildTurns_GroupsByUserMessage(t *testing.T) {
	t.Parallel()

	thread := SimplifiedConversation{
		ConversationID: "c1",
		Messages: []SimplifiedMessage{
			{Role: "user", Text: "u1"},
			{Role: "assistant", Text: "a1"},
			{Role: "tool", Name: "browser", ContentType: "tether_quote", Title: "T", URL: "https://x"},
			{Role: "user", Text: "u2"},
			{Role: "assistant", Text: "a2"},
		},
	}

	turns := BuildTurns(thread)
	if len(turns) != 2 {
		t.Fatalf("len(turns)=%d, want 2", len(turns))
	}
	if turns[0].StartMessageIndex != 0 || turns[0].EndMessageIndex != 2 {
		t.Fatalf("turn0 range=%d..%d, want 0..2", turns[0].StartMessageIndex, turns[0].EndMessageIndex)
	}
	if turns[1].StartMessageIndex != 3 || turns[1].EndMessageIndex != 4 {
		t.Fatalf("turn1 range=%d..%d, want 3..4", turns[1].StartMessageIndex, turns[1].EndMessageIndex)
	}
}

func TestApplyTurnBreakpoints_SlicesMessages(t *testing.T) {
	t.Parallel()

	thread := SimplifiedConversation{
		ConversationID: "c1",
		Title:          "t",
		Messages: []SimplifiedMessage{
			{Role: "user", Text: "u1"},
			{Role: "assistant", Text: "a1"},
			{Role: "user", Text: "u2"},
			{Role: "assistant", Text: "a2"},
			{Role: "user", Text: "u3"},
			{Role: "assistant", Text: "a3"},
		},
	}
	turns := BuildTurns(thread)

	chunks, err := ApplyTurnBreakpoints(thread, turns, []int{2})
	if err != nil {
		t.Fatalf("ApplyTurnBreakpoints: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("len(chunks)=%d, want 2", len(chunks))
	}
	if chunks[0].TurnStart != 0 || chunks[0].TurnEnd != 2 {
		t.Fatalf("chunk0 turns=%d..%d, want 0..2", chunks[0].TurnStart, chunks[0].TurnEnd)
	}
	if len(chunks[0].Messages) != 4 {
		t.Fatalf("len(chunk0.Messages)=%d, want 4", len(chunks[0].Messages))
	}
	if chunks[1].TurnStart != 2 || chunks[1].TurnEnd != 3 {
		t.Fatalf("chunk1 turns=%d..%d, want 2..3", chunks[1].TurnStart, chunks[1].TurnEnd)
	}
	if len(chunks[1].Messages) != 2 {
		t.Fatalf("len(chunk1.Messages)=%d, want 2", len(chunks[1].Messages))
	}
}

func TestChunkThread_WritesFilesWithTimestampPrefix(t *testing.T) {
	t.Parallel()

	ct := 1707142860.0
	thread := SimplifiedConversation{
		ConversationID: "c1",
		Title:          "t",
		CreateTime:     &ct,
		Messages: []SimplifiedMessage{
			{Role: "user", Text: "u1"},
			{Role: "assistant", Text: "a1"},
			{Role: "user", Text: "u2"},
			{Role: "assistant", Text: "a2"},
		},
	}

	inPath := filepath.Join(t.TempDir(), "thread.json")
	b, err := json.Marshal(thread)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(inPath, b, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	outDir := filepath.Join(t.TempDir(), "chunks")
	written, err := ChunkThread(context.Background(), inPath, fakeDecider{breakpoints: []int{1}}, 20, ChunkOptions{
		OutputDir: outDir,
		Pretty:    true,
	})
	if err != nil {
		t.Fatalf("ChunkThread: %v", err)
	}
	if len(written) != 2 {
		t.Fatalf("len(written)=%d, want 2", len(written))
	}

	// Ensure files exist.
	for _, p := range written {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("stat %s: %v", p, err)
		}
	}
}
