package migration

import "testing"

func TestBuildIndexRecord_Dedupes(t *testing.T) {
	t.Parallel()

	chunk := Chunk{
		ConversationID: "c1",
		ChunkNumber:    1,
		TurnStart:      0,
		TurnEnd:        2,
	}
	sum := ChunkSummary{
		Summary: " hi ",
		Tags:    []string{"Foo", "foo", "  ", "Bar"},
		Terms:   []string{"Vix", "vix"},
	}
	rec := BuildIndexRecord(chunk, "c.json", sum, "s.json")
	if rec.Summary != "hi" {
		t.Fatalf("Summary=%q, want hi", rec.Summary)
	}
	if len(rec.Tags) != 2 {
		t.Fatalf("Tags=%v, want 2", rec.Tags)
	}
	if len(rec.Terms) != 1 {
		t.Fatalf("Terms=%v, want 1", rec.Terms)
	}
}
