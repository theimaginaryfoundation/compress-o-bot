package migration

import "testing"

func TestBuildThreadIndexRecord_Dedupes(t *testing.T) {
	t.Parallel()

	ts := ThreadSummary{
		ConversationID: "c1",
		Summary:        " hi ",
		Tags:           []string{"Foo", "foo", "Bar"},
		Terms:          []string{"Vix", "vix"},
	}
	rec := BuildThreadIndexRecord(ts, "t.summary.json")
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
