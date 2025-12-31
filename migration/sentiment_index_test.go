package migration

import "testing"

func TestBuildThreadSentimentIndexRecord_TrimsAndDedupes(t *testing.T) {
	t.Parallel()

	ts := ThreadSentimentSummary{
		ConversationID:     "c1",
		EmotionalSummary:   " hi ",
		DominantEmotions:   []string{"Joy", "joy"},
		PresentEmotions:    []string{"Playful"},
		RememberedEmotions: []string{},
		Themes:             []string{"Trust", "trust"},
	}

	rec := BuildThreadSentimentIndexRecord(ts, "x.json")
	if rec.EmotionalSummary != "hi" {
		t.Fatalf("EmotionalSummary=%q", rec.EmotionalSummary)
	}
	if len(rec.DominantEmotions) != 1 {
		t.Fatalf("DominantEmotions=%v", rec.DominantEmotions)
	}
	if len(rec.Themes) != 1 {
		t.Fatalf("Themes=%v", rec.Themes)
	}
}
