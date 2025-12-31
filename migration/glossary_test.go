package migration

import (
	"testing"
)

func TestMergeGlossary_AddsAndIncrementsAndPrefersLongerDefinition(t *testing.T) {
	t.Parallel()

	g := Glossary{
		Version: 1,
		Entries: []GlossaryEntry{
			{Term: "Vix", Definition: "short", Count: 2},
		},
	}

	ts := 123.0
	terms := MergeGlossary(&g, []GlossaryAddition{
		{Term: "vix", Definition: "a longer, better definition"},
		{Term: "Sparky", Definition: "companion agent"},
		{Term: "sparky", Definition: "duplicate, should dedupe in one merge call"},
	}, &ts)

	if len(terms) != 2 {
		t.Fatalf("terms=%v, want 2 terms", terms)
	}

	// Find Vix.
	var vix *GlossaryEntry
	var sparky *GlossaryEntry
	for i := range g.Entries {
		switch g.Entries[i].Term {
		case "Vix":
			vix = &g.Entries[i]
		case "Sparky":
			sparky = &g.Entries[i]
		}
	}
	if vix == nil {
		t.Fatalf("missing Vix entry")
	}
	if vix.Count != 3 {
		t.Fatalf("Vix.Count=%d, want 3", vix.Count)
	}
	if vix.Definition != "a longer, better definition" {
		t.Fatalf("Vix.Definition=%q", vix.Definition)
	}
	if sparky == nil {
		t.Fatalf("missing Sparky entry")
	}
	if sparky.Count != 1 {
		t.Fatalf("Sparky.Count=%d, want 1", sparky.Count)
	}
}

func TestCullGlossary_RemovesInfrequent(t *testing.T) {
	t.Parallel()

	g := Glossary{
		Version: 1,
		Entries: []GlossaryEntry{
			{Term: "A", Count: 1},
			{Term: "B", Count: 2},
		},
	}
	CullGlossary(&g, 2)
	if len(g.Entries) != 1 || g.Entries[0].Term != "B" {
		t.Fatalf("entries=%v, want only B", g.Entries)
	}
}
