package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/fileutils"
)

// ThreadRolluper produces thread-level semantic summaries from chunk summaries.
type ThreadRolluper interface {
	Rollup(ctx context.Context, conversationID string, chunks []migration.ChunkSummary, glossaryExcerpt string) (migration.ThreadSummary, error)
	RollupFromThreadSummaries(ctx context.Context, conversationID string, parts []migration.ThreadSummary, glossaryExcerpt string) (migration.ThreadSummary, error)
}

// ThreadSentimentRolluper produces thread-level sentiment summaries from chunk sentiment summaries.
type ThreadSentimentRolluper interface {
	Rollup(ctx context.Context, conversationID string, chunks []migration.ChunkSentimentSummary, glossaryExcerpt string) (migration.ThreadSentimentSummary, error)
	RollupFromThreadSentimentSummaries(ctx context.Context, conversationID string, parts []migration.ThreadSentimentSummary, glossaryExcerpt string) (migration.ThreadSentimentSummary, error)
}

// rollupResponse is the raw JSON shape returned by the thread rollup model.
type rollupResponse struct {
	Title       string   `json:"title"`
	ThreadStart *float64 `json:"thread_start_time"`
	Summary     string   `json:"summary"`
	KeyPoints   []string `json:"key_points"`
	Tags        []string `json:"tags"`
	Terms       []string `json:"terms"`
}

// sentimentRollupResponse is the raw JSON shape returned by the thread sentiment rollup model.
type sentimentRollupResponse struct {
	Title       string   `json:"title"`
	ThreadStart *float64 `json:"thread_start_time"`

	EmotionalSummary string `json:"emotional_summary"`

	DominantEmotions   []string `json:"dominant_emotions"`
	RememberedEmotions []string `json:"remembered_emotions"`
	PresentEmotions    []string `json:"present_emotions"`
	EmotionalTensions  []string `json:"emotional_tensions"`

	RelationalShift string `json:"relational_shift"`

	EmotionalArc       string   `json:"emotional_arc"`
	Themes             []string `json:"themes"`
	SymbolsOrMetaphors []string `json:"symbols_or_metaphors"`

	ResonanceNotes string   `json:"resonance_notes"`
	ToneMarkers    []string `json:"tone_markers"`
}

// SplitChunksIntoWindows splits a slice into windows of at most max items each.
// If max <= 0 or len(in) <= max, returns a single-element slice containing the whole input.
func SplitChunksIntoWindows[T any](in []T, max int) [][]T {
	if max <= 0 || len(in) <= max {
		return [][]T{in}
	}
	out := make([][]T, 0, (len(in)+max-1)/max)
	for start := 0; start < len(in); start += max {
		end := start + max
		if end > len(in) {
			end = len(in)
		}
		out = append(out, in[start:end])
	}
	return out
}

// ---- Input builders ----

func buildThreadRollupInput(conversationID string, chunks []migration.ChunkSummary, glossaryExcerpt string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "conversation_id=%s\nchunks=%d\n\n", conversationID, len(chunks))

	if glossaryExcerpt != "" {
		b.WriteString("glossary:\n")
		b.WriteString(glossaryExcerpt)
		b.WriteString("\n")
	}

	b.WriteString("chunk_summaries:\n")
	const maxChars = 80_000
	total := 0
	for _, c := range chunks {
		row := fmt.Sprintf("- chunk=%d turn_range=%d..%d\n  summary=%s\n  key_points=%s\n  tags=%s\n  terms=%s\n",
			c.ChunkNumber, c.TurnStart, c.TurnEnd,
			truncate(c.Summary, 1200),
			truncate(strings.Join(c.KeyPoints, "; "), 1800),
			truncate(strings.Join(c.Tags, ", "), 600),
			truncate(strings.Join(c.Terms, ", "), 600),
		)
		if total+len(row) > maxChars {
			b.WriteString("... [chunk_summaries truncated]\n")
			break
		}
		b.WriteString(row)
		total += len(row)
	}
	return b.String()
}

func buildThreadRollupMergeInput(conversationID string, parts []migration.ThreadSummary, glossaryExcerpt string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "conversation_id=%s\npartial_rollups=%d\n\n", conversationID, len(parts))

	if glossaryExcerpt != "" {
		b.WriteString("glossary:\n")
		b.WriteString(glossaryExcerpt)
		b.WriteString("\n")
	}

	b.WriteString("partial_thread_summaries:\n")
	const maxChars = 60_000
	total := 0
	for i, p := range parts {
		row := fmt.Sprintf("- part=%d title=%s thread_start_time=%v\n  summary=%s\n  key_points=%s\n  tags=%s\n  terms=%s\n",
			i+1,
			truncate(p.Title, 80),
			p.ThreadStart,
			truncate(p.Summary, 2500),
			truncate(strings.Join(p.KeyPoints, "; "), 2500),
			truncate(strings.Join(p.Tags, ", "), 1200),
			truncate(strings.Join(p.Terms, ", "), 800),
		)
		if total+len(row) > maxChars {
			b.WriteString("... [partial_thread_summaries truncated]\n")
			break
		}
		b.WriteString(row)
		total += len(row)
	}
	return b.String()
}

func buildThreadSentimentRollupInput(conversationID string, chunks []migration.ChunkSentimentSummary, glossaryExcerpt string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "conversation_id=%s\nchunks=%d\n\n", conversationID, len(chunks))

	if glossaryExcerpt != "" {
		b.WriteString("glossary:\n")
		b.WriteString(glossaryExcerpt)
		b.WriteString("\n")
	}

	b.WriteString("chunk_sentiment_summaries:\n")
	const maxChars = 80_000
	total := 0
	for _, c := range chunks {
		row := fmt.Sprintf("- chunk=%d turn_range=%d..%d\n  emotional_summary=%s\n  dominant_emotions=%s\n  remembered_emotions=%s\n  present_emotions=%s\n  emotional_tensions=%s\n  relational_shift=%s\n  emotional_arc=%s\n  themes=%s\n  symbols_or_metaphors=%s\n",
			c.ChunkNumber, c.TurnStart, c.TurnEnd,
			truncate(c.EmotionalSummary, 1200),
			truncate(strings.Join(c.DominantEmotions, ", "), 600),
			truncate(strings.Join(c.RememberedEmotions, ", "), 600),
			truncate(strings.Join(c.PresentEmotions, ", "), 600),
			truncate(strings.Join(c.EmotionalTensions, ", "), 600),
			truncate(c.RelationalShift, 600),
			truncate(c.EmotionalArc, 600),
			truncate(strings.Join(c.Themes, ", "), 800),
			truncate(strings.Join(c.SymbolsOrMetaphors, ", "), 800),
		)
		if total+len(row) > maxChars {
			b.WriteString("... [chunk_sentiment_summaries truncated]\n")
			break
		}
		b.WriteString(row)
		total += len(row)
	}
	return b.String()
}

func buildThreadSentimentRollupMergeInput(conversationID string, parts []migration.ThreadSentimentSummary, glossaryExcerpt string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "conversation_id=%s\npartial_rollups=%d\n\n", conversationID, len(parts))

	if glossaryExcerpt != "" {
		b.WriteString("glossary:\n")
		b.WriteString(glossaryExcerpt)
		b.WriteString("\n")
	}

	b.WriteString("partial_thread_sentiment_summaries:\n")
	const maxChars = 60_000
	total := 0
	for i, p := range parts {
		row := fmt.Sprintf("- part=%d title=%s thread_start_time=%v\n  emotional_summary=%s\n  dominant_emotions=%s\n  remembered_emotions=%s\n  present_emotions=%s\n  emotional_tensions=%s\n  relational_shift=%s\n  emotional_arc=%s\n  themes=%s\n  symbols_or_metaphors=%s\n",
			i+1,
			truncate(p.Title, 80),
			p.ThreadStart,
			truncate(p.EmotionalSummary, 2500),
			truncate(strings.Join(p.DominantEmotions, ", "), 1200),
			truncate(strings.Join(p.RememberedEmotions, ", "), 1200),
			truncate(strings.Join(p.PresentEmotions, ", "), 1200),
			truncate(strings.Join(p.EmotionalTensions, ", "), 1200),
			truncate(p.RelationalShift, 600),
			truncate(p.EmotionalArc, 1000),
			truncate(strings.Join(p.Themes, ", "), 1500),
			truncate(strings.Join(p.SymbolsOrMetaphors, ", "), 1500),
		)
		if total+len(row) > maxChars {
			b.WriteString("... [partial_thread_sentiment_summaries truncated]\n")
			break
		}
		b.WriteString(row)
		total += len(row)
	}
	return b.String()
}

// ---- Min thread-start helpers ----

func minThreadStartFromChunkSummaries(chunks []migration.ChunkSummary) *float64 {
	var (
		min float64
		ok  bool
	)
	for _, c := range chunks {
		if c.ThreadStart == nil {
			continue
		}
		if !ok || *c.ThreadStart < min {
			min = *c.ThreadStart
			ok = true
		}
	}
	if !ok {
		return nil
	}
	return float64Ptr(min)
}

func minThreadStartFromChunkSentimentSummaries(chunks []migration.ChunkSentimentSummary) *float64 {
	var (
		min float64
		ok  bool
	)
	for _, c := range chunks {
		if c.ThreadStart == nil {
			continue
		}
		if !ok || *c.ThreadStart < min {
			min = *c.ThreadStart
			ok = true
		}
	}
	if !ok {
		return nil
	}
	return float64Ptr(min)
}

func minThreadStartFromThreadSummaries(parts []migration.ThreadSummary) *float64 {
	var (
		min float64
		ok  bool
	)
	for _, p := range parts {
		if p.ThreadStart == nil {
			continue
		}
		if !ok || *p.ThreadStart < min {
			min = *p.ThreadStart
			ok = true
		}
	}
	if !ok {
		return nil
	}
	return float64Ptr(min)
}

func minThreadStartFromThreadSentimentSummaries(parts []migration.ThreadSentimentSummary) *float64 {
	var (
		min float64
		ok  bool
	)
	for _, p := range parts {
		if p.ThreadStart == nil {
			continue
		}
		if !ok || *p.ThreadStart < min {
			min = *p.ThreadStart
			ok = true
		}
	}
	if !ok {
		return nil
	}
	return float64Ptr(min)
}

func float64Ptr(v float64) *float64 { return &v }

func truncate(s string, max int) string {
	return fileutils.Truncate(s, max)
}
