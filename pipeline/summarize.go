package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/fileutils"
)

// Summarizer produces chunk-level summaries from a Chunk and an optional glossary excerpt.
type Summarizer interface {
	SummarizeChunk(ctx context.Context, chunk migration.Chunk, glossaryExcerpt string, opts PromptOptions) (migration.ChunkSummary, []migration.GlossaryAddition, error)
	SummarizeChunkSentiment(ctx context.Context, chunk migration.Chunk, glossaryExcerpt string, opts PromptOptions) (migration.ChunkSentimentSummary, error)
}

// summarizeResponse is the raw JSON shape returned by the summarizer model.
type summarizeResponse struct {
	Summary           string                       `json:"summary"`
	KeyPoints         []string                     `json:"key_points"`
	Tags              []string                     `json:"tags"`
	Terms             []string                     `json:"terms"`
	GlossaryAdditions []migration.GlossaryAddition `json:"glossary_additions"`
}

// summarizeSentimentResponse is the raw JSON shape returned by the sentiment summarizer model.
type summarizeSentimentResponse struct {
	EmotionalSummary   string   `json:"emotional_summary"`
	DominantEmotions   []string `json:"dominant_emotions"`
	RememberedEmotions []string `json:"remembered_emotions"`
	PresentEmotions    []string `json:"present_emotions"`
	EmotionalTensions  []string `json:"emotional_tensions"`
	RelationalShift    string   `json:"relational_shift"`
	EmotionalArc       string   `json:"emotional_arc"`
	Themes             []string `json:"themes"`
	SymbolsOrMetaphors []string `json:"symbols_or_metaphors"`
	ResonanceNotes     string   `json:"resonance_notes"`
	ToneMarkers        []string `json:"tone_markers"`
}

// ComposeSentimentInstructions builds the full sentiment model instructions by combining
// a (possibly custom) header with the required security/output tail.
func ComposeSentimentInstructions(header string) string {
	header = strings.TrimSpace(header)
	if header == "" {
		header = strings.TrimSpace(DefaultSentimentPromptHeader)
	}
	tail := strings.TrimSpace(SentimentPromptRequiredTail)
	return header + "\n\n" + tail
}

// BuildChunkPromptInput builds the user-turn text sent to the summarizer model.
func BuildChunkPromptInput(chunk migration.Chunk, glossaryExcerpt string, opt PromptOptions) string {
	var b strings.Builder
	fmt.Fprintf(&b, "chunk_metadata:\nconversation_id=%s\nchunk_number=%d\nturn_range=%d..%d\n\n",
		chunk.ConversationID, chunk.ChunkNumber, chunk.TurnStart, chunk.TurnEnd)

	if glossaryExcerpt != "" {
		b.WriteString("glossary:\n")
		b.WriteString(glossaryExcerpt)
		b.WriteString("\n")
	}

	b.WriteString("transcript:\n")
	maxTranscriptChars := opt.MaxTranscriptChars
	if maxTranscriptChars <= 0 {
		maxTranscriptChars = 80_000
	}
	total := 0
	for _, m := range chunk.Messages {
		role := m.Role
		if role == "" {
			role = "unknown"
		}
		name := ""
		if m.Name != "" {
			name = ":" + m.Name
		}

		var line string
		if !opt.IncludeToolText && role == "tool" {
			desc := strings.TrimSpace(m.ContentType)
			if desc == "" {
				desc = "tool"
			}
			parts := []string{"[tool", m.Name, desc, m.Title, m.URL}
			line = strings.TrimSpace(strings.Join(parts, " "))
		} else if strings.TrimSpace(m.Text) != "" {
			line = m.Text
		} else if m.URL != "" || m.Title != "" {
			line = strings.TrimSpace(strings.Join([]string{m.Title, m.URL}, " "))
		} else {
			line = "[" + strings.TrimSpace(m.ContentType) + "]"
		}
		line = fileutils.Truncate(line, 2000)
		row := fmt.Sprintf("- %s%s: %s\n", role, name, fileutils.SanitizeNewlines(line))
		if total+len(row) > maxTranscriptChars {
			b.WriteString("... [transcript truncated]\n")
			break
		}
		b.WriteString(row)
		total += len(row)
	}
	return b.String()
}
