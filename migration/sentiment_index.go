package migration

import "strings"

// BuildThreadSentimentIndexRecord creates an index row for a thread sentiment summary.
func BuildThreadSentimentIndexRecord(ts ThreadSentimentSummary, path string) ThreadSentimentIndexRecord {
	return ThreadSentimentIndexRecord{
		ConversationID:             ts.ConversationID,
		ThreadStart:                ts.ThreadStart,
		Title:                      ts.Title,
		ThreadSentimentSummaryPath: path,
		EmotionalSummary:           strings.TrimSpace(ts.EmotionalSummary),
		DominantEmotions:           dedupeStrings(ts.DominantEmotions),
		RememberedEmotions:         dedupeStrings(ts.RememberedEmotions),
		PresentEmotions:            dedupeStrings(ts.PresentEmotions),
		EmotionalTensions:          dedupeStrings(ts.EmotionalTensions),
		RelationalShift:            strings.TrimSpace(ts.RelationalShift),
		EmotionalArc:               strings.TrimSpace(ts.EmotionalArc),
		Themes:                     dedupeStrings(ts.Themes),
	}
}
