package migration

import "strings"

// BuildThreadIndexRecord creates a stable index row for a thread summary file.
func BuildThreadIndexRecord(ts ThreadSummary, threadSummaryPath string) ThreadIndexRecord {
	return ThreadIndexRecord{
		ConversationID:    ts.ConversationID,
		ThreadStart:       ts.ThreadStart,
		Title:             ts.Title,
		ThreadSummaryPath: threadSummaryPath,
		Summary:           strings.TrimSpace(ts.Summary),
		Tags:              dedupeStrings(ts.Tags),
		Terms:             dedupeStrings(ts.Terms),
	}
}
