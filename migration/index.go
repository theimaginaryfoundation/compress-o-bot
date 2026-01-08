package migration

import (
	"strings"
)

// BuildIndexRecord creates a stable index row for a chunk + its summary.
func BuildIndexRecord(chunk Chunk, chunkPath string, summary ChunkSummary, summaryPath string) IndexRecord {
	return IndexRecord{
		ConversationID: chunk.ConversationID,
		ThreadStart:    chunk.ThreadStart,
		ChunkNumber:    chunk.ChunkNumber,
		TurnStart:      chunk.TurnStart,
		TurnEnd:        chunk.TurnEnd,
		ChunkPath:      chunkPath,
		SummaryPath:    summaryPath,
		Summary:        strings.TrimSpace(summary.Summary),
		Tags:           dedupeStrings(summary.Tags),
		Terms:          dedupeStrings(summary.Terms),
	}
}

func dedupeStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		key := strings.ToLower(s)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, s)
	}
	return out
}
