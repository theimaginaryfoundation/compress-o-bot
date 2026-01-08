package migration

// Glossary is a cross-chunk evolving glossary for consistent naming and retrieval.
type Glossary struct {
	Version int             `json:"version"`
	Entries []GlossaryEntry `json:"entries"`
	Meta    map[string]any  `json:"meta,omitempty"`
}

// GlossaryEntry is one term in the glossary.
type GlossaryEntry struct {
	Term        string   `json:"term"`
	Definition  string   `json:"definition,omitempty"`
	Count       int      `json:"count"`
	FirstSeenAt *float64 `json:"first_seen_at,omitempty"`
	LastSeenAt  *float64 `json:"last_seen_at,omitempty"`
}

// ChunkSummary is the model-produced summary artifact for one chunk file.
type ChunkSummary struct {
	ConversationID string   `json:"conversation_id"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`
	ChunkNumber    int      `json:"chunk_number"`
	TurnStart      int      `json:"turn_start"`
	TurnEnd        int      `json:"turn_end"`

	// Summary is a tight prose summary (1-3 short paragraphs).
	Summary string `json:"summary"`

	// KeyPoints are bullet-style claims/facts worth retrieving later.
	KeyPoints []string `json:"key_points,omitempty"`

	// Tags are high-level topics/entities for indexing/filtering.
	Tags []string `json:"tags,omitempty"`

	// Terms are glossary terms referenced/added by this chunk (for index joins).
	Terms []string `json:"terms,omitempty"`
}

// ThreadSummary is the model-produced summary artifact for an entire thread, aggregated from chunk summaries.
type ThreadSummary struct {
	ConversationID string   `json:"conversation_id"`
	Title          string   `json:"title,omitempty"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`

	// Summary is a tight prose summary (2-6 short paragraphs) describing the whole thread.
	Summary string `json:"summary"`

	// KeyPoints are retrievable facts/decisions/claims spanning the thread.
	KeyPoints []string `json:"key_points,omitempty"`

	// Tags are high-level topics/entities for indexing/filtering.
	Tags []string `json:"tags,omitempty"`

	// Terms are glossary terms referenced/added by this thread.
	Terms []string `json:"terms,omitempty"`
}

// ThreadIndexRecord is a row in thread_index.jsonl mapping a thread to its rollup file.
type ThreadIndexRecord struct {
	ConversationID string   `json:"conversation_id"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`
	Title          string   `json:"title,omitempty"`

	ThreadSummaryPath string `json:"thread_summary_path"`

	Summary string   `json:"summary"`
	Tags    []string `json:"tags,omitempty"`
	Terms   []string `json:"terms,omitempty"`
}

// IndexRecord is a single row in index.jsonl.
type IndexRecord struct {
	ConversationID string   `json:"conversation_id"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`
	ChunkNumber    int      `json:"chunk_number"`
	TurnStart      int      `json:"turn_start"`
	TurnEnd        int      `json:"turn_end"`

	ChunkPath   string `json:"chunk_path"`
	SummaryPath string `json:"summary_path"`

	// Summary is duplicated (shortened) here for quick scanning without opening the summary file.
	Summary string `json:"summary"`

	Tags  []string `json:"tags,omitempty"`
	Terms []string `json:"terms,omitempty"`
}
