package migration

// ChunkSentimentSummary is the model-produced sentiment artifact for one chunk file.
// This mirrors the shape produced by cmd/chunk-summarizer for *.sentiment.summary.json.
type ChunkSentimentSummary struct {
	ConversationID string   `json:"conversation_id"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`
	ChunkNumber    int      `json:"chunk_number"`
	TurnStart      int      `json:"turn_start"`
	TurnEnd        int      `json:"turn_end"`

	EmotionalSummary string `json:"emotional_summary"`

	DominantEmotions   []string `json:"dominant_emotions"`
	RememberedEmotions []string `json:"remembered_emotions"`
	PresentEmotions    []string `json:"present_emotions"`
	EmotionalTensions  []string `json:"emotional_tensions"`

	RelationalShift string `json:"relational_shift"`

	EmotionalArc       string   `json:"emotional_arc"`
	Themes             []string `json:"themes"`
	SymbolsOrMetaphors []string `json:"symbols_or_metaphors"`

	ResonanceNotes string   `json:"resonance_notes,omitempty"`
	ToneMarkers    []string `json:"tone_markers,omitempty"`
}

// ThreadSentimentSummary is the model-produced sentiment artifact for an entire thread, aggregated from chunk sentiment summaries.
type ThreadSentimentSummary struct {
	ConversationID string   `json:"conversation_id"`
	Title          string   `json:"title,omitempty"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`

	EmotionalSummary string `json:"emotional_summary"`

	DominantEmotions   []string `json:"dominant_emotions"`
	RememberedEmotions []string `json:"remembered_emotions"`
	PresentEmotions    []string `json:"present_emotions"`
	EmotionalTensions  []string `json:"emotional_tensions"`

	RelationalShift string `json:"relational_shift"`

	EmotionalArc       string   `json:"emotional_arc"`
	Themes             []string `json:"themes"`
	SymbolsOrMetaphors []string `json:"symbols_or_metaphors"`

	ResonanceNotes string   `json:"resonance_notes,omitempty"`
	ToneMarkers    []string `json:"tone_markers,omitempty"`
}

// ThreadSentimentIndexRecord is a row mapping a thread to its sentiment rollup file.
type ThreadSentimentIndexRecord struct {
	ConversationID string   `json:"conversation_id"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`
	Title          string   `json:"title,omitempty"`

	ThreadSentimentSummaryPath string `json:"thread_sentiment_summary_path"`

	EmotionalSummary   string   `json:"emotional_summary"`
	DominantEmotions   []string `json:"dominant_emotions,omitempty"`
	RememberedEmotions []string `json:"remembered_emotions,omitempty"`
	PresentEmotions    []string `json:"present_emotions,omitempty"`
	EmotionalTensions  []string `json:"emotional_tensions,omitempty"`
	RelationalShift    string   `json:"relational_shift,omitempty"`
	EmotionalArc       string   `json:"emotional_arc,omitempty"`
	Themes             []string `json:"themes,omitempty"`
}
