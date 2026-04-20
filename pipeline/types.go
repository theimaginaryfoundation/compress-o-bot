package pipeline

import (
	"github.com/openai/openai-go"
	"github.com/theimaginaryfoundation/compress-o-bot/migration"
)

// Type aliases expose migration types so callers only need to import this package.
type (
	SimplifiedConversation = migration.SimplifiedConversation
	SimplifiedMessage      = migration.SimplifiedMessage
	Chunk                  = migration.Chunk
	ChunkSummary           = migration.ChunkSummary
	ChunkSentimentSummary  = migration.ChunkSentimentSummary
	ThreadSummary          = migration.ThreadSummary
	ThreadSentimentSummary = migration.ThreadSentimentSummary
	Glossary               = migration.Glossary
	GlossaryEntry          = migration.GlossaryEntry
	GlossaryAddition       = migration.GlossaryAddition
	Turn                   = migration.Turn
	BreakpointDecider      = migration.BreakpointDecider
)

// PromptOptions controls prompt input construction for a single summarization call.
type PromptOptions struct {
	MaxTranscriptChars int
	IncludeToolText    bool
}

// Options configures a full pipeline run via Run.
type Options struct {
	OpenAIClient *openai.Client

	SemanticModel  string
	SentimentModel string // defaults to SemanticModel when empty

	TargetTurnsPerChunk int // turns per chunk; default 20
	MaxChunksPerThread  int // 0 = no limit
	Concurrency         int // max concurrent summarization calls; default 1
	BatchSize           int // chunks per glossary-chaining batch; 0 = all

	SentimentPromptHeader string // optional override header for sentiment summarization
	GlossaryMaxTerms      int    // max glossary terms injected into prompt; 0 = all
	GlossaryMinCount      int    // cull glossary terms with count < N at end; 0 = keep all

	// MaxTranscriptChars caps the transcript size sent on the first summarization attempt.
	// Defaults to 80_000 when <= 0.
	MaxTranscriptChars int

	// RetryMaxTranscriptChars is used on a retry if the first attempt fails. Defaults to 40_000
	// when <= 0. Tool text is also excluded on retry to reduce payload.
	RetryMaxTranscriptChars int

	InputGlossary Glossary // optional starting glossary; zero value = empty
}

// Result holds all in-memory outputs from a full pipeline run.
type Result struct {
	Threads          []SimplifiedConversation
	Chunks           map[string][]Chunk              // keyed by conversation_id
	ChunkSummaries   map[string][]ChunkSummary       // keyed by conversation_id
	ChunkSentiments  map[string][]ChunkSentimentSummary // keyed by conversation_id
	ThreadSummaries  map[string]ThreadSummary        // keyed by conversation_id
	ThreadSentiments map[string]ThreadSentimentSummary // keyed by conversation_id
	Glossary         Glossary
}
