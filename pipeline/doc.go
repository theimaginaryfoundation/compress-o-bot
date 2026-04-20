// Package pipeline provides in-memory implementations of the four compress-o-bot processing stages:
// split, chunk, summarize, and rollup. It is designed to be imported by services (e.g. a webservice)
// that need to run the full pipeline without writing any intermediate results to disk.
//
// Quick start:
//
//	result, err := pipeline.Run(ctx, conversationsJSONReader, pipeline.Options{
//	    OpenAIClient:        myClient,
//	    SemanticModel:       "gpt-4o-mini",
//	    TargetTurnsPerChunk: 20,
//	    Concurrency:         4,
//	})
//
// For partial pipelines or custom stage injection, call the individual stage functions directly:
// SplitArchive, ChunkThread, and use the Summarizer / ThreadRolluper / ThreadSentimentRolluper
// interfaces with your own implementations or the provided OpenAI defaults.
package pipeline
