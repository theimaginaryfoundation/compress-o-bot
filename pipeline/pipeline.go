package pipeline

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"sync"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
)

// Run executes the full split → chunk → summarize → rollup pipeline in memory.
// No files are written to disk. The OpenAI client on opts is used to construct
// the default Summarizer/ThreadRolluper/BreakpointDecider implementations.
func Run(ctx context.Context, conversationsJSON io.Reader, opts Options) (*Result, error) {
	if opts.OpenAIClient == nil {
		return nil, errors.New("pipeline.Run: OpenAIClient is nil")
	}
	if opts.SemanticModel == "" {
		return nil, errors.New("pipeline.Run: SemanticModel is empty")
	}
	if opts.SentimentModel == "" {
		opts.SentimentModel = opts.SemanticModel
	}

	sentimentInstructions := ComposeSentimentInstructions(opts.SentimentPromptHeader)
	summarizer := &OpenAISummarizer{
		Client:                opts.OpenAIClient,
		Model:                 opts.SemanticModel,
		SentimentModel:        opts.SentimentModel,
		SentimentInstructions: sentimentInstructions,
	}
	rolluper := &OpenAIThreadRolluper{Client: opts.OpenAIClient, Model: opts.SemanticModel}
	sentRolluper := &OpenAIThreadSentimentRolluper{Client: opts.OpenAIClient, Model: opts.SentimentModel}
	decider := &OpenAIBreakpointDecider{Client: opts.OpenAIClient, Model: opts.SemanticModel}
	return runWithDeps(ctx, conversationsJSON, opts, summarizer, rolluper, sentRolluper, decider)
}

// runWithDeps is the pure dependency-injected pipeline core. Exposed for tests.
func runWithDeps(
	ctx context.Context,
	conversationsJSON io.Reader,
	opts Options,
	summarizer Summarizer,
	rolluper ThreadRolluper,
	sentRolluper ThreadSentimentRolluper,
	decider migration.BreakpointDecider,
) (*Result, error) {
	if ctx == nil {
		return nil, errors.New("pipeline.Run: ctx is nil")
	}
	if conversationsJSON == nil {
		return nil, errors.New("pipeline.Run: conversationsJSON is nil")
	}
	if summarizer == nil {
		return nil, errors.New("pipeline.Run: summarizer is nil")
	}
	if rolluper == nil {
		return nil, errors.New("pipeline.Run: rolluper is nil")
	}
	if sentRolluper == nil {
		return nil, errors.New("pipeline.Run: sentRolluper is nil")
	}
	if decider == nil {
		return nil, errors.New("pipeline.Run: decider is nil")
	}
	if opts.TargetTurnsPerChunk <= 0 {
		opts.TargetTurnsPerChunk = 20
	}
	if opts.Concurrency <= 0 {
		opts.Concurrency = 1
	}
	if opts.MaxTranscriptChars <= 0 {
		opts.MaxTranscriptChars = 80_000
	}
	if opts.RetryMaxTranscriptChars <= 0 {
		opts.RetryMaxTranscriptChars = 40_000
	}

	// 1. Split
	threads, err := migration.SplitConversationArchiveFromReader(ctx, conversationsJSON, migration.ReaderSplitOptions{})
	if err != nil {
		return nil, fmt.Errorf("pipeline.Run: split: %w", err)
	}

	// 2. Chunk
	chunksMap := make(map[string][]migration.Chunk, len(threads))
	for _, thread := range threads {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		chunks, err := migration.ChunkThreadInMemory(ctx, thread, decider, opts.TargetTurnsPerChunk)
		if err != nil {
			return nil, fmt.Errorf("pipeline.Run: chunk thread %q: %w", thread.ConversationID, err)
		}
		chunksMap[thread.ConversationID] = chunks
	}

	// Flatten all chunks for batch processing.
	type indexedChunk struct {
		convID string
		chunk  migration.Chunk
	}
	var allChunks []indexedChunk
	for _, thread := range threads {
		for _, ch := range chunksMap[thread.ConversationID] {
			allChunks = append(allChunks, indexedChunk{convID: thread.ConversationID, chunk: ch})
		}
	}

	// 3. Summarize (batched for glossary chaining + concurrency)
	glossary := opts.InputGlossary
	if len(glossary.Entries) == 0 {
		glossary = migration.Glossary{Version: 1, Entries: []migration.GlossaryEntry{}}
	} else if glossary.Version == 0 {
		glossary.Version = 1
	}

	chunkSummariesMap := make(map[string][]migration.ChunkSummary)
	chunkSentimentsMap := make(map[string][]migration.ChunkSentimentSummary)

	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = len(allChunks)
	}
	if batchSize == 0 {
		batchSize = 1
	}

	for bstart := 0; bstart < len(allChunks); bstart += batchSize {
		bend := bstart + batchSize
		if bend > len(allChunks) {
			bend = len(allChunks)
		}
		batch := allChunks[bstart:bend]
		glossaryExcerpt := GlossaryForPrompt(glossary, opts.GlossaryMaxTerms)

		type batchResult struct {
			convID   string
			semantic migration.ChunkSummary
			sentiment migration.ChunkSentimentSummary
			additions []migration.GlossaryAddition
			seenAt    *float64
		}

		sem := make(chan struct{}, opts.Concurrency)
		resultsCh := make(chan batchResult, len(batch))
		errCh := make(chan error, len(batch))
		var wg sync.WaitGroup

		spawnLoop:
		for _, ic := range batch {
			select {
			case <-ctx.Done():
				errCh <- ctx.Err()
				break spawnLoop
			case sem <- struct{}{}:
			}
			wg.Add(1)
			ic := ic
			go func() {
				defer wg.Done()
				defer func() { <-sem }()

				defaultOpts := PromptOptions{MaxTranscriptChars: opts.MaxTranscriptChars, IncludeToolText: true}
				retryOpts := PromptOptions{MaxTranscriptChars: opts.RetryMaxTranscriptChars, IncludeToolText: false}

				semantic, additions, err := summarizer.SummarizeChunk(ctx, ic.chunk, glossaryExcerpt, defaultOpts)
				if err != nil {
					semantic, additions, err = summarizer.SummarizeChunk(ctx, ic.chunk, glossaryExcerpt, retryOpts)
					if err != nil {
						errCh <- fmt.Errorf("summarize chunk %s/%d: %w", ic.convID, ic.chunk.ChunkNumber, err)
						return
					}
				}

				sentiment, err := summarizer.SummarizeChunkSentiment(ctx, ic.chunk, glossaryExcerpt, defaultOpts)
				if err != nil {
					sentiment, err = summarizer.SummarizeChunkSentiment(ctx, ic.chunk, glossaryExcerpt, retryOpts)
					if err != nil {
						errCh <- fmt.Errorf("summarize sentiment chunk %s/%d: %w", ic.convID, ic.chunk.ChunkNumber, err)
						return
					}
				}

				resultsCh <- batchResult{
					convID:    ic.convID,
					semantic:  semantic,
					sentiment: sentiment,
					additions: additions,
					seenAt:    ic.chunk.ThreadStart,
				}
			}()
		}

		wg.Wait()
		close(errCh)
		close(resultsCh)

		var errs []error
		for err := range errCh {
			if err != nil {
				errs = append(errs, err)
			}
		}
		if len(errs) > 0 {
			return nil, errors.Join(errs...)
		}

		for res := range resultsCh {
			chunkSummariesMap[res.convID] = append(chunkSummariesMap[res.convID], res.semantic)
			chunkSentimentsMap[res.convID] = append(chunkSentimentsMap[res.convID], res.sentiment)
			migration.MergeGlossary(&glossary, res.additions, res.seenAt)
		}
	}

	if opts.GlossaryMinCount > 1 {
		migration.CullGlossary(&glossary, opts.GlossaryMinCount)
	}

	// Sort summaries by chunk number within each thread.
	for k := range chunkSummariesMap {
		sort.Slice(chunkSummariesMap[k], func(i, j int) bool {
			return chunkSummariesMap[k][i].ChunkNumber < chunkSummariesMap[k][j].ChunkNumber
		})
	}
	for k := range chunkSentimentsMap {
		sort.Slice(chunkSentimentsMap[k], func(i, j int) bool {
			return chunkSentimentsMap[k][i].ChunkNumber < chunkSentimentsMap[k][j].ChunkNumber
		})
	}

	// 4. Rollup
	glossaryExcerpt := GlossaryForPrompt(glossary, opts.GlossaryMaxTerms)

	threadSummariesMap := make(map[string]migration.ThreadSummary, len(threads))
	threadSentimentsMap := make(map[string]migration.ThreadSentimentSummary, len(threads))

	for _, thread := range threads {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		convID := thread.ConversationID

		chunks := chunkSummariesMap[convID]
		ts, err := rollupThread(ctx, convID, chunks, rolluper, opts.MaxChunksPerThread, glossaryExcerpt)
		if err != nil {
			return nil, fmt.Errorf("pipeline.Run: rollup thread %q: %w", convID, err)
		}
		threadSummariesMap[convID] = ts

		sentChunks := chunkSentimentsMap[convID]
		tss, err := rollupSentimentThread(ctx, convID, sentChunks, sentRolluper, opts.MaxChunksPerThread, glossaryExcerpt)
		if err != nil {
			return nil, fmt.Errorf("pipeline.Run: sentiment rollup thread %q: %w", convID, err)
		}
		threadSentimentsMap[convID] = tss
	}

	return &Result{
		Threads:          threads,
		Chunks:           chunksMap,
		ChunkSummaries:   chunkSummariesMap,
		ChunkSentiments:  chunkSentimentsMap,
		ThreadSummaries:  threadSummariesMap,
		ThreadSentiments: threadSentimentsMap,
		Glossary:         glossary,
	}, nil
}

func rollupThread(
	ctx context.Context,
	convID string,
	chunks []migration.ChunkSummary,
	rolluper ThreadRolluper,
	maxChunksPerThread int,
	glossaryExcerpt string,
) (migration.ThreadSummary, error) {
	if maxChunksPerThread <= 0 || len(chunks) <= maxChunksPerThread {
		return rolluper.Rollup(ctx, convID, chunks, glossaryExcerpt)
	}

	parts := SplitChunksIntoWindows(chunks, maxChunksPerThread)
	partSummaries := make([]migration.ThreadSummary, 0, len(parts))
	for _, win := range parts {
		ps, err := rolluper.Rollup(ctx, convID, win, glossaryExcerpt)
		if err != nil {
			return migration.ThreadSummary{}, err
		}
		partSummaries = append(partSummaries, ps)
	}
	return rolluper.RollupFromThreadSummaries(ctx, convID, partSummaries, glossaryExcerpt)
}

func rollupSentimentThread(
	ctx context.Context,
	convID string,
	chunks []migration.ChunkSentimentSummary,
	rolluper ThreadSentimentRolluper,
	maxChunksPerThread int,
	glossaryExcerpt string,
) (migration.ThreadSentimentSummary, error) {
	if maxChunksPerThread <= 0 || len(chunks) <= maxChunksPerThread {
		return rolluper.Rollup(ctx, convID, chunks, glossaryExcerpt)
	}

	parts := SplitChunksIntoWindows(chunks, maxChunksPerThread)
	partSummaries := make([]migration.ThreadSentimentSummary, 0, len(parts))
	for _, win := range parts {
		ps, err := rolluper.Rollup(ctx, convID, win, glossaryExcerpt)
		if err != nil {
			return migration.ThreadSentimentSummary{}, err
		}
		partSummaries = append(partSummaries, ps)
	}
	return rolluper.RollupFromThreadSentimentSummaries(ctx, convID, partSummaries, glossaryExcerpt)
}
