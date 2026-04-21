package pipeline

import (
	"context"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
)

// ChunkThread segments an in-memory conversation into chunks using decider to pick breakpoints.
// It is the in-memory equivalent of migration.ChunkThread.
func ChunkThread(ctx context.Context, thread migration.SimplifiedConversation, decider migration.BreakpointDecider, targetTurnsPerChunk int) ([]migration.Chunk, error) {
	return migration.ChunkThreadInMemory(ctx, thread, decider, targetTurnsPerChunk)
}
