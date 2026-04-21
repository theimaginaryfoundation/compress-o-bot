package pipeline

import (
	"context"
	"io"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
)

// SplitArchive reads a conversations JSON export from r and returns all conversations in memory.
// The input format is identical to what SplitConversationArchive accepts on disk.
func SplitArchive(ctx context.Context, r io.Reader, opts migration.ReaderSplitOptions) ([]migration.SimplifiedConversation, error) {
	return migration.SplitConversationArchiveFromReader(ctx, r, opts)
}
