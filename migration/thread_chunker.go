package migration

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Turn represents a user-led segment of the conversation: a user message plus any following assistant/tool/system
// messages until the next user message.
type Turn struct {
	TurnIndex         int
	StartMessageIndex int
	EndMessageIndex   int

	StartTime *float64

	UserText      string
	AssistantText string
}

// Chunk is a summarizer-ready slice of a thread.
type Chunk struct {
	ConversationID string              `json:"conversation_id"`
	Title          string              `json:"title,omitempty"`
	ThreadStart    *float64            `json:"thread_start_time,omitempty"`
	ChunkNumber    int                 `json:"chunk_number"`
	TurnStart      int                 `json:"turn_start"`
	TurnEnd        int                 `json:"turn_end"` // exclusive
	Messages       []SimplifiedMessage `json:"messages"`
}

// ChunkOptions controls how thread chunks are written.
type ChunkOptions struct {
	// OutputDir is where chunk JSON files are written.
	OutputDir string

	// OverwriteExisting controls whether existing chunk files should be overwritten.
	OverwriteExisting bool

	// Pretty controls whether each chunk JSON file is indented for readability.
	Pretty bool

	// DirMode is used when creating output directories (defaults to 0o755).
	DirMode fs.FileMode

	// FileMode is used when creating output files (defaults to 0o644).
	FileMode fs.FileMode
}

// BreakpointDecider decides where to split a thread into chunks.
// Breakpoints are expressed as turn indices (0-based) where a new chunk should start.
// Example: totalTurns=55, breakpoints=[20,40] produces chunks [0..20), [20..40), [40..55).
type BreakpointDecider interface {
	DecideBreakpoints(ctx context.Context, thread SimplifiedConversation, turns []Turn, targetTurnsPerChunk int) ([]int, error)
}

// BuildTurns groups a SimplifiedConversation into user-led turns.
func BuildTurns(thread SimplifiedConversation) []Turn {
	msgs := thread.Messages
	if len(msgs) == 0 {
		return nil
	}

	userIdxs := make([]int, 0, 64)
	for i := range msgs {
		if msgs[i].Role == "user" {
			userIdxs = append(userIdxs, i)
		}
	}

	if len(userIdxs) == 0 {
		// No explicit user messages; treat entire thread as one turn.
		return []Turn{turnFromRange(0, 0, len(msgs)-1, msgs)}
	}

	turns := make([]Turn, 0, len(userIdxs))
	for ti, start := range userIdxs {
		end := len(msgs) - 1
		if ti+1 < len(userIdxs) {
			end = userIdxs[ti+1] - 1
		}
		turns = append(turns, turnFromRange(ti, start, end, msgs))
	}
	return turns
}

func turnFromRange(turnIndex, start, end int, msgs []SimplifiedMessage) Turn {
	var startTime *float64
	if start >= 0 && start < len(msgs) {
		startTime = msgs[start].CreateTime
	}

	userText := ""
	assistantParts := make([]string, 0, 8)
	for i := start; i <= end && i < len(msgs); i++ {
		m := msgs[i]
		switch m.Role {
		case "user":
			if userText == "" {
				userText = strings.TrimSpace(m.Text)
			} else if s := strings.TrimSpace(m.Text); s != "" {
				userText = userText + "\n" + s
			}
		default:
			if s := strings.TrimSpace(m.Text); s != "" {
				assistantParts = append(assistantParts, s)
			} else if m.URL != "" || m.Title != "" {
				assistantParts = append(assistantParts, strings.TrimSpace(strings.Join([]string{m.Title, m.URL}, " ")))
			}
		}
	}

	return Turn{
		TurnIndex:         turnIndex,
		StartMessageIndex: start,
		EndMessageIndex:   end,
		StartTime:         startTime,
		UserText:          userText,
		AssistantText:     strings.Join(assistantParts, "\n"),
	}
}

// ChunkThread reads a single thread JSON file, decides breakpoints, and writes chunk files.
func ChunkThread(ctx context.Context, threadPath string, decider BreakpointDecider, targetTurnsPerChunk int, opts ChunkOptions) ([]string, error) {
	if ctx == nil {
		return nil, errors.New("ChunkThread: ctx is nil")
	}
	if threadPath == "" {
		return nil, errors.New("ChunkThread: threadPath is empty")
	}
	if decider == nil {
		return nil, errors.New("ChunkThread: decider is nil")
	}
	if targetTurnsPerChunk <= 0 {
		return nil, errors.New("ChunkThread: targetTurnsPerChunk must be > 0")
	}
	if opts.OutputDir == "" {
		return nil, errors.New("ChunkThread: opts.OutputDir is empty")
	}
	if opts.DirMode == 0 {
		opts.DirMode = 0o755
	}
	if opts.FileMode == 0 {
		opts.FileMode = 0o644
	}
	if err := os.MkdirAll(opts.OutputDir, opts.DirMode); err != nil {
		return nil, fmt.Errorf("ChunkThread: mkdir output dir: %w", err)
	}

	b, err := os.ReadFile(threadPath)
	if err != nil {
		return nil, fmt.Errorf("ChunkThread: read thread: %w", err)
	}

	var thread SimplifiedConversation
	if err := json.Unmarshal(b, &thread); err != nil {
		return nil, fmt.Errorf("ChunkThread: unmarshal thread: %w", err)
	}

	turns := BuildTurns(thread)
	if len(turns) == 0 {
		return nil, errors.New("ChunkThread: thread has no messages/turns")
	}

	breakpoints, err := decider.DecideBreakpoints(ctx, thread, turns, targetTurnsPerChunk)
	if err != nil {
		return nil, fmt.Errorf("ChunkThread: decide breakpoints: %w", err)
	}
	if len(breakpoints) == 0 {
		breakpoints = fallbackBreakpoints(len(turns), targetTurnsPerChunk)
	}

	chunks, err := ApplyTurnBreakpoints(thread, turns, breakpoints)
	if err != nil {
		return nil, err
	}

	threadStart := threadStartTime(thread)
	startStamp := formatUnixSeconds(threadStart)
	if startStamp == "" {
		startStamp = "thread"
	}

	var written []string
	for i, ch := range chunks {
		ch.ChunkNumber = i + 1
		ch.ThreadStart = threadStart

		filename := fmt.Sprintf("%s_%d.json", startStamp, ch.ChunkNumber)
		outPath := filepath.Join(opts.OutputDir, filename)
		if !opts.OverwriteExisting {
			if _, err := os.Stat(outPath); err == nil {
				return nil, fmt.Errorf("ChunkThread: output file already exists: %s", outPath)
			} else if !errors.Is(err, fs.ErrNotExist) {
				return nil, fmt.Errorf("ChunkThread: stat output file: %w", err)
			}
		}

		var out []byte
		if opts.Pretty {
			out, err = json.MarshalIndent(ch, "", "  ")
		} else {
			out, err = json.Marshal(ch)
		}
		if err != nil {
			return nil, fmt.Errorf("ChunkThread: marshal chunk: %w", err)
		}

		if _, err := writeFileAtomic(opts.OutputDir, outPath, out, opts.FileMode); err != nil {
			return nil, fmt.Errorf("ChunkThread: write chunk file: %w", err)
		}
		written = append(written, outPath)
	}

	return written, nil
}

func threadStartTime(thread SimplifiedConversation) *float64 {
	if thread.CreateTime != nil {
		return thread.CreateTime
	}
	if len(thread.Messages) > 0 && thread.Messages[0].CreateTime != nil {
		return thread.Messages[0].CreateTime
	}
	return nil
}

func formatUnixSeconds(t *float64) string {
	if t == nil {
		return ""
	}
	sec := int64(math.Floor(*t))
	if sec <= 0 {
		return ""
	}
	return fmt.Sprintf("%d", sec)
}

// ApplyTurnBreakpoints converts turn breakpoints into chunk objects.
func ApplyTurnBreakpoints(thread SimplifiedConversation, turns []Turn, breakpoints []int) ([]Chunk, error) {
	totalTurns := len(turns)
	if totalTurns == 0 {
		return nil, errors.New("ApplyTurnBreakpoints: no turns")
	}

	bps, err := normalizeBreakpoints(breakpoints, totalTurns)
	if err != nil {
		return nil, err
	}

	// Build boundaries: always include 0 and totalTurns.
	boundaries := make([]int, 0, len(bps)+2)
	boundaries = append(boundaries, 0)
	boundaries = append(boundaries, bps...)
	boundaries = append(boundaries, totalTurns)

	var chunks []Chunk
	for i := 0; i+1 < len(boundaries); i++ {
		ts := boundaries[i]
		te := boundaries[i+1]
		if ts >= te {
			continue
		}
		ms := turns[ts].StartMessageIndex
		me := turns[te-1].EndMessageIndex
		if ms < 0 || me < ms || me >= len(thread.Messages) {
			return nil, fmt.Errorf("ApplyTurnBreakpoints: invalid message range for turns [%d,%d): %d..%d", ts, te, ms, me)
		}

		chunks = append(chunks, Chunk{
			ConversationID: thread.ConversationID,
			Title:          thread.Title,
			TurnStart:      ts,
			TurnEnd:        te,
			Messages:       append([]SimplifiedMessage(nil), thread.Messages[ms:me+1]...),
		})
	}

	if len(chunks) == 0 {
		return nil, errors.New("ApplyTurnBreakpoints: produced no chunks")
	}
	return chunks, nil
}

func normalizeBreakpoints(breakpoints []int, totalTurns int) ([]int, error) {
	if totalTurns <= 1 {
		return nil, nil
	}

	if len(breakpoints) == 0 {
		return nil, nil
	}

	// Copy + sort + dedupe.
	bps := append([]int(nil), breakpoints...)
	sort.Ints(bps)

	out := bps[:0]
	prev := -1
	for _, b := range bps {
		if b <= 0 || b >= totalTurns {
			continue
		}
		if b == prev {
			continue
		}
		out = append(out, b)
		prev = b
	}

	// If everything got filtered, that's okay: single chunk.
	return out, nil
}

func fallbackBreakpoints(totalTurns int, targetTurnsPerChunk int) []int {
	if targetTurnsPerChunk <= 0 || totalTurns <= targetTurnsPerChunk {
		return nil
	}
	var bps []int
	for i := targetTurnsPerChunk; i < totalTurns; i += targetTurnsPerChunk {
		bps = append(bps, i)
	}
	return bps
}
