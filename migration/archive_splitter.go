package migration

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// SimplifiedConversation is a summarization-friendly representation of a conversation/thread.
// It keeps just the fields that are typically useful for building condensed summaries and RAG indexes.
type SimplifiedConversation struct {
	ConversationID string              `json:"conversation_id"`
	Title          string              `json:"title,omitempty"`
	CreateTime     *float64            `json:"create_time,omitempty"`
	UpdateTime     *float64            `json:"update_time,omitempty"`
	Messages       []SimplifiedMessage `json:"messages"`
}

// SimplifiedMessage is a summarization-friendly representation of a single message.
type SimplifiedMessage struct {
	Role        string   `json:"role"`
	Name        string   `json:"name,omitempty"`
	CreateTime  *float64 `json:"create_time,omitempty"`
	ContentType string   `json:"content_type,omitempty"`
	Text        string   `json:"text,omitempty"`

	// Common tool/web fields (kept only when present).
	Domain string `json:"domain,omitempty"`
	Title  string `json:"title,omitempty"`
	URL    string `json:"url,omitempty"`
}

// SplitOptions controls how SplitConversationArchive writes per-thread files.
type SplitOptions struct {
	// ArrayField is the JSON field name that contains the conversation array,
	// when the top-level JSON value is an object.
	//
	// If empty, SplitConversationArchive will try to find the first array-valued
	// field and treat it as the conversations array.
	ArrayField string

	// OverwriteExisting controls whether existing output files should be overwritten.
	// If false and a file already exists, SplitConversationArchive returns an error.
	OverwriteExisting bool

	// Pretty controls whether each output JSON file is indented for readability.
	Pretty bool

	// DirMode is used when creating output directories (defaults to 0o755).
	DirMode fs.FileMode

	// FileMode is used when creating output files (defaults to 0o644).
	FileMode fs.FileMode
}

// SplitResult contains basic stats from a split run.
type SplitResult struct {
	ThreadsWritten int
	BytesWritten   int64
}

// SplitConversationArchive reads a large OpenAI conversations export and writes one JSON file per
// thread (conversation) into outputDir.
//
// The input is expected to be either:
// - a top-level JSON array: [ { ...conversation... }, ... ]
// - a top-level JSON object containing an array field (e.g. { "conversations": [ ... ] })
//
// It uses a streaming decoder and never reads the full file into memory at once.
func SplitConversationArchive(ctx context.Context, inputPath, outputDir string, opts SplitOptions) (SplitResult, error) {
	if ctx == nil {
		return SplitResult{}, errors.New("SplitConversationArchive: ctx is nil")
	}
	if inputPath == "" {
		return SplitResult{}, errors.New("SplitConversationArchive: inputPath is empty")
	}
	if outputDir == "" {
		return SplitResult{}, errors.New("SplitConversationArchive: outputDir is empty")
	}
	if opts.DirMode == 0 {
		opts.DirMode = 0o755
	}
	if opts.FileMode == 0 {
		opts.FileMode = 0o644
	}
	if err := os.MkdirAll(outputDir, opts.DirMode); err != nil {
		return SplitResult{}, fmt.Errorf("SplitConversationArchive: mkdir outputDir: %w", err)
	}

	f, err := os.Open(inputPath)
	if err != nil {
		return SplitResult{}, fmt.Errorf("SplitConversationArchive: open input: %w", err)
	}
	defer f.Close()

	// The export is typically one huge line; use a larger buffer than default.
	dec := json.NewDecoder(bufio.NewReaderSize(f, 1<<20))
	dec.UseNumber()

	tok, err := dec.Token()
	if err != nil {
		return SplitResult{}, fmt.Errorf("SplitConversationArchive: read first token: %w", err)
	}

	delim, ok := tok.(json.Delim)
	if !ok {
		return SplitResult{}, fmt.Errorf("SplitConversationArchive: expected JSON array/object, got %T", tok)
	}

	seen := make(map[string]int)
	var res SplitResult

	switch delim {
	case '[':
		if err := splitArrayFromOpen(ctx, dec, outputDir, opts, seen, &res); err != nil {
			return SplitResult{}, err
		}
		// Consume the closing ']'.
		if tok, err := dec.Token(); err != nil {
			return SplitResult{}, fmt.Errorf("SplitConversationArchive: read closing array token: %w", err)
		} else if d, ok := tok.(json.Delim); !ok || d != ']' {
			return SplitResult{}, fmt.Errorf("SplitConversationArchive: expected closing ']', got %v", tok)
		}
		return res, nil
	case '{':
		// Scan fields until we find the conversations array.
		foundArray := false
		for dec.More() {
			select {
			case <-ctx.Done():
				return SplitResult{}, ctx.Err()
			default:
			}

			keyTok, err := dec.Token()
			if err != nil {
				return SplitResult{}, fmt.Errorf("SplitConversationArchive: read object key: %w", err)
			}
			key, ok := keyTok.(string)
			if !ok {
				return SplitResult{}, fmt.Errorf("SplitConversationArchive: expected string key, got %T", keyTok)
			}

			valTok, err := dec.Token()
			if err != nil {
				return SplitResult{}, fmt.Errorf("SplitConversationArchive: read value token for key %q: %w", key, err)
			}

			isTarget := opts.ArrayField != "" && key == opts.ArrayField
			if !isTarget && opts.ArrayField == "" && !foundArray {
				if d, ok := valTok.(json.Delim); ok && d == '[' {
					isTarget = true
				}
			}

			if isTarget {
				d, ok := valTok.(json.Delim)
				if !ok || d != '[' {
					return SplitResult{}, fmt.Errorf("SplitConversationArchive: key %q was chosen as array but value isn't an array", key)
				}
				foundArray = true
				if err := splitArrayFromOpen(ctx, dec, outputDir, opts, seen, &res); err != nil {
					return SplitResult{}, err
				}
				// Consume the closing ']'.
				if tok, err := dec.Token(); err != nil {
					return SplitResult{}, fmt.Errorf("SplitConversationArchive: read closing array token: %w", err)
				} else if d, ok := tok.(json.Delim); !ok || d != ']' {
					return SplitResult{}, fmt.Errorf("SplitConversationArchive: expected closing ']', got %v", tok)
				}
				continue
			}

			if err := skipValue(dec, valTok); err != nil {
				return SplitResult{}, fmt.Errorf("SplitConversationArchive: skip key %q value: %w", key, err)
			}
		}

		// Consume the closing '}'.
		if tok, err := dec.Token(); err != nil {
			return SplitResult{}, fmt.Errorf("SplitConversationArchive: read closing object token: %w", err)
		} else if d, ok := tok.(json.Delim); !ok || d != '}' {
			return SplitResult{}, fmt.Errorf("SplitConversationArchive: expected closing '}', got %v", tok)
		}
		if !foundArray {
			return SplitResult{}, errors.New("SplitConversationArchive: no conversations array found in top-level object")
		}
		return res, nil
	default:
		return SplitResult{}, fmt.Errorf("SplitConversationArchive: unsupported top-level delimiter %q", delim)
	}
}

func splitArrayFromOpen(ctx context.Context, dec *json.Decoder, outputDir string, opts SplitOptions, seen map[string]int, res *SplitResult) error {
	for dec.More() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return fmt.Errorf("SplitConversationArchive: decode conversation element: %w", err)
		}

		simplified, id, err := simplifyConversation(raw)
		if err != nil {
			return err
		}

		base := sanitizeFilenameComponent(id)
		if base == "" {
			base = "thread"
		}

		seenCount := seen[base]
		seen[base] = seenCount + 1

		filename := base
		if seenCount > 0 {
			filename = fmt.Sprintf("%s-%d", base, seenCount+1)
		}
		filename += ".json"

		outPath := filepath.Join(outputDir, filename)
		if !opts.OverwriteExisting {
			if _, err := os.Stat(outPath); err == nil {
				return fmt.Errorf("SplitConversationArchive: output file already exists: %s", outPath)
			} else if !errors.Is(err, fs.ErrNotExist) {
				return fmt.Errorf("SplitConversationArchive: stat output file: %w", err)
			}
		}

		var toWrite []byte
		if opts.Pretty {
			b, err := json.MarshalIndent(simplified, "", "  ")
			if err != nil {
				return fmt.Errorf("SplitConversationArchive: marshal indent (id=%q): %w", id, err)
			}
			toWrite = b
		} else {
			b, err := json.Marshal(simplified)
			if err != nil {
				return fmt.Errorf("SplitConversationArchive: marshal (id=%q): %w", id, err)
			}
			toWrite = b
		}

		n, err := writeFileAtomic(outputDir, outPath, toWrite, opts.FileMode)
		if err != nil {
			return fmt.Errorf("SplitConversationArchive: write output (id=%q): %w", id, err)
		}
		res.ThreadsWritten++
		res.BytesWritten += n
	}
	return nil
}

type rawConversation struct {
	ConversationID string                `json:"conversation_id"`
	ID             string                `json:"id"`
	Title          string                `json:"title"`
	CreateTime     *float64              `json:"create_time"`
	UpdateTime     *float64              `json:"update_time"`
	CurrentNode    string                `json:"current_node"`
	Mapping        map[string]rawMapNode `json:"mapping"`
}

type rawMapNode struct {
	ID       string      `json:"id"`
	Message  *rawMessage `json:"message"`
	Parent   *string     `json:"parent"`
	Children []string    `json:"children"`
}

type rawMessage struct {
	Author     rawAuthor       `json:"author"`
	CreateTime *float64        `json:"create_time"`
	Content    json.RawMessage `json:"content"`
	Metadata   map[string]any  `json:"metadata"`
}

type rawAuthor struct {
	Role string  `json:"role"`
	Name *string `json:"name"`
}

func simplifyConversation(raw json.RawMessage) (SimplifiedConversation, string, error) {
	var conv rawConversation
	if err := json.Unmarshal(raw, &conv); err != nil {
		return SimplifiedConversation{}, "", fmt.Errorf("SplitConversationArchive: unmarshal conversation: %w", err)
	}

	id := conv.ConversationID
	if id == "" {
		id = conv.ID
	}
	if id == "" {
		return SimplifiedConversation{}, "", errors.New("SplitConversationArchive: conversation element missing conversation_id/id")
	}

	msgs, err := linearizeMessages(conv.Mapping, conv.CurrentNode)
	if err != nil {
		return SimplifiedConversation{}, "", fmt.Errorf("SplitConversationArchive: linearize messages (id=%q): %w", id, err)
	}

	return SimplifiedConversation{
		ConversationID: id,
		Title:          conv.Title,
		CreateTime:     conv.CreateTime,
		UpdateTime:     conv.UpdateTime,
		Messages:       msgs,
	}, id, nil
}

func linearizeMessages(mapping map[string]rawMapNode, currentNode string) ([]SimplifiedMessage, error) {
	if len(mapping) == 0 {
		return nil, nil
	}

	start := currentNode
	if start == "" {
		start = pickBestLeaf(mapping)
	}
	if start == "" {
		return nil, errors.New("no current_node and no leaf node found")
	}

	visited := make(map[string]struct{}, len(mapping))
	var reversed []SimplifiedMessage

	for i := 0; i < len(mapping)+5; i++ {
		n, ok := mapping[start]
		if !ok {
			return nil, fmt.Errorf("missing node %q in mapping", start)
		}
		if _, ok := visited[start]; ok {
			return nil, fmt.Errorf("cycle detected at node %q", start)
		}
		visited[start] = struct{}{}

		if n.Message != nil {
			sm, ok := simplifyMessage(*n.Message)
			if ok {
				reversed = append(reversed, sm)
			}
		}

		if n.Parent == nil || *n.Parent == "" {
			break
		}
		start = *n.Parent
	}

	// Reverse to chronological order.
	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}
	return reversed, nil
}

func pickBestLeaf(mapping map[string]rawMapNode) string {
	var (
		bestID   string
		bestTime float64
		hasBest  bool
	)
	for id, n := range mapping {
		if len(n.Children) != 0 || n.Message == nil {
			continue
		}
		ct := 0.0
		if n.Message.CreateTime != nil {
			ct = *n.Message.CreateTime
		}
		if !hasBest || ct > bestTime {
			bestID = id
			bestTime = ct
			hasBest = true
		}
	}
	return bestID
}

func simplifyMessage(m rawMessage) (SimplifiedMessage, bool) {
	role := strings.TrimSpace(m.Author.Role)
	if role == "" {
		role = "unknown"
	}
	name := ""
	if m.Author.Name != nil {
		name = strings.TrimSpace(*m.Author.Name)
	}

	ct, text, extra := extractContentSummary(m.Content)

	// Drop empty, hidden system nodes (very common in exports).
	if role == "system" && strings.TrimSpace(text) == "" && isHiddenFromConversation(m.Metadata) {
		return SimplifiedMessage{}, false
	}

	sm := SimplifiedMessage{
		Role:        role,
		Name:        name,
		CreateTime:  m.CreateTime,
		ContentType: ct,
		Text:        text,
		Domain:      extra.Domain,
		Title:       extra.Title,
		URL:         extra.URL,
	}

	// Drop "imagey" tool messages that carry no useful text/URL metadata.
	// In OpenAI exports these often show up as role=tool with content_type like "image" (or similar),
	// but parts are non-string and the result is just noise for text summarization.
	if sm.Role == "tool" &&
		strings.TrimSpace(sm.Text) == "" &&
		strings.TrimSpace(sm.Title) == "" &&
		strings.TrimSpace(sm.URL) == "" &&
		isImageLikeContentType(sm.ContentType) {
		return SimplifiedMessage{}, false
	}

	// If there's no usable content at all, skip.
	if strings.TrimSpace(sm.Text) == "" && sm.ContentType == "" && sm.URL == "" && sm.Title == "" {
		return SimplifiedMessage{}, false
	}
	return sm, true
}

type contentExtra struct {
	Domain string
	Title  string
	URL    string
}

func extractContentSummary(raw json.RawMessage) (contentType string, text string, extra contentExtra) {
	if len(raw) == 0 {
		return "", "", contentExtra{}
	}

	// Common export shape:
	// { "content_type": "text", "parts": ["..."] }
	// Tool/browser shape:
	// { "content_type": "tether_quote", "text": "...", "url": "...", ... }
	var probe struct {
		ContentType string `json:"content_type"`
		Parts       []any  `json:"parts"`
		Text        string `json:"text"`
		Domain      string `json:"domain"`
		Title       string `json:"title"`
		URL         string `json:"url"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return "", "", contentExtra{}
	}

	var parts []string
	for _, p := range probe.Parts {
		if s, ok := p.(string); ok {
			parts = append(parts, s)
		}
	}

	switch {
	case len(parts) > 0:
		text = strings.Join(parts, "\n")
	case probe.Text != "":
		text = probe.Text
	}

	return strings.TrimSpace(probe.ContentType), text, contentExtra{
		Domain: strings.TrimSpace(probe.Domain),
		Title:  strings.TrimSpace(probe.Title),
		URL:    strings.TrimSpace(probe.URL),
	}
}

func isHiddenFromConversation(metadata map[string]any) bool {
	if len(metadata) == 0 {
		return false
	}
	v, ok := metadata["is_visually_hidden_from_conversation"]
	if !ok {
		return false
	}
	b, ok := v.(bool)
	return ok && b
}

func isImageLikeContentType(ct string) bool {
	ct = strings.ToLower(strings.TrimSpace(ct))
	if ct == "" {
		return false
	}
	// Keep common useful tool types like tether_quote (handled by the caller condition anyway),
	// but specifically treat "image" typed tool outputs as low-signal when they have no text/url/title.
	return strings.Contains(ct, "image")
}

func sanitizeFilenameComponent(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}

	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(r)
		case r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}

	out := b.String()
	out = strings.Trim(out, "._-")
	out = strings.TrimPrefix(out, "..")
	out = strings.TrimPrefix(out, ".")
	out = strings.TrimSpace(out)
	return out
}

func writeFileAtomic(tmpDir, finalPath string, data []byte, mode fs.FileMode) (int64, error) {
	if err := os.MkdirAll(filepath.Dir(finalPath), 0o755); err != nil {
		return 0, err
	}

	tmp, err := os.CreateTemp(tmpDir, "archive_split_*.json")
	if err != nil {
		return 0, err
	}
	tmpName := tmp.Name()
	defer func() {
		_ = os.Remove(tmpName)
	}()

	if err := tmp.Chmod(mode); err != nil {
		_ = tmp.Close()
		return 0, err
	}

	n, err := tmp.Write(data)
	if err != nil {
		_ = tmp.Close()
		return int64(n), err
	}
	if _, err := tmp.Write([]byte("\n")); err != nil {
		_ = tmp.Close()
		return int64(n), err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return int64(n), err
	}
	if err := tmp.Close(); err != nil {
		return int64(n), err
	}

	if err := os.Rename(tmpName, finalPath); err != nil {
		return int64(n), err
	}
	return int64(n), nil
}

func skipValue(dec *json.Decoder, first json.Token) error {
	d, ok := first.(json.Delim)
	if !ok {
		// Primitive (string/number/bool/null): already fully consumed.
		return nil
	}

	switch d {
	case '{', '[':
		// Consume tokens until the matching closing delimiter.
	default:
		// '}' or ']' shouldn't appear as a value token.
		return fmt.Errorf("skipValue: unexpected delimiter %q", d)
	}

	depth := 1
	for depth > 0 {
		tok, err := dec.Token()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return io.ErrUnexpectedEOF
			}
			return err
		}
		if dd, ok := tok.(json.Delim); ok {
			switch dd {
			case '{', '[':
				depth++
			case '}', ']':
				depth--
			}
		}
	}
	return nil
}
