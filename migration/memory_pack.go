package migration

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// MemoryPackOptions controls how markdown shards are created.
type MemoryPackOptions struct {
	OutDir    string
	MaxBytes  int // default ~100KB
	Overwrite bool

	// IncludeKeyPoints adds the KeyPoints list under each thread.
	IncludeKeyPoints bool

	// IncludeTags adds Tags/Terms lines under each thread (useful for human inspection).
	IncludeTags bool
}

// MemoryShardIndexRecord maps one thread to a markdown shard file and anchor.
type MemoryShardIndexRecord struct {
	ConversationID string   `json:"conversation_id"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`
	ThreadStartISO string   `json:"thread_start_time_iso8601,omitempty"`
	Title          string   `json:"title,omitempty"`

	ShardFile string `json:"shard_file"`
	Anchor    string `json:"anchor"`

	// Summary is duplicated (shortened) here for quick scanning.
	Summary string   `json:"summary"`
	Tags    []string `json:"tags,omitempty"`
	Terms   []string `json:"terms,omitempty"`
}

// WriteMemoryShards writes markdown shard files and an index.jsonl that maps threads -> shard files.
// Thread summaries are packed sequentially into shard files of roughly MaxBytes (UTF-8 bytes).
func WriteMemoryShards(threadSummaries []ThreadSummary, opts MemoryPackOptions) ([]MemoryShardIndexRecord, error) {
	if opts.OutDir == "" {
		return nil, errors.New("WriteMemoryShards: OutDir is empty")
	}
	if opts.MaxBytes <= 0 {
		opts.MaxBytes = 100 * 1024
	}
	if err := os.MkdirAll(opts.OutDir, 0o755); err != nil {
		return nil, fmt.Errorf("WriteMemoryShards: mkdir OutDir: %w", err)
	}

	// Stable ordering: start time (if present), then conversation_id.
	summaries := append([]ThreadSummary(nil), threadSummaries...)
	sort.SliceStable(summaries, func(i, j int) bool {
		ti := float64(0)
		tj := float64(0)
		if summaries[i].ThreadStart != nil {
			ti = *summaries[i].ThreadStart
		}
		if summaries[j].ThreadStart != nil {
			tj = *summaries[j].ThreadStart
		}
		if ti != tj {
			return ti < tj
		}
		return summaries[i].ConversationID < summaries[j].ConversationID
	})

	var (
		shardNum     = 1
		curr         strings.Builder
		currBytes    = 0
		currFilename = ""
		index        []MemoryShardIndexRecord
	)

	flush := func() error {
		if currBytes == 0 {
			return nil
		}
		if currFilename == "" {
			currFilename = shardName(shardNum)
		}
		outPath := filepath.Join(opts.OutDir, currFilename)
		if !opts.Overwrite {
			if _, err := os.Stat(outPath); err == nil {
				return fmt.Errorf("WriteMemoryShards: shard exists: %s", outPath)
			}
		}
		if _, err := writeFileAtomic(opts.OutDir, outPath, []byte(curr.String()), 0o644); err != nil {
			return fmt.Errorf("WriteMemoryShards: write shard: %w", err)
		}
		shardNum++
		curr.Reset()
		currBytes = 0
		currFilename = ""
		return nil
	}

	for _, ts := range summaries {
		if ts.ConversationID == "" {
			continue
		}
		section, anchor := renderThreadMarkdown(ts, opts.IncludeKeyPoints, opts.IncludeTags)
		sectionBytes := len([]byte(section))

		if currBytes > 0 && currBytes+sectionBytes > opts.MaxBytes {
			if err := flush(); err != nil {
				return nil, err
			}
		}

		if currBytes == 0 {
			currFilename = shardName(shardNum)
			header := fmt.Sprintf("# Memory Shard %04d\n\n", shardNum)
			curr.WriteString(header)
			currBytes += len([]byte(header))
		}

		curr.WriteString(section)
		currBytes += sectionBytes

		index = append(index, MemoryShardIndexRecord{
			ConversationID: ts.ConversationID,
			ThreadStart:    ts.ThreadStart,
			ThreadStartISO: threadStartISO8601(ts.ThreadStart),
			Title:          ts.Title,
			ShardFile:      currFilename,
			Anchor:         anchor,
			Summary:        truncateForIndex(ts.Summary, 400),
			Tags:           dedupeStrings(ts.Tags),
			Terms:          dedupeStrings(ts.Terms),
		})
	}

	if err := flush(); err != nil {
		return nil, err
	}
	return index, nil
}

func shardName(n int) string {
	return fmt.Sprintf("memories_%04d.md", n)
}

func renderThreadMarkdown(ts ThreadSummary, includeKeyPoints bool, includeTags bool) (section string, anchor string) {
	anchor = "thread-" + sanitizeAnchor(ts.ConversationID)
	title := strings.TrimSpace(ts.Title)
	if title == "" {
		title = ts.ConversationID
	}

	var b strings.Builder
	fmt.Fprintf(&b, "<a id=\"%s\"></a>\n", anchor)
	fmt.Fprintf(&b, "## %s\n\n", escapeMarkdownInline(title))
	fmt.Fprintf(&b, "- conversation_id: `%s`\n", ts.ConversationID)
	if ts.ThreadStart != nil {
		iso := threadStartISO8601(ts.ThreadStart)
		if iso != "" {
			fmt.Fprintf(&b, "- thread_start_time: `%.3f` (`%s`)\n", *ts.ThreadStart, iso)
		} else {
			fmt.Fprintf(&b, "- thread_start_time: `%.3f`\n", *ts.ThreadStart)
		}
	}
	b.WriteString("\n")

	sum := strings.TrimSpace(ts.Summary)
	if sum != "" {
		b.WriteString(sum)
		b.WriteString("\n\n")
	}

	if includeKeyPoints && len(ts.KeyPoints) > 0 {
		b.WriteString("### Key points\n")
		for _, kp := range ts.KeyPoints {
			kp = strings.TrimSpace(kp)
			if kp == "" {
				continue
			}
			fmt.Fprintf(&b, "- %s\n", sanitizeNewlines(kp))
		}
		b.WriteString("\n")
	}

	if includeTags {
		if len(ts.Tags) > 0 {
			fmt.Fprintf(&b, "**tags**: %s\n\n", escapeMarkdownInline(strings.Join(dedupeStrings(ts.Tags), ", ")))
		}
		if len(ts.Terms) > 0 {
			fmt.Fprintf(&b, "**terms**: %s\n\n", escapeMarkdownInline(strings.Join(dedupeStrings(ts.Terms), ", ")))
		}
	}

	b.WriteString("\n---\n\n")
	return b.String(), anchor
}

func sanitizeAnchor(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return "thread"
	}
	var out strings.Builder
	out.Grow(len(s))
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			out.WriteRune(r)
		} else {
			out.WriteByte('-')
		}
	}
	return strings.Trim(out.String(), "-")
}

func escapeMarkdownInline(s string) string {
	// Minimal: avoid accidental code fences/headers in titles/tags.
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.TrimSpace(s)
	return s
}

func truncateForIndex(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

// WriteMemoryIndex writes index records as JSONL.
func WriteMemoryIndex(path string, records []MemoryShardIndexRecord, overwrite bool) error {
	if path == "" {
		return errors.New("WriteMemoryIndex: path is empty")
	}
	if !overwrite {
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("WriteMemoryIndex: file exists: %s", path)
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	var b strings.Builder
	for _, r := range records {
		line, err := json.Marshal(r)
		if err != nil {
			return err
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	_, err := writeFileAtomic(filepath.Dir(path), path, []byte(b.String()), 0o644)
	return err
}

func sanitizeNewlines(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.ReplaceAll(s, "\n", "\\n")
	return s
}
