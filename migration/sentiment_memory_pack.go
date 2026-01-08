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

// SentimentMemoryShardIndexRecord maps one sentiment thread summary to a markdown shard file and anchor.
type SentimentMemoryShardIndexRecord struct {
	ConversationID string   `json:"conversation_id"`
	ThreadStart    *float64 `json:"thread_start_time,omitempty"`
	ThreadStartISO string   `json:"thread_start_time_iso8601,omitempty"`
	Title          string   `json:"title,omitempty"`

	ShardFile string `json:"shard_file"`
	Anchor    string `json:"anchor"`

	EmotionalSummary   string   `json:"emotional_summary"`
	DominantEmotions   []string `json:"dominant_emotions,omitempty"`
	RememberedEmotions []string `json:"remembered_emotions,omitempty"`
	PresentEmotions    []string `json:"present_emotions,omitempty"`
	EmotionalTensions  []string `json:"emotional_tensions,omitempty"`
	RelationalShift    string   `json:"relational_shift,omitempty"`
	EmotionalArc       string   `json:"emotional_arc,omitempty"`
	Themes             []string `json:"themes,omitempty"`
}

// WriteSentimentMemoryShards writes markdown shard files for sentiment thread summaries.
func WriteSentimentMemoryShards(threadSummaries []ThreadSentimentSummary, opts MemoryPackOptions) ([]SentimentMemoryShardIndexRecord, error) {
	if opts.OutDir == "" {
		return nil, errors.New("WriteSentimentMemoryShards: OutDir is empty")
	}
	if opts.MaxBytes <= 0 {
		opts.MaxBytes = 100 * 1024
	}
	if err := os.MkdirAll(opts.OutDir, 0o755); err != nil {
		return nil, fmt.Errorf("WriteSentimentMemoryShards: mkdir OutDir: %w", err)
	}

	summaries := append([]ThreadSentimentSummary(nil), threadSummaries...)
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
		index        []SentimentMemoryShardIndexRecord
	)

	flush := func() error {
		if currBytes == 0 {
			return nil
		}
		if currFilename == "" {
			currFilename = sentimentShardName(shardNum)
		}
		outPath := filepath.Join(opts.OutDir, currFilename)
		if !opts.Overwrite {
			if _, err := os.Stat(outPath); err == nil {
				return fmt.Errorf("WriteSentimentMemoryShards: shard exists: %s", outPath)
			}
		}
		if _, err := writeFileAtomic(opts.OutDir, outPath, []byte(curr.String()), 0o644); err != nil {
			return fmt.Errorf("WriteSentimentMemoryShards: write shard: %w", err)
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
		section, anchor := renderThreadSentimentMarkdown(ts)
		sectionBytes := len([]byte(section))

		if currBytes > 0 && currBytes+sectionBytes > opts.MaxBytes {
			if err := flush(); err != nil {
				return nil, err
			}
		}

		if currBytes == 0 {
			currFilename = sentimentShardName(shardNum)
			header := fmt.Sprintf("# Sentiment Memory Shard %04d\n\n", shardNum)
			curr.WriteString(header)
			currBytes += len([]byte(header))
		}

		curr.WriteString(section)
		currBytes += sectionBytes

		index = append(index, SentimentMemoryShardIndexRecord{
			ConversationID:     ts.ConversationID,
			ThreadStart:        ts.ThreadStart,
			ThreadStartISO:     threadStartISO8601(ts.ThreadStart),
			Title:              ts.Title,
			ShardFile:          currFilename,
			Anchor:             anchor,
			EmotionalSummary:   truncateForIndex(ts.EmotionalSummary, 400),
			DominantEmotions:   dedupeStrings(ts.DominantEmotions),
			RememberedEmotions: dedupeStrings(ts.RememberedEmotions),
			PresentEmotions:    dedupeStrings(ts.PresentEmotions),
			EmotionalTensions:  dedupeStrings(ts.EmotionalTensions),
			RelationalShift:    strings.TrimSpace(ts.RelationalShift),
			EmotionalArc:       strings.TrimSpace(ts.EmotionalArc),
			Themes:             dedupeStrings(ts.Themes),
		})
	}

	if err := flush(); err != nil {
		return nil, err
	}
	return index, nil
}

func sentimentShardName(n int) string {
	return fmt.Sprintf("sentiment_memories_%04d.md", n)
}

func renderThreadSentimentMarkdown(ts ThreadSentimentSummary) (section string, anchor string) {
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

	if s := strings.TrimSpace(ts.EmotionalSummary); s != "" {
		b.WriteString(s)
		b.WriteString("\n\n")
	}

	writeList := func(label string, items []string) {
		items = dedupeStrings(items)
		if len(items) == 0 {
			return
		}
		fmt.Fprintf(&b, "**%s**: %s\n\n", label, escapeMarkdownInline(strings.Join(items, ", ")))
	}

	writeList("dominant_emotions", ts.DominantEmotions)
	writeList("remembered_emotions", ts.RememberedEmotions)
	writeList("present_emotions", ts.PresentEmotions)
	writeList("emotional_tensions", ts.EmotionalTensions)
	writeList("themes", ts.Themes)
	if strings.TrimSpace(ts.RelationalShift) != "" {
		fmt.Fprintf(&b, "**relational_shift**: %s\n\n", escapeMarkdownInline(strings.TrimSpace(ts.RelationalShift)))
	}
	if strings.TrimSpace(ts.EmotionalArc) != "" {
		fmt.Fprintf(&b, "**emotional_arc**: %s\n\n", escapeMarkdownInline(strings.TrimSpace(ts.EmotionalArc)))
	}

	b.WriteString("\n---\n\n")
	return b.String(), anchor
}

// WriteSentimentMemoryIndex writes sentiment shard index records as JSONL.
func WriteSentimentMemoryIndex(path string, records []SentimentMemoryShardIndexRecord, overwrite bool) error {
	if path == "" {
		return errors.New("WriteSentimentMemoryIndex: path is empty")
	}
	if !overwrite {
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("WriteSentimentMemoryIndex: file exists: %s", path)
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
