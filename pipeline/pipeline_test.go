package pipeline

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
)

// ---- fakes ----

type fakeSummarizer struct {
	calls int32
}

func (f *fakeSummarizer) SummarizeChunk(_ context.Context, c migration.Chunk, _ string, _ PromptOptions) (migration.ChunkSummary, []migration.GlossaryAddition, error) {
	atomic.AddInt32(&f.calls, 1)
	return migration.ChunkSummary{
		ConversationID: c.ConversationID,
		ThreadStart:    c.ThreadStart,
		ChunkNumber:    c.ChunkNumber,
		TurnStart:      c.TurnStart,
		TurnEnd:        c.TurnEnd,
		Summary:        "s",
		Terms:          []string{"alpha"},
	}, []migration.GlossaryAddition{{Term: "alpha", Definition: "first"}}, nil
}

func (f *fakeSummarizer) SummarizeChunkSentiment(_ context.Context, c migration.Chunk, _ string, _ PromptOptions) (migration.ChunkSentimentSummary, error) {
	return migration.ChunkSentimentSummary{
		ConversationID:   c.ConversationID,
		ThreadStart:      c.ThreadStart,
		ChunkNumber:      c.ChunkNumber,
		TurnStart:        c.TurnStart,
		TurnEnd:          c.TurnEnd,
		EmotionalSummary: "calm",
	}, nil
}

type fakeRolluper struct{}

func (fakeRolluper) Rollup(_ context.Context, id string, chunks []migration.ChunkSummary, _ string) (migration.ThreadSummary, error) {
	return migration.ThreadSummary{ConversationID: id, Title: "t", Summary: "rolled"}, nil
}

func (fakeRolluper) RollupFromThreadSummaries(_ context.Context, id string, parts []migration.ThreadSummary, _ string) (migration.ThreadSummary, error) {
	return migration.ThreadSummary{ConversationID: id, Title: "t", Summary: "merged"}, nil
}

type fakeSentRolluper struct{}

func (fakeSentRolluper) Rollup(_ context.Context, id string, _ []migration.ChunkSentimentSummary, _ string) (migration.ThreadSentimentSummary, error) {
	return migration.ThreadSentimentSummary{ConversationID: id, Title: "t", EmotionalSummary: "calm"}, nil
}

func (fakeSentRolluper) RollupFromThreadSentimentSummaries(_ context.Context, id string, _ []migration.ThreadSentimentSummary, _ string) (migration.ThreadSentimentSummary, error) {
	return migration.ThreadSentimentSummary{ConversationID: id, Title: "t", EmotionalSummary: "merged"}, nil
}

type fixedBreakpointDecider struct{ bps []int }

func (f fixedBreakpointDecider) DecideBreakpoints(_ context.Context, _ migration.SimplifiedConversation, _ []migration.Turn, _ int) ([]int, error) {
	return f.bps, nil
}

// ---- unit tests for helpers that were orphaned from cmd/thread-rollup ----

func TestIsRecoverableModelJSONError(t *testing.T) {
	t.Parallel()

	if isRecoverableModelJSONError(nil) {
		t.Fatal("nil should not be recoverable")
	}
	if !isRecoverableModelJSONError(errors.New("no JSON object found in model output (len=123)")) {
		t.Fatal("expected no-JSON-object error to be recoverable")
	}
	if !isRecoverableModelJSONError(errors.New("unexpected end of JSON input")) {
		t.Fatal("expected truncation error to be recoverable")
	}
	if !isRecoverableModelJSONError(io.ErrUnexpectedEOF) {
		t.Fatal("expected io.ErrUnexpectedEOF to be recoverable")
	}
	if isRecoverableModelJSONError(errors.New("some other parse error")) {
		t.Fatal("unexpected recoverable")
	}
}

func TestSplitChunksIntoWindows(t *testing.T) {
	t.Parallel()

	in := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	got := SplitChunksIntoWindows(in, 10)
	if len(got) != 2 || len(got[0]) != 10 || len(got[1]) != 6 {
		t.Fatalf("windows=%v", got)
	}

	// max <= 0 returns single window
	if one := SplitChunksIntoWindows(in, 0); len(one) != 1 || len(one[0]) != len(in) {
		t.Fatalf("expected single window, got %d", len(one))
	}
}

func TestMinThreadStartFromChunkSummaries(t *testing.T) {
	t.Parallel()

	a, b := 100.0, 50.0
	got := minThreadStartFromChunkSummaries([]migration.ChunkSummary{
		{ThreadStart: &a}, {ThreadStart: &b}, {ThreadStart: nil},
	})
	if got == nil || *got != 50.0 {
		t.Fatalf("got=%v", got)
	}

	if minThreadStartFromChunkSummaries(nil) != nil {
		t.Fatal("nil input should return nil")
	}
}

func TestMinThreadStartFromThreadSummaries(t *testing.T) {
	t.Parallel()

	a, b := 10.0, 20.0
	got := minThreadStartFromThreadSummaries([]migration.ThreadSummary{
		{ThreadStart: &b}, {ThreadStart: &a},
	})
	if got == nil || *got != 10.0 {
		t.Fatalf("got=%v", got)
	}
}

// ---- helpers ----

func TestGlossaryForPrompt(t *testing.T) {
	t.Parallel()

	g := migration.Glossary{
		Version: 1,
		Entries: []migration.GlossaryEntry{
			{Term: "alpha", Definition: "first"},
			{Term: "beta", Definition: "second"},
			{Term: "empty", Definition: ""},
		},
	}
	out := GlossaryForPrompt(g, 0)
	if out != "" {
		t.Fatalf("maxTerms=0 should return empty, got %q", out)
	}
	out = GlossaryForPrompt(g, -1)
	if !strings.Contains(out, "alpha: first") || !strings.Contains(out, "beta: second") {
		t.Fatalf("expected both terms, got %q", out)
	}
	if strings.Contains(out, "empty") {
		t.Fatalf("empty-def entry should be skipped, got %q", out)
	}
	out = GlossaryForPrompt(g, 1)
	if !strings.Contains(out, "alpha") || strings.Contains(out, "beta") {
		t.Fatalf("maxTerms=1 should cap, got %q", out)
	}
}

func TestComposeSentimentInstructions(t *testing.T) {
	t.Parallel()

	got := ComposeSentimentInstructions("custom header")
	if !strings.HasPrefix(got, "custom header") {
		t.Fatalf("missing header prefix: %q", got)
	}
	if !strings.Contains(got, "\n\nSECURITY:\n") {
		t.Fatalf("missing SECURITY tail")
	}

	// Empty header falls back to default.
	got = ComposeSentimentInstructions("")
	if strings.HasPrefix(got, "custom") {
		t.Fatal("empty header should fall back to default, not leak prior header")
	}
	if !strings.Contains(got, "SECURITY:") {
		t.Fatalf("missing tail in default: %q", got)
	}
}

// ---- end-to-end in-memory Run test ----

func TestRun_InMemory_NoDiskIO(t *testing.T) {
	t.Parallel()

	// Two conversations wrapped in a JSON object with an array field.
	input := `{
	  "conversations": [
	    {
	      "conversation_id": "c1",
	      "title": "First",
	      "create_time": 1000,
	      "mapping": {
	        "m1": {"id": "m1", "message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["hello"]}, "create_time": 1000}, "parent": null, "children": ["m2"]},
	        "m2": {"id": "m2", "message": {"author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["hi"]}, "create_time": 1001}, "parent": "m1", "children": ["m3"]},
	        "m3": {"id": "m3", "message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["bye"]}, "create_time": 1002}, "parent": "m2", "children": ["m4"]},
	        "m4": {"id": "m4", "message": {"author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["ok"]}, "create_time": 1003}, "parent": "m3", "children": []}
	      },
	      "current_node": "m4"
	    },
	    {
	      "conversation_id": "c2",
	      "title": "Second",
	      "create_time": 2000,
	      "mapping": {
	        "n1": {"id": "n1", "message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["q"]}, "create_time": 2000}, "parent": null, "children": ["n2"]},
	        "n2": {"id": "n2", "message": {"author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["a"]}, "create_time": 2001}, "parent": "n1", "children": []}
	      },
	      "current_node": "n2"
	    }
	  ]
	}`

	// chdir to an empty tempdir so any stray relative-path writes would land here.
	sandbox := t.TempDir()
	prevWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(sandbox); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(prevWD) })

	// Use unexported runner variant by wiring fakes directly.
	r, err := runWithDeps(
		context.Background(),
		strings.NewReader(input),
		Options{
			SemanticModel:       "fake",
			TargetTurnsPerChunk: 10,
			Concurrency:         2,
		},
		&fakeSummarizer{},
		fakeRolluper{},
		fakeSentRolluper{},
		fixedBreakpointDecider{bps: nil},
	)
	if err != nil {
		t.Fatalf("runWithDeps: %v", err)
	}

	if len(r.Threads) != 2 {
		t.Fatalf("threads=%d want 2", len(r.Threads))
	}
	if len(r.ChunkSummaries["c1"]) == 0 || len(r.ChunkSummaries["c2"]) == 0 {
		t.Fatalf("missing chunk summaries: %v", r.ChunkSummaries)
	}
	if r.ThreadSummaries["c1"].Summary != "rolled" || r.ThreadSummaries["c2"].Summary != "rolled" {
		t.Fatalf("thread summaries=%v", r.ThreadSummaries)
	}
	if r.ThreadSentiments["c1"].EmotionalSummary == "" {
		t.Fatalf("missing sentiment rollup")
	}
	if len(r.Glossary.Entries) == 0 {
		t.Fatalf("expected glossary to accumulate")
	}

	entries, err := os.ReadDir(sandbox)
	if err != nil {
		t.Fatalf("readdir sandbox: %v", err)
	}
	if len(entries) != 0 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("pipeline wrote files to CWD: %v", names)
	}
}

func TestRun_PreservesInputGlossary(t *testing.T) {
	t.Parallel()

	input := `[{"conversation_id":"c1","mapping":{"m1":{"id":"m1","message":{"author":{"role":"user"},"content":{"content_type":"text","parts":["hi"]}},"parent":null,"children":["m2"]},"m2":{"id":"m2","message":{"author":{"role":"assistant"},"content":{"content_type":"text","parts":["hey"]}},"parent":"m1","children":[]}},"current_node":"m2"}]`

	pre := migration.Glossary{
		Entries: []migration.GlossaryEntry{{Term: "preexisting", Definition: "d", Count: 5}},
	}

	r, err := runWithDeps(
		context.Background(),
		strings.NewReader(input),
		Options{
			SemanticModel:       "fake",
			TargetTurnsPerChunk: 10,
			InputGlossary:       pre, // Version=0 but has entries — must not be clobbered
		},
		&fakeSummarizer{},
		fakeRolluper{},
		fakeSentRolluper{},
		fixedBreakpointDecider{bps: nil},
	)
	if err != nil {
		t.Fatalf("runWithDeps: %v", err)
	}

	var foundPre bool
	for _, e := range r.Glossary.Entries {
		if e.Term == "preexisting" {
			foundPre = true
			break
		}
	}
	if !foundPre {
		t.Fatalf("input glossary entry was clobbered: %+v", r.Glossary)
	}
	if r.Glossary.Version != 1 {
		t.Fatalf("expected Version normalized to 1, got %d", r.Glossary.Version)
	}
}

func TestRun_ConcurrencyCap(t *testing.T) {
	t.Parallel()

	// Build one conversation with many turns so we get many chunks.
	var mappingB strings.Builder
	mappingB.WriteString(`"mapping":{`)
	prev := "null"
	parent := "null"
	_ = prev
	ids := []string{}
	for i := 0; i < 20; i++ {
		uid := "u" + itoa(i)
		aid := "a" + itoa(i)
		ids = append(ids, uid, aid)
		if i > 0 {
			mappingB.WriteString(",")
		}
		mappingB.WriteString(`"` + uid + `":{"id":"` + uid + `","message":{"author":{"role":"user"},"content":{"content_type":"text","parts":["u"]}},"parent":` + parent + `,"children":["` + aid + `"]},`)
		mappingB.WriteString(`"` + aid + `":{"id":"` + aid + `","message":{"author":{"role":"assistant"},"content":{"content_type":"text","parts":["a"]}},"parent":"` + uid + `","children":[`)
		if i < 19 {
			mappingB.WriteString(`"u` + itoa(i+1) + `"`)
		}
		mappingB.WriteString(`]}`)
		parent = `"` + aid + `"`
	}
	mappingB.WriteString("}")
	input := `[{"conversation_id":"c1",` + mappingB.String() + `,"current_node":"` + ids[len(ids)-1] + `"}]`

	// Force many small chunks.
	bps := []int{2, 4, 6, 8, 10, 12, 14, 16, 18}

	var counter concurrencyCounter
	delay := make(chan struct{})
	sum := &countingSummarizer{c: &counter, delay: delay}

	done := make(chan error, 1)
	go func() {
		_, err := runWithDeps(
			context.Background(),
			strings.NewReader(input),
			Options{SemanticModel: "fake", TargetTurnsPerChunk: 2, Concurrency: 3},
			sum,
			fakeRolluper{},
			fakeSentRolluper{},
			fixedBreakpointDecider{bps: bps},
		)
		done <- err
	}()

	// Wait until the cap saturates (3 in flight). Then release.
	deadline := time.After(2 * time.Second)
	for counter.max() < 3 {
		select {
		case <-deadline:
			close(delay)
			<-done
			t.Fatalf("cap never saturated: max=%d", counter.max())
		default:
			time.Sleep(2 * time.Millisecond)
		}
	}
	close(delay)
	if err := <-done; err != nil {
		t.Fatalf("runWithDeps: %v", err)
	}

	if counter.max() > 3 {
		t.Fatalf("concurrency exceeded cap: max=%d", counter.max())
	}
}

// ---- concurrency counter ----

type concurrencyCounter struct {
	mu      sync.Mutex
	cur, hi int
}

func (c *concurrencyCounter) enter() {
	c.mu.Lock()
	c.cur++
	if c.cur > c.hi {
		c.hi = c.cur
	}
	c.mu.Unlock()
}

func (c *concurrencyCounter) exit() {
	c.mu.Lock()
	c.cur--
	c.mu.Unlock()
}

func (c *concurrencyCounter) max() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hi
}

type countingSummarizer struct {
	c     *concurrencyCounter
	delay chan struct{} // released when cap should be observed
}

func (s *countingSummarizer) SummarizeChunk(_ context.Context, ch migration.Chunk, _ string, _ PromptOptions) (migration.ChunkSummary, []migration.GlossaryAddition, error) {
	s.c.enter()
	defer s.c.exit()
	if s.delay != nil {
		<-s.delay
	}
	return migration.ChunkSummary{ConversationID: ch.ConversationID, ChunkNumber: ch.ChunkNumber}, nil, nil
}

func (s *countingSummarizer) SummarizeChunkSentiment(_ context.Context, ch migration.Chunk, _ string, _ PromptOptions) (migration.ChunkSentimentSummary, error) {
	return migration.ChunkSentimentSummary{ConversationID: ch.ConversationID, ChunkNumber: ch.ChunkNumber}, nil
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	neg := i < 0
	if neg {
		i = -i
	}
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		b[pos] = '-'
	}
	return string(b[pos:])
}
