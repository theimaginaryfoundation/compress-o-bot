package migration

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSplitConversationArchive_TopLevelArray(t *testing.T) {
	t.Parallel()

	in := `[{"title":"A","conversation_id":"c1","id":"c1","current_node":"m2","mapping":{"m1":{"id":"m1","message":{"author":{"role":"user","name":null},"create_time":1,"content":{"content_type":"text","parts":["hi"]},"metadata":{}},"parent":null,"children":["m2"]},"m2":{"id":"m2","message":{"author":{"role":"assistant","name":null},"create_time":2,"content":{"content_type":"text","parts":["hello"]},"metadata":{}},"parent":"m1","children":[]}}},{"title":"B","conversation_id":"c2","id":"c2","mapping":{}}]`
	inPath := filepath.Join(t.TempDir(), "in.json")
	if err := os.WriteFile(inPath, []byte(in), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	outDir := filepath.Join(t.TempDir(), "out")
	res, err := SplitConversationArchive(context.Background(), inPath, outDir, SplitOptions{})
	if err != nil {
		t.Fatalf("SplitConversationArchive: %v", err)
	}
	if res.ThreadsWritten != 2 {
		t.Fatalf("ThreadsWritten=%d, want 2", res.ThreadsWritten)
	}

	assertConversationIDInFile(t, filepath.Join(outDir, "c1.json"), "c1")
	assertConversationIDInFile(t, filepath.Join(outDir, "c2.json"), "c2")

	c1 := readSimplifiedConversation(t, filepath.Join(outDir, "c1.json"))
	if len(c1.Messages) != 2 {
		t.Fatalf("len(Messages)=%d, want 2", len(c1.Messages))
	}
	if c1.Messages[0].Role != "user" || c1.Messages[0].Text != "hi" {
		t.Fatalf("msg0=%+v, want role=user text=hi", c1.Messages[0])
	}
	if c1.Messages[1].Role != "assistant" || c1.Messages[1].Text != "hello" {
		t.Fatalf("msg1=%+v, want role=assistant text=hello", c1.Messages[1])
	}
}

func TestSplitConversationArchive_ObjectWrappedArray(t *testing.T) {
	t.Parallel()

	in := `{"conversations":[{"title":"A","conversation_id":"c1","id":"c1","mapping":{}},{"title":"B","conversation_id":"c2","id":"c2","mapping":{}}],"other":123}`
	inPath := filepath.Join(t.TempDir(), "in.json")
	if err := os.WriteFile(inPath, []byte(in), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	outDir := filepath.Join(t.TempDir(), "out")
	res, err := SplitConversationArchive(context.Background(), inPath, outDir, SplitOptions{ArrayField: "conversations"})
	if err != nil {
		t.Fatalf("SplitConversationArchive: %v", err)
	}
	if res.ThreadsWritten != 2 {
		t.Fatalf("ThreadsWritten=%d, want 2", res.ThreadsWritten)
	}
	assertConversationIDInFile(t, filepath.Join(outDir, "c1.json"), "c1")
	assertConversationIDInFile(t, filepath.Join(outDir, "c2.json"), "c2")
}

func TestSplitConversationArchive_DuplicateIDs(t *testing.T) {
	t.Parallel()

	in := `[{"conversation_id":"dup","id":"dup","mapping":{}},{"conversation_id":"dup","id":"dup","mapping":{}}]`
	inPath := filepath.Join(t.TempDir(), "in.json")
	if err := os.WriteFile(inPath, []byte(in), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	outDir := filepath.Join(t.TempDir(), "out")
	res, err := SplitConversationArchive(context.Background(), inPath, outDir, SplitOptions{})
	if err != nil {
		t.Fatalf("SplitConversationArchive: %v", err)
	}
	if res.ThreadsWritten != 2 {
		t.Fatalf("ThreadsWritten=%d, want 2", res.ThreadsWritten)
	}

	assertConversationIDInFile(t, filepath.Join(outDir, "dup.json"), "dup")
	assertConversationIDInFile(t, filepath.Join(outDir, "dup-2.json"), "dup")
}

func TestSplitConversationArchive_DropsHiddenEmptySystemMessage(t *testing.T) {
	t.Parallel()

	in := `[{"conversation_id":"c1","id":"c1","current_node":"a","mapping":{"root":{"id":"root","message":null,"parent":null,"children":["sys"]},"sys":{"id":"sys","message":{"author":{"role":"system","name":null},"create_time":1,"content":{"content_type":"text","parts":[""]},"metadata":{"is_visually_hidden_from_conversation":true}},"parent":"root","children":["u"]},"u":{"id":"u","message":{"author":{"role":"user","name":null},"create_time":2,"content":{"content_type":"text","parts":["q"]},"metadata":{}},"parent":"sys","children":["a"]},"a":{"id":"a","message":{"author":{"role":"assistant","name":null},"create_time":3,"content":{"content_type":"text","parts":["a"]},"metadata":{}},"parent":"u","children":[]}}}]`
	inPath := filepath.Join(t.TempDir(), "in.json")
	if err := os.WriteFile(inPath, []byte(in), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	outDir := filepath.Join(t.TempDir(), "out")
	_, err := SplitConversationArchive(context.Background(), inPath, outDir, SplitOptions{})
	if err != nil {
		t.Fatalf("SplitConversationArchive: %v", err)
	}

	c1 := readSimplifiedConversation(t, filepath.Join(outDir, "c1.json"))
	if len(c1.Messages) != 2 {
		t.Fatalf("len(Messages)=%d, want 2", len(c1.Messages))
	}
	if c1.Messages[0].Role != "user" || c1.Messages[0].Text != "q" {
		t.Fatalf("msg0=%+v, want user q", c1.Messages[0])
	}
	if c1.Messages[1].Role != "assistant" || c1.Messages[1].Text != "a" {
		t.Fatalf("msg1=%+v, want assistant a", c1.Messages[1])
	}
}

func TestSplitConversationArchive_ToolTetherQuoteKept(t *testing.T) {
	t.Parallel()

	in := `[{"conversation_id":"c1","id":"c1","current_node":"tool","mapping":{"u":{"id":"u","message":{"author":{"role":"user","name":null},"create_time":1,"content":{"content_type":"text","parts":["search this"]},"metadata":{}},"parent":null,"children":["tool"]},"tool":{"id":"tool","message":{"author":{"role":"tool","name":"browser"},"create_time":2,"content":{"content_type":"tether_quote","text":"hello world","url":"https://example.com","title":"Example","domain":"example.com"},"metadata":{"sonic_classification_result":{"x":1}}},"parent":"u","children":[]}}}]`
	inPath := filepath.Join(t.TempDir(), "in.json")
	if err := os.WriteFile(inPath, []byte(in), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	outDir := filepath.Join(t.TempDir(), "out")
	_, err := SplitConversationArchive(context.Background(), inPath, outDir, SplitOptions{})
	if err != nil {
		t.Fatalf("SplitConversationArchive: %v", err)
	}

	c1 := readSimplifiedConversation(t, filepath.Join(outDir, "c1.json"))
	if len(c1.Messages) != 2 {
		t.Fatalf("len(Messages)=%d, want 2", len(c1.Messages))
	}
	if c1.Messages[1].Role != "tool" || c1.Messages[1].Name != "browser" {
		t.Fatalf("tool msg=%+v, want role=tool name=browser", c1.Messages[1])
	}
	if c1.Messages[1].ContentType != "tether_quote" || c1.Messages[1].Text != "hello world" {
		t.Fatalf("tool msg=%+v, want tether_quote + text", c1.Messages[1])
	}
	if c1.Messages[1].URL != "https://example.com" || c1.Messages[1].Domain != "example.com" || c1.Messages[1].Title != "Example" {
		t.Fatalf("tool msg=%+v, want url/domain/title", c1.Messages[1])
	}
}

func TestSanitizeFilenameComponent(t *testing.T) {
	t.Parallel()

	got := sanitizeFilenameComponent("  ../weird id: 123  ")
	if got == "" {
		t.Fatalf("expected non-empty")
	}
	if got[0] == '.' {
		t.Fatalf("expected not to start with '.', got %q", got)
	}
}

func assertConversationIDInFile(t *testing.T, path, want string) {
	t.Helper()

	c := readSimplifiedConversation(t, path)
	if c.ConversationID != want {
		t.Fatalf("conversation_id=%q, want %q in %s", c.ConversationID, want, path)
	}
}

func readSimplifiedConversation(t *testing.T, path string) SimplifiedConversation {
	t.Helper()

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var c SimplifiedConversation
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatalf("unmarshal %s: %v", path, err)
	}
	return c
}
