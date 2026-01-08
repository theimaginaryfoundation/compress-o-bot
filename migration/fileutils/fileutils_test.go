package fileutils

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCopyFileIfExists(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	src := filepath.Join(dir, "src.txt")
	dst := filepath.Join(dir, "out", "dst.txt")

	// Missing src: no-op.
	copied, err := CopyFileIfExists(src, dst, false)
	if err != nil {
		t.Fatalf("copy missing src: %v", err)
	}
	if copied {
		t.Fatalf("expected copied=false for missing src")
	}

	if err := os.WriteFile(src, []byte("hello"), 0o644); err != nil {
		t.Fatalf("write src: %v", err)
	}

	// First copy should create dst.
	copied, err = CopyFileIfExists(src, dst, false)
	if err != nil {
		t.Fatalf("copy: %v", err)
	}
	if !copied {
		t.Fatalf("expected copied=true")
	}
	b, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("read dst: %v", err)
	}
	if string(b) != "hello" {
		t.Fatalf("dst=%q", string(b))
	}

	// Without overwrite, should not change dst.
	if err := os.WriteFile(src, []byte("new"), 0o644); err != nil {
		t.Fatalf("write src2: %v", err)
	}
	copied, err = CopyFileIfExists(src, dst, false)
	if err != nil {
		t.Fatalf("copy no-overwrite: %v", err)
	}
	if copied {
		t.Fatalf("expected copied=false when dst exists and overwrite=false")
	}
	b, _ = os.ReadFile(dst)
	if string(b) != "hello" {
		t.Fatalf("dst changed unexpectedly: %q", string(b))
	}

	// With overwrite, should update dst.
	copied, err = CopyFileIfExists(src, dst, true)
	if err != nil {
		t.Fatalf("copy overwrite: %v", err)
	}
	if !copied {
		t.Fatalf("expected copied=true when overwrite=true")
	}
	b, _ = os.ReadFile(dst)
	if string(b) != "new" {
		t.Fatalf("dst=%q", string(b))
	}
}
