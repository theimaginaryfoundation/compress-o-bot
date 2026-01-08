package fileutils

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

func FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func SanitizeNewlines(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.ReplaceAll(s, "\n", "\\n")
	return s
}

func Truncate(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func CopyFileIfExists(srcPath, dstPath string, overwrite bool) (bool, error) {
	if srcPath == "" || dstPath == "" {
		return false, errors.New("copyFileIfExists: empty path")
	}

	if _, err := os.Stat(srcPath); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, nil
		}
		return false, err
	}

	if !overwrite {
		if _, err := os.Stat(dstPath); err == nil {
			return false, nil
		} else if !errors.Is(err, fs.ErrNotExist) {
			return false, err
		}
	}

	b, err := os.ReadFile(srcPath)
	if err != nil {
		return false, err
	}

	if err := os.MkdirAll(filepath.Dir(dstPath), 0o755); err != nil {
		return false, err
	}

	tmp, err := os.CreateTemp(filepath.Dir(dstPath), ".tmp_copy_*")
	if err != nil {
		return false, err
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()

	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		return false, err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return false, err
	}
	if err := tmp.Close(); err != nil {
		return false, err
	}
	if err := os.Rename(tmpName, dstPath); err != nil {
		return false, err
	}
	return true, nil
}

func WriteJSONFileAtomic(path string, v any, pretty bool) error {
	var b []byte
	var err error
	if pretty {
		b, err = json.MarshalIndent(v, "", "  ")
	} else {
		b, err = json.Marshal(v)
	}
	if err != nil {
		return fmt.Errorf("marshal json: %w", err)
	}
	if err := WriteFileAtomicSameDir(path, b, 0o644); err != nil {
		return fmt.Errorf("write json: %w", err)
	}
	return nil
}

func WriteFileAtomicSameDir(path string, data []byte, mode fs.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dir, ".tmp_summary_*.json")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer func() {
		_ = os.Remove(tmpName)
	}()

	if err := tmp.Chmod(mode); err != nil {
		_ = tmp.Close()
		return err
	}

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write([]byte("\n")); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	return os.Rename(tmpName, path)
}
