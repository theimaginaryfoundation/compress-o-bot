package fileutils

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// decodeModelJSON unmarshals JSON from a model response, with a small amount of robustness
// for cases where the model wraps the JSON in extra text or returns leading/trailing whitespace.
func DecodeModelJSON(outputText string, v any) error {
	s := strings.TrimSpace(outputText)
	if s == "" {
		return io.ErrUnexpectedEOF
	}

	// Fast path: valid JSON as-is.
	if err := json.Unmarshal([]byte(s), v); err == nil {
		return nil
	}

	// Fallback: attempt to extract the first top-level JSON object.
	start := strings.IndexByte(s, '{')
	end := strings.LastIndexByte(s, '}')
	if start == -1 || end == -1 || end <= start {
		return fmt.Errorf("no JSON object found in model output (len=%d)", len(s))
	}

	sub := s[start : end+1]
	if err := json.Unmarshal([]byte(sub), v); err != nil {
		return fmt.Errorf("failed to unmarshal extracted JSON (len=%d): %w", len(sub), err)
	}
	return nil
}
