package migration

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// GlossaryAddition is a model-proposed term to add/update in the glossary.
type GlossaryAddition struct {
	Term       string `json:"term"`
	Definition string `json:"definition,omitempty"`
}

// LoadGlossary reads a glossary JSON file. If the file doesn't exist, it returns an empty glossary.
func LoadGlossary(path string) (Glossary, error) {
	if path == "" {
		return Glossary{}, errors.New("LoadGlossary: path is empty")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return Glossary{Version: 1, Entries: []GlossaryEntry{}}, nil
		}
		return Glossary{}, fmt.Errorf("LoadGlossary: read file: %w", err)
	}
	var g Glossary
	if err := json.Unmarshal(b, &g); err != nil {
		return Glossary{}, fmt.Errorf("LoadGlossary: unmarshal: %w", err)
	}
	if g.Version == 0 {
		g.Version = 1
	}
	if g.Entries == nil {
		g.Entries = []GlossaryEntry{}
	}
	return g, nil
}

// SaveGlossary writes the glossary JSON file atomically.
func SaveGlossary(path string, g Glossary) error {
	if path == "" {
		return errors.New("SaveGlossary: path is empty")
	}
	b, err := json.MarshalIndent(g, "", "  ")
	if err != nil {
		return fmt.Errorf("SaveGlossary: marshal: %w", err)
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("SaveGlossary: mkdir dir: %w", err)
	}
	_, err = writeFileAtomic(dir, path, b, 0o644)
	if err != nil {
		return fmt.Errorf("SaveGlossary: write: %w", err)
	}
	return nil
}

// MergeGlossary applies additions, bumps occurrence counts, and returns the list of terms that were touched.
func MergeGlossary(g *Glossary, additions []GlossaryAddition, seenAt *float64) []string {
	if g == nil {
		return nil
	}
	if g.Version == 0 {
		g.Version = 1
	}
	if g.Entries == nil {
		g.Entries = []GlossaryEntry{}
	}

	index := make(map[string]int, len(g.Entries))
	for i := range g.Entries {
		key := normalizeGlossaryKey(g.Entries[i].Term)
		if key != "" {
			index[key] = i
		}
	}

	seenKeys := make(map[string]struct{}, len(additions))
	for _, a := range additions {
		key := normalizeGlossaryKey(a.Term)
		if key == "" {
			continue
		}
		if _, ok := seenKeys[key]; ok {
			continue
		}
		seenKeys[key] = struct{}{}

		def := strings.TrimSpace(a.Definition)
		if i, ok := index[key]; ok {
			e := &g.Entries[i]
			e.Count++
			if e.FirstSeenAt == nil {
				e.FirstSeenAt = seenAt
			}
			e.LastSeenAt = seenAt
			// Prefer a longer non-empty definition.
			if def != "" && len(def) > len(strings.TrimSpace(e.Definition)) {
				e.Definition = def
			}
			continue
		}

		term := strings.TrimSpace(a.Term)
		g.Entries = append(g.Entries, GlossaryEntry{
			Term:        term,
			Definition:  def,
			Count:       1,
			FirstSeenAt: seenAt,
			LastSeenAt:  seenAt,
		})
		index[key] = len(g.Entries) - 1
	}

	// Keep stable ordering: highest count first, then term.
	sort.SliceStable(g.Entries, func(i, j int) bool {
		if g.Entries[i].Count != g.Entries[j].Count {
			return g.Entries[i].Count > g.Entries[j].Count
		}
		return strings.ToLower(g.Entries[i].Term) < strings.ToLower(g.Entries[j].Term)
	})

	terms := make([]string, 0, len(seenKeys))
	for key := range seenKeys {
		terms = append(terms, key)
	}
	sort.Strings(terms)
	return terms
}

// CullGlossary removes entries with Count < minCount.
func CullGlossary(g *Glossary, minCount int) {
	if g == nil || minCount <= 1 {
		return
	}
	out := g.Entries[:0]
	for _, e := range g.Entries {
		if e.Count >= minCount {
			out = append(out, e)
		}
	}
	g.Entries = out
}

func normalizeGlossaryKey(term string) string {
	term = strings.TrimSpace(term)
	if term == "" {
		return ""
	}
	return strings.ToLower(term)
}
