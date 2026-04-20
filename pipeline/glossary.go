package pipeline

import (
	"fmt"
	"strings"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
)

// GlossaryForPrompt formats the top maxTerms entries of g into a prompt-friendly string.
// Returns an empty string when maxTerms is 0 or the glossary is empty.
func GlossaryForPrompt(g migration.Glossary, maxTerms int) string {
	if maxTerms == 0 || len(g.Entries) == 0 {
		return ""
	}
	entries := g.Entries
	if maxTerms > 0 && len(entries) > maxTerms {
		entries = entries[:maxTerms]
	}
	var b strings.Builder
	for _, e := range entries {
		term := strings.TrimSpace(e.Term)
		if term == "" {
			continue
		}
		def := strings.TrimSpace(e.Definition)
		if def == "" {
			continue
		}
		fmt.Fprintf(&b, "- %s: %s\n", term, def)
	}
	return b.String()
}
