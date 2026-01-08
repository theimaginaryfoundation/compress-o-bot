package main

import (
	"errors"
	"path/filepath"
)

type Config struct {
	InPath           string
	OutDir           string
	IndexPath        string
	MaxBytes         int
	Overwrite        bool
	IncludeKeyPoints bool
	IncludeTags      bool
	Mode             string

	IndexSummaryMaxChars int
	IndexTagsMax         int
	IndexTermsMax        int
	IndexIncludeTags     bool
	IndexIncludeTerms    bool
}

func (c Config) Validate() error {
	if c.InPath == "" {
		return errors.New("missing -in")
	}
	if c.OutDir == "" {
		return errors.New("missing -out")
	}
	if c.MaxBytes <= 0 {
		return errors.New("max-bytes must be > 0")
	}
	return nil
}

func defaultConfig() Config {
	return Config{
		InPath:               filepath.FromSlash("docs/peanut-gallery/threads/thread_summaries"),
		OutDir:               filepath.FromSlash("docs/peanut-gallery/threads/memory_shards"),
		MaxBytes:             100 * 1024,
		IncludeKeyPoints:     true,
		IncludeTags:          true,
		Mode:                 "semantic",
		IndexSummaryMaxChars: 400,
		IndexTagsMax:         5,
		IndexTermsMax:        15,
		IndexIncludeTags:     true,
		IndexIncludeTerms:    true,
	}
}
