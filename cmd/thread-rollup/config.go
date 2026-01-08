package main

import (
	"errors"
	"path/filepath"
)

type Config struct {
	InPath               string
	OutDir               string
	Model                string
	Pretty               bool
	Overwrite            bool
	APIKey               string
	IndexPath            string
	GlossaryPath         string
	GlossaryMaxTerms     int
	SentimentOutDir      string
	SentimentIndexPath   string
	SentimentModel       string
	Resume               bool
	Reindex              bool
	Concurrency          int
	MaxChunksPerThread   int
	IndexSummaryMaxChars int
	IndexTagsMax         int
	IndexTermsMax        int
}

func (c Config) Validate() error {
	if c.InPath == "" {
		return errors.New("missing -in")
	}
	if c.OutDir == "" {
		return errors.New("missing -out")
	}
	if c.Model == "" {
		return errors.New("missing -model")
	}
	if c.GlossaryMaxTerms < 0 {
		return errors.New("glossary-max-terms must be >= 0")
	}
	if c.Concurrency < 0 {
		return errors.New("concurrency must be >= 0")
	}
	if c.MaxChunksPerThread < 0 {
		return errors.New("max-chunks-per-thread must be >= 0")
	}
	if c.IndexSummaryMaxChars < 0 || c.IndexTagsMax < 0 || c.IndexTermsMax < 0 {
		return errors.New("index limits must be >= 0")
	}
	return nil
}

func defaultConfig() Config {
	return Config{
		InPath:               filepath.FromSlash("docs/peanut-gallery/threads/summaries"),
		OutDir:               filepath.FromSlash("docs/peanut-gallery/threads/thread_summaries"),
		Model:                "gpt-5-mini",
		GlossaryMaxTerms:     60,
		SentimentOutDir:      filepath.FromSlash("docs/peanut-gallery/threads/thread_sentiment_summaries"),
		SentimentModel:       "gpt-5-mini",
		Resume:               true,
		Reindex:              true,
		Concurrency:          6,
		MaxChunksPerThread:   5,
		IndexSummaryMaxChars: 600,
		IndexTagsMax:         5,
		IndexTermsMax:        15,
	}
}
