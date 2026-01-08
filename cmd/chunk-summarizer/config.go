package main

import (
	"errors"
	"path/filepath"
)

type Config struct {
	InPath              string
	OutDir              string
	Model               string
	SentimentModel      string
	SentimentPromptFile string
	Pretty              bool
	Overwrite           bool
	APIKey              string
	IndexPath           string
	SentimentIndexPath  string
	GlossaryPath        string
	GlossaryMaxTerms    int
	GlossaryMinCount    int
	MaxChunks           int

	Resume  bool
	Reindex bool

	Concurrency int
	BatchSize   int

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
	if c.SentimentModel == "" {
		return errors.New("missing -sentiment-model")
	}
	if c.GlossaryMaxTerms < 0 {
		return errors.New("glossary-max-terms must be >= 0")
	}
	if c.GlossaryMinCount < 0 {
		return errors.New("glossary-min-count must be >= 0")
	}
	if c.MaxChunks < 0 {
		return errors.New("max-chunks must be >= 0")
	}
	if c.Concurrency < 0 {
		return errors.New("concurrency must be >= 0")
	}
	if c.BatchSize < 0 {
		return errors.New("batch-size must be >= 0")
	}
	if c.IndexSummaryMaxChars < 0 || c.IndexTagsMax < 0 || c.IndexTermsMax < 0 {
		return errors.New("index limits must be >= 0")
	}
	return nil
}

func defaultConfig() Config {
	return Config{
		InPath:               filepath.FromSlash("docs/peanut-gallery/threads/chunks"),
		OutDir:               filepath.FromSlash("docs/peanut-gallery/threads/summaries"),
		Model:                "gpt-5-mini",
		SentimentModel:       "",
		GlossaryMaxTerms:     60,
		GlossaryMinCount:     2,
		Resume:               true,
		Reindex:              true,
		Concurrency:          6,
		BatchSize:            25,
		IndexSummaryMaxChars: 600,
		IndexTagsMax:         5,
		IndexTermsMax:        15,
	}
}
