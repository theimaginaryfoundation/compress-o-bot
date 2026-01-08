package main

import (
	"errors"
	"path/filepath"
)

func (c Config) Validate() error {
	if c.ConversationsPath == "" {
		return errors.New("missing -conversations")
	}
	if c.BaseDir == "" {
		return errors.New("missing -base-dir")
	}
	if c.Model == "" {
		return errors.New("missing -model")
	}
	if c.TargetTurns <= 0 {
		return errors.New("target-turns must be > 0")
	}
	if c.Concurrency < 0 || c.BatchSize < 0 || c.MaxChunks < 0 {
		return errors.New("concurrency/batch-size/max-chunks must be >= 0")
	}
	if c.MaxShardBytes <= 0 {
		return errors.New("max-shard-bytes must be > 0")
	}
	if c.IndexSummaryMaxChars < 0 || c.IndexTagsMax < 0 || c.IndexTermsMax < 0 {
		return errors.New("index limits must be >= 0")
	}
	if c.OnlyStage != "" && c.FromStage != "" {
		return errors.New("use only one of -only-stage or -from-stage")
	}
	return nil
}

func defaultConfig() Config {
	return Config{
		ConversationsPath:    filepath.FromSlash("docs/peanut-gallery/conversations.json"),
		BaseDir:              filepath.FromSlash("docs/peanut-gallery"),
		Model:                "gpt-5-mini",
		SentimentModel:       "",
		TargetTurns:          20,
		Concurrency:          6,
		BatchSize:            25,
		MaxChunks:            0,
		MaxShardBytes:        100 * 1024,
		IndexSummaryMaxChars: 600,
		IndexTagsMax:         5,
		IndexTermsMax:        15,
		Pretty:               false,
		Overwrite:            false,
	}
}
