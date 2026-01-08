package main

import (
	"errors"
	"path/filepath"
)

type Config struct {
	InputPath   string
	OutputDir   string
	Model       string
	TargetTurns int
	Pretty      bool
	Overwrite   bool
	APIKey      string
}

func (c Config) Validate() error {
	if c.InputPath == "" {
		return errors.New("missing -in")
	}
	if c.OutputDir == "" {
		return errors.New("missing -out")
	}
	if c.Model == "" {
		return errors.New("missing -model")
	}
	if c.TargetTurns <= 0 {
		return errors.New("target turns must be > 0")
	}
	return nil
}

func defaultConfig() Config {
	return Config{
		InputPath:   "",
		OutputDir:   filepath.FromSlash("docs/peanut-gallery/threads/chunks"),
		Model:       "gpt-5-mini",
		TargetTurns: 20,
	}
}
