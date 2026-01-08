package main

import (
	"fmt"
	"path/filepath"
)

type Config struct {
	InputPath  string
	OutputDir  string
	ArrayField string
	Pretty     bool
	Overwrite  bool
}

func (c Config) Validate() error {
	if c.InputPath == "" {
		return fmt.Errorf("missing -in")
	}
	if c.OutputDir == "" {
		return fmt.Errorf("missing -out")
	}
	return nil
}

func defaultConfig() Config {
	return Config{
		InputPath: filepath.FromSlash("docs/peanut-gallery/conversations.json"),
		OutputDir: filepath.FromSlash("docs/peanut-gallery/threads"),
	}
}
