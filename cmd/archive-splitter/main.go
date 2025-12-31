package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/theimaginaryfoundation/compress-o-bot/migration"
)

func main() {
	cfg, err := parseFlags(flag.CommandLine, os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}
	if err := cfg.Validate(); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	res, err := migration.SplitConversationArchive(ctx, cfg.InputPath, cfg.OutputDir, migration.SplitOptions{
		ArrayField:        cfg.ArrayField,
		OverwriteExisting: cfg.Overwrite,
		Pretty:            cfg.Pretty,
		DirMode:           0o755,
		FileMode:          0o644,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	fmt.Fprintf(os.Stdout, "threads_written=%d bytes_written=%d out_dir=%s\n", res.ThreadsWritten, res.BytesWritten, cfg.OutputDir)
}

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

func parseFlags(fs *flag.FlagSet, args []string) (Config, error) {
	cfg := defaultConfig()

	// Avoid mutating the global FlagSet if called from tests.
	fs.SetOutput(os.Stderr)

	fs.StringVar(&cfg.InputPath, "in", cfg.InputPath, "Path to conversations.json (OpenAI export)")
	fs.StringVar(&cfg.OutputDir, "out", cfg.OutputDir, "Directory to write per-thread JSON files into")
	fs.BoolVar(&cfg.Pretty, "pretty", false, "Pretty-print each output JSON file (more CPU/memory per thread)")
	fs.BoolVar(&cfg.Overwrite, "overwrite", false, "Overwrite existing output files")
	fs.StringVar(&cfg.ArrayField, "array-field", "", "If top-level JSON is an object, name of field containing conversations array (e.g. conversations)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage:\n  %s [flags]\n\nFlags:\n", filepath.Base(os.Args[0]))
		fs.PrintDefaults()
		fmt.Fprintln(fs.Output(), "\nExamples:")
		fmt.Fprintln(fs.Output(), "  go run ./cmd/archive-splitter -pretty -overwrite")
		fmt.Fprintln(fs.Output(), "  go run ./cmd/archive-splitter -in docs/peanut-gallery/conversations.json -out docs/peanut-gallery/threads")
	}

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}

	cfg.InputPath = filepath.Clean(cfg.InputPath)
	cfg.OutputDir = filepath.Clean(cfg.OutputDir)
	return cfg, nil
}
