package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/theimaginaryfoundation/compress-o-bot/migration"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/fileutils"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/provider"
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

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "missing OPENAI_API_KEY (or pass -api-key)")
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	client := openai.NewClient(option.WithAPIKey(apiKey))
	decider := openAIBreakpointDecider{
		client: &client,
		model:  cfg.Model,
	}

	inputFiles, err := collectInputFiles(cfg.InputPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}
	if len(inputFiles) == 0 {
		fmt.Fprintln(os.Stderr, "no input .json files found")
		os.Exit(2)
	}

	start := time.Now()
	var allWritten []string
	for i, inFile := range inputFiles {
		// To avoid filename collisions across threads (same thread_start_time), create a per-thread subdir.
		threadSubdir := filepath.Join(cfg.OutputDir, strings.TrimSuffix(filepath.Base(inFile), filepath.Ext(inFile)))

		written, err := migration.ChunkThread(ctx, inFile, decider, cfg.TargetTurns, migration.ChunkOptions{
			OutputDir:         threadSubdir,
			OverwriteExisting: cfg.Overwrite,
			Pretty:            cfg.Pretty,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed chunking %s: %s\n", inFile, err.Error())
			os.Exit(1)
		}
		allWritten = append(allWritten, written...)

		// Progress logging: this can take a long time and is otherwise mostly silent.
		fmt.Fprintf(os.Stderr, "progress thread-chunker: %d/%d threads chunked (last=%s chunks=%d elapsed=%s)\n",
			i+1, len(inputFiles), filepath.Base(inFile), len(written), time.Since(start).Round(time.Second))
	}

	fmt.Fprintf(os.Stdout, "threads_processed=%d chunks_written=%d out_dir=%s\n", len(inputFiles), len(allWritten), cfg.OutputDir)
	for _, p := range allWritten {
		fmt.Fprintln(os.Stdout, p)
	}
}

func parseFlags(fs *flag.FlagSet, args []string) (Config, error) {
	cfg := defaultConfig()
	fs.SetOutput(os.Stderr)

	fs.StringVar(&cfg.InputPath, "in", cfg.InputPath, "Path to a single simplified thread JSON file OR a directory containing thread JSON files")
	fs.StringVar(&cfg.OutputDir, "out", cfg.OutputDir, "Directory to write chunk JSON files into")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "OpenAI model to use for breakpoint detection (e.g. gpt-5-mini)")
	fs.IntVar(&cfg.TargetTurns, "target-turns", cfg.TargetTurns, "Target turns per chunk (a turn is user message + following assistant/tool messages)")
	fs.BoolVar(&cfg.Pretty, "pretty", false, "Pretty-print each chunk JSON file")
	fs.BoolVar(&cfg.Overwrite, "overwrite", false, "Overwrite existing chunk files")
	fs.StringVar(&cfg.APIKey, "api-key", "", "OpenAI API key (overrides OPENAI_API_KEY env var)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage:\n  %s [flags]\n\nFlags:\n", filepath.Base(os.Args[0]))
		fs.PrintDefaults()
		fmt.Fprintln(fs.Output(), "\nExample:")
		fmt.Fprintln(fs.Output(), "  go run ./cmd/thread-chunker -in docs/peanut-gallery/threads/<thread>.json -overwrite -pretty")
	}

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	cfg.InputPath = filepath.Clean(cfg.InputPath)
	cfg.OutputDir = filepath.Clean(cfg.OutputDir)
	return cfg, nil
}

func collectInputFiles(inputPath string) ([]string, error) {
	fi, err := os.Stat(inputPath)
	if err != nil {
		return nil, fmt.Errorf("stat -in: %w", err)
	}

	if !fi.IsDir() {
		if strings.ToLower(filepath.Ext(inputPath)) != ".json" {
			return nil, fmt.Errorf("input file must be .json: %s", inputPath)
		}
		return []string{inputPath}, nil
	}

	entries, err := os.ReadDir(inputPath)
	if err != nil {
		return nil, fmt.Errorf("read input dir: %w", err)
	}

	var files []string
	for _, e := range entries {
		if e.IsDir() {
			// Common: docs/peanut-gallery/threads/chunks/ exists alongside threads. Skip it.
			if strings.EqualFold(e.Name(), "chunks") {
				continue
			}
			continue
		}
		name := e.Name()
		if strings.ToLower(filepath.Ext(name)) != ".json" {
			continue
		}
		info, err := e.Info()
		if err != nil {
			return nil, fmt.Errorf("read dir entry info %s: %w", name, err)
		}
		if info.Mode()&fs.ModeType != 0 {
			continue
		}
		files = append(files, filepath.Join(inputPath, name))
	}
	sortStrings(files)
	return files, nil
}

func sortStrings(s []string) {
	// small local sort to avoid importing sort just for one call
	for i := 0; i < len(s); i++ {
		for j := i + 1; j < len(s); j++ {
			if s[j] < s[i] {
				s[i], s[j] = s[j], s[i]
			}
		}
	}
}

type openAIBreakpointDecider struct {
	client *openai.Client
	model  string
}

type breakpointRequest struct {
	ConversationID      string            `json:"conversation_id"`
	Title               string            `json:"title,omitempty"`
	TargetTurnsPerChunk int               `json:"target_turns_per_chunk"`
	TotalTurns          int               `json:"total_turns"`
	Turns               []turnForDecision `json:"turns"`
}

type turnForDecision struct {
	Turn      int      `json:"turn"`
	StartTime *float64 `json:"start_time,omitempty"`
	User      string   `json:"user,omitempty"`
	Assistant string   `json:"assistant,omitempty"`
}

type breakpointResponse struct {
	Breakpoints []int `json:"breakpoints"`
}

var breakpointSchema = provider.GenerateSchema[breakpointResponse]()

func (d openAIBreakpointDecider) DecideBreakpoints(ctx context.Context, thread migration.SimplifiedConversation, turns []migration.Turn, targetTurnsPerChunk int) ([]int, error) {
	if d.client == nil {
		return nil, errors.New("openAIBreakpointDecider: client is nil")
	}
	if d.model == "" {
		return nil, errors.New("openAIBreakpointDecider: model is empty")
	}

	payload, err := buildBreakpointRequestPayload(thread, turns, targetTurnsPerChunk)
	if err != nil {
		return nil, err
	}

	format := responses.ResponseFormatTextConfigUnionParam{
		OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name:        "TurnBreakpoints",
			Schema:      breakpointSchema,
			Strict:      openai.Bool(true),
			Description: openai.String("Turn breakpoints JSON"),
			Type:        "json_schema",
		},
	}

	instructions := chunkBreakpointsPrompt
	input := []responses.ResponseInputItemUnionParam{
		responses.ResponseInputItemParamOfMessage(string(payload), responses.EasyInputMessageRoleUser),
	}
	params := responses.ResponseNewParams{
		Model:           d.model,
		MaxOutputTokens: openai.Int(1500),
		Instructions:    openai.String(instructions),
		ServiceTier:     responses.ResponseNewParamsServiceTierFlex,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: input,
		},
		Text: responses.ResponseTextConfigParam{
			Format: format,
		},
	}

	resp, err := provider.CallWithRetry(ctx, d.client, params)
	if err != nil {
		return nil, err
	}

	var out breakpointResponse
	if err := fileutils.DecodeModelJSON(resp.OutputText(), &out); err != nil {
		// If the model output is truncated/invalid, fall back to deterministic breakpoints so the pipeline keeps moving.
		// This will typically produce ~targetTurnsPerChunk chunks.
		return fallbackBreakpoints(len(turns), targetTurnsPerChunk), nil
	}
	return out.Breakpoints, nil
}

func buildBreakpointRequestPayload(thread migration.SimplifiedConversation, turns []migration.Turn, targetTurnsPerChunk int) ([]byte, error) {
	// First attempt: include light text snippets.
	payload, err := json.Marshal(buildBreakpointRequest(thread, turns, targetTurnsPerChunk, true))
	if err != nil {
		return nil, err
	}

	// If the thread is huge, the request itself can exceed context limits. In that case, drop text entirely
	// (omitempty will remove user/assistant) and rely on structure-only segmentation.
	const maxRequestBytes = 250_000
	const maxTurnsWithText = 250
	if len(payload) > maxRequestBytes || len(turns) > maxTurnsWithText {
		return json.Marshal(buildBreakpointRequest(thread, turns, targetTurnsPerChunk, false))
	}

	return payload, nil
}

func buildBreakpointRequest(thread migration.SimplifiedConversation, turns []migration.Turn, targetTurnsPerChunk int, includeText bool) breakpointRequest {
	req := breakpointRequest{
		ConversationID:      thread.ConversationID,
		Title:               thread.Title,
		TargetTurnsPerChunk: targetTurnsPerChunk,
		TotalTurns:          len(turns),
		Turns:               make([]turnForDecision, 0, len(turns)),
	}

	for _, t := range turns {
		td := turnForDecision{
			Turn:      t.TurnIndex,
			StartTime: t.StartTime,
		}
		if includeText {
			td.User = fileutils.Truncate(t.UserText, 400)
			td.Assistant = fileutils.Truncate(t.AssistantText, 600)
		}
		req.Turns = append(req.Turns, td)
	}
	return req
}

func fallbackBreakpoints(totalTurns int, targetTurnsPerChunk int) []int {
	if targetTurnsPerChunk <= 0 || totalTurns <= targetTurnsPerChunk {
		return nil
	}
	var bps []int
	for i := targetTurnsPerChunk; i < totalTurns; i += targetTurnsPerChunk {
		bps = append(bps, i)
	}
	return bps
}
