package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/theimaginaryfoundation/compress-o-bot/migration"
)

// NOTE TO SELF -- add ISO 8601 date to the output files!! maybe month:year fields too for even easier search
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

	var allWritten []string
	for _, inFile := range inputFiles {
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
	}

	fmt.Fprintf(os.Stdout, "threads_processed=%d chunks_written=%d out_dir=%s\n", len(inputFiles), len(allWritten), cfg.OutputDir)
	for _, p := range allWritten {
		fmt.Fprintln(os.Stdout, p)
	}
}

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

var breakpointSchema = generateSchema[breakpointResponse]()

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

	resp, err := callWithRetry(ctx, d.client, params)
	if err != nil {
		return nil, err
	}

	var out breakpointResponse
	if err := decodeModelJSON(resp.OutputText(), &out); err != nil {
		// If the model output is truncated/invalid, fall back to deterministic breakpoints so the pipeline keeps moving.
		// This will typically produce ~targetTurnsPerChunk chunks.
		return fallbackBreakpoints(len(turns), targetTurnsPerChunk), nil
	}
	return out.Breakpoints, nil
}

const chunkBreakpointsPrompt = `You are a conversation segmentation assistant.

You will be given a JSON payload describing a conversation as a list of "turns".
A "turn" starts at a user message and includes any assistant/tool messages until the next user message.

Goal: return breakpoints (turn indices) where a NEW chunk should start, producing chunks that are:
- roughly target_turns_per_chunk turns each (15-25 is fine),
- aligned to complete "conversation loops" / topic boundaries when possible,
- not splitting in the middle of a coherent sub-task,
- using as few chunks as reasonable.

Rules:
- breakpoints must be strictly increasing integers
- each breakpoint must satisfy 1 <= breakpoint < total_turns
- DO NOT include 0
- If the thread is short, return an empty array.

Return only JSON matching the schema.`

func truncate(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	// Keep it readable: cut at rune boundary-ish (ASCII is fine here; most content is ASCII).
	return s[:max] + "…"
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
			td.User = truncate(t.UserText, 400)
			td.Assistant = truncate(t.AssistantText, 600)
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

func decodeModelJSON(outputText string, v any) error {
	s := strings.TrimSpace(outputText)
	if s == "" {
		return io.ErrUnexpectedEOF
	}

	if err := json.Unmarshal([]byte(s), v); err == nil {
		return nil
	}

	start := strings.IndexByte(s, '{')
	end := strings.LastIndexByte(s, '}')
	if start == -1 || end == -1 || end <= start {
		return fmt.Errorf("no JSON object found in model output (len=%d)", len(s))
	}
	sub := s[start : end+1]
	if err := json.Unmarshal([]byte(sub), v); err != nil {
		return fmt.Errorf("failed to unmarshal extracted JSON (len=%d): %w", len(sub), err)
	}
	return nil
}

func callWithRetry(ctx context.Context, client *openai.Client, params responses.ResponseNewParams) (*responses.Response, error) {
	const maxRetries = 3
	rateLimitWaitTimes := []time.Duration{65 * time.Second, 100 * time.Second, 135 * time.Second}
	serverErrorWaitTimes := []time.Duration{5 * time.Second, 30 * time.Second, 60 * time.Second}

	for attempt := 0; attempt < maxRetries; attempt++ {
		resp, err := client.Responses.New(ctx, params)
		if err != nil {
			if isRateLimitError(err) {
				if attempt < maxRetries-1 {
					time.Sleep(rateLimitWaitTimes[attempt])
					continue
				}
			} else if isServerError(err) {
				if attempt < maxRetries-1 {
					time.Sleep(serverErrorWaitTimes[attempt])
					continue
				}
			}
			return nil, err
		}
		return resp, nil
	}
	return nil, fmt.Errorf("failed after %d attempts due to OpenAI API issues", maxRetries)
}

func isRateLimitError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	return strings.Contains(errStr, "429") ||
		strings.Contains(errStr, "rate limit") ||
		strings.Contains(errStr, "too many requests")
}

func isServerError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	return strings.Contains(errStr, "500") ||
		strings.Contains(errStr, "internal server error") ||
		strings.Contains(errStr, "server_error")
}

// generateSchema is a small local copy of our structured-output JSON schema helper
// (compatible with OpenAI Structured Outputs' JSON schema subset).
func generateSchema[T any]() map[string]interface{} {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties:  false,
		DoNotReference:             true,
		RequiredFromJSONSchemaTags: true,
	}
	var v T
	schema := reflector.Reflect(v)
	schemaObj, err := schemaToMap(schema)
	if err != nil {
		panic(err)
	}
	ensureOpenAICompliance(schemaObj)
	return schemaObj
}

func schemaToMap(schema *jsonschema.Schema) (map[string]interface{}, error) {
	b, err := schema.MarshalJSON()
	if err != nil {
		return nil, err
	}
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	return m, nil
}

const (
	propertiesKey           = "properties"
	additionalPropertiesKey = "additionalProperties"
	typeKey                 = "type"
	requiredKey             = "required"
	itemsKey                = "items"
)

func ensureOpenAICompliance(schema map[string]interface{}) {
	if schemaType, ok := schema[typeKey].(string); ok && schemaType == "object" {
		schema[additionalPropertiesKey] = false

		if properties, ok := schema[propertiesKey].(map[string]interface{}); ok {
			var requiredFields []string
			for propName := range properties {
				requiredFields = append(requiredFields, propName)
			}
			if len(requiredFields) > 0 {
				schema[requiredKey] = requiredFields
			}
		}
	}

	if properties, ok := schema[propertiesKey].(map[string]interface{}); ok {
		for _, prop := range properties {
			if propMap, ok := prop.(map[string]interface{}); ok {
				ensureOpenAICompliance(propMap)
			}
		}
	}

	if items, ok := schema[itemsKey].(map[string]interface{}); ok {
		ensureOpenAICompliance(items)
	}

	if additionalProps, ok := schema[additionalPropertiesKey].(map[string]interface{}); ok {
		ensureOpenAICompliance(additionalProps)
	}
}
