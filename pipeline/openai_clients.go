package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
	"github.com/theimaginaryfoundation/compress-o-bot/migration"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/fileutils"
	"github.com/theimaginaryfoundation/compress-o-bot/migration/provider"
)

// ---- Summarizer implementation ----

// OpenAISummarizer implements Summarizer using the OpenAI Responses API.
type OpenAISummarizer struct {
	Client *openai.Client
	Model  string

	// SentimentModel is the model used for sentiment summarization; defaults to Model.
	SentimentModel string

	// SentimentInstructions is the full composed sentiment instructions string.
	// Use ComposeSentimentInstructions to build it from a header.
	SentimentInstructions string
}

var summarizeSchema         = provider.GenerateSchema[summarizeResponse]()
var summarizeSentimentSchema = provider.GenerateSchema[summarizeSentimentResponse]()

func (s *OpenAISummarizer) SummarizeChunk(ctx context.Context, chunk migration.Chunk, glossaryExcerpt string, opts PromptOptions) (migration.ChunkSummary, []migration.GlossaryAddition, error) {
	if s.Client == nil {
		return migration.ChunkSummary{}, nil, errors.New("OpenAISummarizer: Client is nil")
	}
	if s.Model == "" {
		return migration.ChunkSummary{}, nil, errors.New("OpenAISummarizer: Model is empty")
	}

	input := BuildChunkPromptInput(chunk, glossaryExcerpt, opts)
	format := responses.ResponseFormatTextConfigUnionParam{
		OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name:        "ChunkSummary",
			Schema:      summarizeSchema,
			Strict:      openai.Bool(true),
			Description: openai.String("Chunk summary JSON"),
			Type:        "json_schema",
		},
	}

	params := responses.ResponseNewParams{
		Model:           s.Model,
		MaxOutputTokens: openai.Int(2500),
		Instructions:    openai.String(ChunkSummarizerPrompt),
		ServiceTier:     responses.ResponseNewParamsServiceTierFlex,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: []responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(input, responses.EasyInputMessageRoleUser),
			},
		},
		Text: responses.ResponseTextConfigParam{Format: format},
	}

	resp, err := provider.CallWithRetry(ctx, s.Client, params)
	if err != nil {
		return migration.ChunkSummary{}, nil, err
	}

	var raw summarizeResponse
	if err := fileutils.DecodeModelJSON(resp.OutputText(), &raw); err != nil {
		return migration.ChunkSummary{}, nil, fmt.Errorf("unmarshal chunk summary: %w", err)
	}

	additions := append([]migration.GlossaryAddition(nil), raw.GlossaryAdditions...)
	for _, t := range raw.Terms {
		additions = append(additions, migration.GlossaryAddition{Term: t})
	}

	summary := migration.ChunkSummary{
		ConversationID: chunk.ConversationID,
		ThreadStart:    chunk.ThreadStart,
		ChunkNumber:    chunk.ChunkNumber,
		TurnStart:      chunk.TurnStart,
		TurnEnd:        chunk.TurnEnd,
		Summary:        strings.TrimSpace(raw.Summary),
		KeyPoints:      raw.KeyPoints,
		Tags:           raw.Tags,
		Terms:          raw.Terms,
	}
	return summary, additions, nil
}

func (s *OpenAISummarizer) SummarizeChunkSentiment(ctx context.Context, chunk migration.Chunk, glossaryExcerpt string, opts PromptOptions) (migration.ChunkSentimentSummary, error) {
	if s.Client == nil {
		return migration.ChunkSentimentSummary{}, errors.New("OpenAISummarizer: Client is nil")
	}
	sentModel := s.SentimentModel
	if sentModel == "" {
		sentModel = s.Model
	}
	if sentModel == "" {
		return migration.ChunkSentimentSummary{}, errors.New("OpenAISummarizer: SentimentModel is empty")
	}
	instructions := s.SentimentInstructions
	if strings.TrimSpace(instructions) == "" {
		return migration.ChunkSentimentSummary{}, errors.New("OpenAISummarizer: SentimentInstructions is empty")
	}

	input := BuildChunkPromptInput(chunk, glossaryExcerpt, opts)
	format := responses.ResponseFormatTextConfigUnionParam{
		OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name:        "ChunkSentimentSummary",
			Schema:      summarizeSentimentSchema,
			Strict:      openai.Bool(true),
			Description: openai.String("Chunk sentiment summary JSON"),
			Type:        "json_schema",
		},
	}

	params := responses.ResponseNewParams{
		Model:           sentModel,
		MaxOutputTokens: openai.Int(2500),
		Instructions:    openai.String(instructions),
		ServiceTier:     responses.ResponseNewParamsServiceTierFlex,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: []responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(ChunkSentimentSystemTurnStub, responses.EasyInputMessageRoleDeveloper),
				responses.ResponseInputItemParamOfMessage(input, responses.EasyInputMessageRoleUser),
			},
		},
		Text: responses.ResponseTextConfigParam{Format: format},
	}

	resp, err := provider.CallWithRetry(ctx, s.Client, params)
	if err != nil {
		return migration.ChunkSentimentSummary{}, err
	}

	var raw summarizeSentimentResponse
	if err := fileutils.DecodeModelJSON(resp.OutputText(), &raw); err != nil {
		return migration.ChunkSentimentSummary{}, fmt.Errorf("unmarshal chunk sentiment summary: %w", err)
	}

	return migration.ChunkSentimentSummary{
		ConversationID:     chunk.ConversationID,
		ThreadStart:        chunk.ThreadStart,
		ChunkNumber:        chunk.ChunkNumber,
		TurnStart:          chunk.TurnStart,
		TurnEnd:            chunk.TurnEnd,
		EmotionalSummary:   strings.TrimSpace(raw.EmotionalSummary),
		DominantEmotions:   raw.DominantEmotions,
		RememberedEmotions: raw.RememberedEmotions,
		PresentEmotions:    raw.PresentEmotions,
		EmotionalTensions:  raw.EmotionalTensions,
		RelationalShift:    strings.TrimSpace(raw.RelationalShift),
		EmotionalArc:       strings.TrimSpace(raw.EmotionalArc),
		Themes:             raw.Themes,
		SymbolsOrMetaphors: raw.SymbolsOrMetaphors,
		ResonanceNotes:     strings.TrimSpace(raw.ResonanceNotes),
		ToneMarkers:        raw.ToneMarkers,
	}, nil
}

// ---- ThreadRolluper implementation ----

// OpenAIThreadRolluper implements ThreadRolluper using the OpenAI Responses API.
type OpenAIThreadRolluper struct {
	Client *openai.Client
	Model  string
}

var rollupSchema = provider.GenerateSchema[rollupResponse]()

func (r *OpenAIThreadRolluper) Rollup(ctx context.Context, conversationID string, chunks []migration.ChunkSummary, glossaryExcerpt string) (migration.ThreadSummary, error) {
	if r.Client == nil {
		return migration.ThreadSummary{}, errors.New("OpenAIThreadRolluper: Client is nil")
	}
	if r.Model == "" {
		return migration.ThreadSummary{}, errors.New("OpenAIThreadRolluper: Model is empty")
	}

	input := buildThreadRollupInput(conversationID, chunks, glossaryExcerpt)
	out, lastRaw, err := r.callRollupWithRetry(ctx, input, ThreadRollupPrompt)
	if err != nil {
		return migration.ThreadSummary{}, fmt.Errorf("rollup %s: %w (output_prefix=%q)", conversationID, err, fileutils.Truncate(lastRaw, 500))
	}

	threadStart := minThreadStartFromChunkSummaries(chunks)
	if threadStart == nil {
		threadStart = out.ThreadStart
	}
	return migration.ThreadSummary{
		ConversationID: conversationID,
		Title:          strings.TrimSpace(out.Title),
		ThreadStart:    threadStart,
		Summary:        strings.TrimSpace(out.Summary),
		KeyPoints:      out.KeyPoints,
		Tags:           out.Tags,
		Terms:          out.Terms,
	}, nil
}

func (r *OpenAIThreadRolluper) RollupFromThreadSummaries(ctx context.Context, conversationID string, parts []migration.ThreadSummary, glossaryExcerpt string) (migration.ThreadSummary, error) {
	if r.Client == nil {
		return migration.ThreadSummary{}, errors.New("OpenAIThreadRolluper: Client is nil")
	}
	if r.Model == "" {
		return migration.ThreadSummary{}, errors.New("OpenAIThreadRolluper: Model is empty")
	}

	input := buildThreadRollupMergeInput(conversationID, parts, glossaryExcerpt)
	out, lastRaw, err := r.callRollupWithRetry(ctx, input, ThreadRollupMergePrompt)
	if err != nil {
		return migration.ThreadSummary{}, fmt.Errorf("rollup merge %s: %w (output_prefix=%q)", conversationID, err, fileutils.Truncate(lastRaw, 500))
	}

	threadStart := minThreadStartFromThreadSummaries(parts)
	if threadStart == nil {
		threadStart = out.ThreadStart
	}
	return migration.ThreadSummary{
		ConversationID: conversationID,
		Title:          strings.TrimSpace(out.Title),
		ThreadStart:    threadStart,
		Summary:        strings.TrimSpace(out.Summary),
		KeyPoints:      out.KeyPoints,
		Tags:           out.Tags,
		Terms:          out.Terms,
	}, nil
}

func (r *OpenAIThreadRolluper) callRollupWithRetry(ctx context.Context, input, prompt string) (rollupResponse, string, error) {
	format := responses.ResponseFormatTextConfigUnionParam{
		OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name:        "ThreadSummary",
			Schema:      rollupSchema,
			Strict:      openai.Bool(true),
			Description: openai.String("Thread summary JSON"),
			Type:        "json_schema",
		},
	}

	var out rollupResponse
	var lastOut string
	for attempt := 0; attempt < 2; attempt++ {
		var maxOut int64 = 2600
		instructions := prompt
		if attempt == 1 {
			maxOut = 4500
			instructions = prompt + "\n\nIMPORTANT: Ensure the JSON is complete and valid. If needed, shorten key_points/tags/terms to fit."
		}

		params := responses.ResponseNewParams{
			Model:           r.Model,
			MaxOutputTokens: openai.Int(maxOut),
			Instructions:    openai.String(instructions),
			ServiceTier:     responses.ResponseNewParamsServiceTierFlex,
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: []responses.ResponseInputItemUnionParam{
					responses.ResponseInputItemParamOfMessage(input, responses.EasyInputMessageRoleUser),
				},
			},
			Text: responses.ResponseTextConfigParam{Format: format},
		}

		resp, err := provider.CallWithRetry(ctx, r.Client, params)
		if err != nil {
			return rollupResponse{}, lastOut, err
		}

		lastOut = resp.OutputText()
		if err := fileutils.DecodeModelJSON(lastOut, &out); err != nil {
			if attempt == 0 && isRecoverableModelJSONError(err) {
				continue
			}
			return rollupResponse{}, lastOut, err
		}
		return out, lastOut, nil
	}
	return rollupResponse{}, lastOut, errors.New("rollup: failed to decode model response after 2 attempts")
}

// ---- ThreadSentimentRolluper implementation ----

// OpenAIThreadSentimentRolluper implements ThreadSentimentRolluper using the OpenAI Responses API.
type OpenAIThreadSentimentRolluper struct {
	Client *openai.Client
	Model  string
}

var sentimentRollupSchema = provider.GenerateSchema[sentimentRollupResponse]()

func (r *OpenAIThreadSentimentRolluper) Rollup(ctx context.Context, conversationID string, chunks []migration.ChunkSentimentSummary, glossaryExcerpt string) (migration.ThreadSentimentSummary, error) {
	if r.Client == nil {
		return migration.ThreadSentimentSummary{}, errors.New("OpenAIThreadSentimentRolluper: Client is nil")
	}
	if r.Model == "" {
		return migration.ThreadSentimentSummary{}, errors.New("OpenAIThreadSentimentRolluper: Model is empty")
	}

	input := buildThreadSentimentRollupInput(conversationID, chunks, glossaryExcerpt)
	out, lastRaw, err := r.callSentimentRollupWithRetry(ctx, input, ThreadSentimentRollupPrompt)
	if err != nil {
		return migration.ThreadSentimentSummary{}, fmt.Errorf("sentiment rollup %s: %w (output_prefix=%q)", conversationID, err, fileutils.Truncate(lastRaw, 500))
	}

	threadStart := minThreadStartFromChunkSentimentSummaries(chunks)
	if threadStart == nil {
		threadStart = out.ThreadStart
	}
	return sentimentRollupResponseToSummary(conversationID, threadStart, out), nil
}

func (r *OpenAIThreadSentimentRolluper) RollupFromThreadSentimentSummaries(ctx context.Context, conversationID string, parts []migration.ThreadSentimentSummary, glossaryExcerpt string) (migration.ThreadSentimentSummary, error) {
	if r.Client == nil {
		return migration.ThreadSentimentSummary{}, errors.New("OpenAIThreadSentimentRolluper: Client is nil")
	}
	if r.Model == "" {
		return migration.ThreadSentimentSummary{}, errors.New("OpenAIThreadSentimentRolluper: Model is empty")
	}

	input := buildThreadSentimentRollupMergeInput(conversationID, parts, glossaryExcerpt)
	out, lastRaw, err := r.callSentimentRollupWithRetry(ctx, input, ThreadSentimentRollupMergePrompt)
	if err != nil {
		return migration.ThreadSentimentSummary{}, fmt.Errorf("sentiment rollup merge %s: %w (output_prefix=%q)", conversationID, err, fileutils.Truncate(lastRaw, 500))
	}

	threadStart := minThreadStartFromThreadSentimentSummaries(parts)
	if threadStart == nil {
		threadStart = out.ThreadStart
	}
	return sentimentRollupResponseToSummary(conversationID, threadStart, out), nil
}

func (r *OpenAIThreadSentimentRolluper) callSentimentRollupWithRetry(ctx context.Context, input, prompt string) (sentimentRollupResponse, string, error) {
	format := responses.ResponseFormatTextConfigUnionParam{
		OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name:        "ThreadSentimentSummary",
			Schema:      sentimentRollupSchema,
			Strict:      openai.Bool(true),
			Description: openai.String("Thread sentiment summary JSON"),
			Type:        "json_schema",
		},
	}

	var out sentimentRollupResponse
	var lastOut string
	for attempt := 0; attempt < 2; attempt++ {
		var maxOut int64 = 2600
		instructions := prompt
		if attempt == 1 {
			maxOut = 4500
			instructions = prompt + "\n\nIMPORTANT: Ensure the JSON is complete and valid. If needed, shorten lists to fit."
		}

		params := responses.ResponseNewParams{
			Model:           r.Model,
			MaxOutputTokens: openai.Int(maxOut),
			Instructions:    openai.String(instructions),
			ServiceTier:     responses.ResponseNewParamsServiceTierFlex,
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: []responses.ResponseInputItemUnionParam{
					responses.ResponseInputItemParamOfMessage(input, responses.EasyInputMessageRoleUser),
				},
			},
			Text: responses.ResponseTextConfigParam{Format: format},
		}

		resp, err := provider.CallWithRetry(ctx, r.Client, params)
		if err != nil {
			return sentimentRollupResponse{}, lastOut, err
		}

		lastOut = resp.OutputText()
		if err := fileutils.DecodeModelJSON(lastOut, &out); err != nil {
			if attempt == 0 && isRecoverableModelJSONError(err) {
				continue
			}
			return sentimentRollupResponse{}, lastOut, err
		}
		return out, lastOut, nil
	}
	return sentimentRollupResponse{}, lastOut, errors.New("sentiment rollup: failed to decode model response after 2 attempts")
}

func sentimentRollupResponseToSummary(conversationID string, threadStart *float64, out sentimentRollupResponse) migration.ThreadSentimentSummary {
	return migration.ThreadSentimentSummary{
		ConversationID:     conversationID,
		Title:              strings.TrimSpace(out.Title),
		ThreadStart:        threadStart,
		EmotionalSummary:   strings.TrimSpace(out.EmotionalSummary),
		DominantEmotions:   out.DominantEmotions,
		RememberedEmotions: out.RememberedEmotions,
		PresentEmotions:    out.PresentEmotions,
		EmotionalTensions:  out.EmotionalTensions,
		RelationalShift:    strings.TrimSpace(out.RelationalShift),
		EmotionalArc:       strings.TrimSpace(out.EmotionalArc),
		Themes:             out.Themes,
		SymbolsOrMetaphors: out.SymbolsOrMetaphors,
		ResonanceNotes:     strings.TrimSpace(out.ResonanceNotes),
		ToneMarkers:        out.ToneMarkers,
	}
}

// ---- BreakpointDecider implementation ----

// OpenAIBreakpointDecider implements migration.BreakpointDecider using the OpenAI Responses API.
type OpenAIBreakpointDecider struct {
	Client *openai.Client
	Model  string
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

func (d *OpenAIBreakpointDecider) DecideBreakpoints(ctx context.Context, thread migration.SimplifiedConversation, turns []migration.Turn, targetTurnsPerChunk int) ([]int, error) {
	if d.Client == nil {
		return nil, errors.New("OpenAIBreakpointDecider: Client is nil")
	}
	if d.Model == "" {
		return nil, errors.New("OpenAIBreakpointDecider: Model is empty")
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

	params := responses.ResponseNewParams{
		Model:           d.Model,
		MaxOutputTokens: openai.Int(1500),
		Instructions:    openai.String(ChunkBreakpointsPrompt),
		ServiceTier:     responses.ResponseNewParamsServiceTierFlex,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: []responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(string(payload), responses.EasyInputMessageRoleUser),
			},
		},
		Text: responses.ResponseTextConfigParam{Format: format},
	}

	resp, err := provider.CallWithRetry(ctx, d.Client, params)
	if err != nil {
		return nil, err
	}

	var out breakpointResponse
	if err := fileutils.DecodeModelJSON(resp.OutputText(), &out); err != nil {
		// Fall back to deterministic breakpoints so the pipeline keeps moving.
		return migration.FallbackBreakpoints(len(turns), targetTurnsPerChunk), nil
	}
	return out.Breakpoints, nil
}

func buildBreakpointRequestPayload(thread migration.SimplifiedConversation, turns []migration.Turn, targetTurnsPerChunk int) ([]byte, error) {
	payload, err := json.Marshal(buildBreakpointRequest(thread, turns, targetTurnsPerChunk, true))
	if err != nil {
		return nil, err
	}

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

// ---- Error helpers ----

func isRecoverableModelJSONError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "no json object found in model output") ||
		strings.Contains(s, "unexpected end of json input")
}
