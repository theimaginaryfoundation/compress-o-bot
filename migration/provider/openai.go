package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
)

func CallWithRetry(ctx context.Context, client *openai.Client, params responses.ResponseNewParams) (*responses.Response, error) {
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

func GenerateSchema[T any]() map[string]interface{} {
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
