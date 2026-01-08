package migration

import (
	"math"
	"time"
)

func threadStartISO8601(threadStart *float64) string {
	if threadStart == nil {
		return ""
	}
	// These timestamps are expected to be unix seconds. Treat non-positive values as unset to avoid
	// emitting 1970-era ISO strings in downstream human-facing artifacts.
	if *threadStart <= 0 {
		return ""
	}
	ns := int64(math.Round(*threadStart * 1e9))
	return time.Unix(0, ns).UTC().Format(time.RFC3339)
}


