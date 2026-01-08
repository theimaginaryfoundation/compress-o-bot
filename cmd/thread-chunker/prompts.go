package main

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
