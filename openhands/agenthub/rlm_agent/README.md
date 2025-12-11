# RLM Agent (Recursive Language Model Agent)

The RLM Agent is an iterative problem-solving agent that alternates between three phases to solve tasks, allowing it to learn from previous attempts and improve its solutions over multiple iterations.

## Overview

The RLM Agent runs N iterations (configurable via `rlm_max_iterations` or `RLM_MAX_ITERATIONS` env var, default: 3), with each iteration consisting of three phases:

1. **ATTEMPT** - Work on the task
2. **CHARACTERIZE** - Analyze and summarize the attempt
3. **REFLECT** - Review all attempts, expand attempts of interest, and plan the next one

After completing all iterations, the agent selects and applies the best attempt.

## Phase Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ITERATION 1                                        │
│  ┌─────────┐    ┌──────────────┐    ┌─────────┐                             │
│  │ ATTEMPT │ -> │ CHARACTERIZE │ -> │ REFLECT │                             │
│  └─────────┘    └──────────────┘    └─────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    v (plan for next attempt)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ITERATION 2                                        │
│  ┌─────────┐    ┌──────────────┐    ┌─────────┐                             │
│  │ ATTEMPT │ -> │ CHARACTERIZE │ -> │ REFLECT │                             │
│  └─────────┘    └──────────────┘    └─────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    v
                                          ... more iterations ...
                                                    │
                                                    v
                                      ┌─────────────────────────┐
                                      │ SELECT & APPLY BEST     │
                                      │ ATTEMPT                 │
                                      └─────────────────────────┘
```

## Detailed Phase Descriptions

### ATTEMPT Phase

**System Prompt:** `system_prompt_attempt.j2`

**Context:** Full conversation history from the start of this attempt

**Available Tools:**
- `execute_bash` (if `enable_cmd`) - Run shell commands
- `think` (if `enable_think`) - Record reasoning
- `browser` (if `enable_browsing` and platform permits) - Web browsing
- `ipython` (if `enable_jupyter`) - Run Python code
- `task_tracker` (if `enable_plan_mode`) - Track tasks
- `llm_based_edit` (if `enable_llm_editor`) or `str_replace_editor` (if `enable_editor`) - Edit files
- `condensation_request` (if `enable_condensation_request`) - Request condensation
- `finish` - Complete the attempt with a summary

**Flow:**
```
ATTEMPT system prompt -> tool calls -> finish(summary)
```

**Behavior:**
- Works directly on the task using available tools
- Must call `finish` with a summary when done (successful or not)

### CHARACTERIZE Phase

**System Prompt:** `system_prompt_characterize.j2`

**Context:** Full history from the attempt (to see what was done) + CHARACTERIZE phase events

**Available Tools:**
- `execute_bash` - Run tests, profiling, validation
- `think` - Record analysis
- `finish_characterization` - Complete with a title + semantic summary

**Flow:**
```
CHARACTERIZE system prompt -> characterize_transition message (user) ->
tool calls (tests, profiling) -> finish_characterization(title, summary)
```

**Behavior:**
- Analyzes what was done in the attempt
- Runs validation (tests, linting, performance checks)
- Only the tools above are accepted in this phase; other tool calls are rejected
- Creates a comprehensive semantic summary including:
  - Title/label for the attempt
  - What was modified
  - Validation results
  - Confidence level (high/medium/low)
  - Limitations or notes

### REFLECT Phase

**System Prompt:** `system_prompt_reflect.j2`

**Context:** Fresh context with only REFLECT phase events (summaries of all attempts)

**Available Tools:**
- `browse_attempt` - Browse the full trajectory of a specific attempt
- `think` - Record reasoning
- `finish_reflection` - Complete with a plan for the next attempt
- `submit_attempt_as_final` - Submit a successful attempt as the final solution

**Flow:**
```
REFLECT system prompt -> attempt summaries (user) ->
browse_previous_attempts -> expand_previous_attempt(id) ... ->
finish_reflection(plan) OR submit_attempt_as_final(attempt_id)
```

**Behavior:**
- Reviews summaries of all previous attempts
- Can expand specific attempts for more detail
- Analyzes what worked and what didn't
- Creates a detailed plan for the next attempt
- OR submits a successful attempt as final (skips remaining iterations)

## Inline Reflection During ATTEMPT Phase

Inline reflection is not currently exposed in the ATTEMPT toolset. Browsing and expanding prior attempts happens only in the REFLECT phase via `browse_attempt`, which keeps the ATTEMPT context focused on active work.

## Context Management

Each phase has its own context to prevent confusion and context rot:

| Phase | System Prompt | Initial Context |
|-------|---------------|-----------------|
| ATTEMPT | `system_prompt_attempt.j2` | Full history + previous reflection insights |
| CHARACTERIZE | `system_prompt_characterize.j2` | Full history from attempt (to analyze what was done) |
| REFLECT | `system_prompt_reflect.j2` | Fresh - only REFLECT events + attempt summaries |

**Transition Messages:**
- All transition messages appear as **user messages** to properly instruct the LLM
- Each phase transition updates the `conversation_memory.prompt_manager` to use the correct system prompt
- CHARACTERIZE transition messages always include the attempt summary; REFLECT transitions send two user messages (summary list, then task)
- Responses are parsed with phase-aware tool allow-lists: any tool not offered in the current phase raises `FunctionCallNotExistsError`

## Best Attempt Selection

After completing all iterations (or when `submit_attempt_as_final` is called):

1. **LLM Reflection** (if available): Uses the LLM to evaluate which attempt is best based on summaries and characterizations
2. **Heuristic Fallback**: Returns the most recent completed attempt

The best attempt's patch is then applied (if `rlm_apply_patch_cmd` is configured).

## Configuration

| Config/Env Variable | Default | Description |
|---------------------|---------|-------------|
| `rlm_max_iterations` / `RLM_MAX_ITERATIONS` | 3 | Number of ATTEMPT→CHARACTERIZE→REFLECT iterations. If omitted, the agent also honors `max_iterations` provided via the runtime/CLI (e.g., `run_infer_rlm.sh --max-iterations`). |
| `rlm_extract_patch_cmd` | None | Command to extract patch after each attempt |
| `rlm_apply_patch_cmd` | None | Command to apply the best attempt's patch |


