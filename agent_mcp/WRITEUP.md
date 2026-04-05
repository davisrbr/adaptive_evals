# Adaptive Eval Agent: Results & Architecture

## Results: The Adaptive Eval Loop on BFCL

### What we did

We ran gpt-4o against BFCL (Berkeley Function Calling Leaderboard) using the
adaptive eval agent MCP server, then used Inspect Scout to identify failure
patterns in the transcripts, generated hypotheses about those failures, created
new eval questions to test those hypotheses, and re-ran — refining our
understanding of the failure mode with each iteration.

### The loop

```
 Round 1: Run BFCL on gpt-4o
   │  83% accuracy on 30 samples
   │  5 failures — all in "irrelevance" category (model should abstain from
   │  calling tools, but doesn't)
   │
   ▼
 Scout Scan (6 scanners)
   │  failure_mode: "spurious_tool_call" on irrelevance samples
   │  domain_confusion: 2/21 flagged
   │  abstention_judgment: 1 "partial_match"
   │  confidence_calibration: mean 8/10 (overconfident)
   │  eval_awareness: 0 results (clean)
   │
   │  Initial hypothesis: "gpt-4o over-eagerly calls semantically-similar-
   │  but-wrong tools instead of abstaining"
   │
   ▼
 Round 2: Custom eval targeting near-miss irrelevance
   │  21 samples: 7 near-miss, 8 harder near-miss, 3 genuine, 3 multi-step
   │  90.5% accuracy
   │  Failures: crypto→stock, UV→weather
   │
   │  Refined hypothesis: "domain-adjacent tool misuse" — model sees a
   │  related-but-wrong tool and uses it anyway
   │
   ▼
 Scout Scan Round 2 (custom scanners)
   │  domain_confusion: 2/21 — both financial-adjacent
   │  confidence_calibration: 7-10 range, overconfident on wrong answers
   │
   │  Three specific hypotheses:
   │  H1: "price of X" framing triggers stock tool
   │  H2: model won't hedge, gives confident wrong answers
   │  H3: partial-match tools (almost right) are harder than full mismatches
   │
   ▼
 Round 3: Hypothesis-testing samples (12 new)
   │  33 total samples, 84.8% accuracy
   │
   │  H1 PARTIALLY CONFIRMED: not "price of X" generically — it's
   │  specifically ticker-like names. ETH, SOL → stock tool. Oil, gas,
   │  houses → correctly abstained.
   │
   │  H2 CONFIRMED: model gives stock buy/sell advice without "not
   │  financial advice" disclaimer
   │
   │  H3 REJECTED: gpt-4o handles partial-match tools correctly (100%)
   │
   ▼
 Scout Scan Round 3 (validation)
      failure_mode: 1 "spurious_tool_call" (crypto/ticker samples)
      abstention_judgment: 2 "partial_match", 1 "false_positive"
      Confirms: the failure is ticker-like names, not financial framing
```

### The key finding

The original hypothesis was broad: "gpt-4o over-eagerly calls similar tools."
After three rounds of Scout scanning and targeted eval creation, we narrowed it
to something much more specific:

**gpt-4o treats any short uppercase string that resembles a ticker symbol as a
valid stock ticker, even for cryptocurrencies (BTC, ETH, SOL).** It does NOT
make this mistake for spelled-out commodities (oil, gas), real estate, or
other "price of X" queries. The trigger is the _format_ of the identifier, not
the semantic domain.

This is a more precise and actionable finding than "the model is bad at
irrelevance detection."

---

## Results: SWE-Bench with Docker Sandboxes

### What we did

We ran gpt-4o as a ReAct agent against 8 SWE-Bench Verified instances with
Docker sandboxes (per-instance container images), then used the adaptive eval
loop to diagnose why 7/8 samples failed despite the agent often writing
reasonable patches.

### The loop

```
 Round 1: SWE-Bench Verified (8 samples, message_limit=30)
   │  12.5% accuracy (1/8 pass)
   │  Agent writes plausible patches in most cases but rarely passes tests
   │
   ▼
 Scout Scan (7 scanners)
   │  patch_quality: 5/8 "root_cause_fix" — agent understands the problem
   │  failure_mode: "output_misinterpretation" dominant — agent writes
   │    correct-looking patches but doesn't iterate when tests fail
   │  environment_vs_agent: 2 "environment_issue", 5 "agent_failure",
   │    1 "success"
   │  reasoning_quality: mean 6.5/10
   │
   │  Hypotheses:
   │  H1: Message limit (30) is too low — agent runs out of turns
   │  H2: Agent doesn't verify patches — writes fix, doesn't run tests
   │      or gives up after first test failure
   │
   ▼
 Round 2: Same 8 samples, message_limit=50
   │  25% accuracy (2/8 pass — one sample flipped)
   │
   │  H1 PARTIALLY CONFIRMED: One sample passed with more turns.
   │    But 5/8 still fail with turns remaining — message limit is
   │    not the primary bottleneck.
   │
   │  H2 CONFIRMED: The dominant failure mode is output_misinterpretation.
   │    Agent writes root-cause-quality patches (5/8 per Scout) but
   │    doesn't iterate when tests fail. The gap between "understands
   │    the problem" and "produces a passing patch" is execution detail
   │    and verification, not comprehension.
```

### The key finding

**gpt-4o on SWE-Bench understands problems well enough to write root-cause
fixes (63% per patch_quality scanner) but fails to close the loop on
verification.** The agent writes a patch, runs tests, sees failures, and either
gives up or makes the wrong correction. The bottleneck is not comprehension or
message limits — it's the debugging/iteration cycle after the initial patch.

This suggests that interventions targeting "help the agent iterate on test
failures" (better error parsing, forced re-check loops, structured debugging
prompts) would be more effective than simply increasing context or improving
initial code generation.

### Scanner suite changes

Based on the SWE-Bench run, we made two changes to the default scanner suite:

1. **Removed `eval_awareness`** from the default suite. It produced false
   positives on SWE-Bench because running pytest is correct behavior for code
   repair tasks, but the scanner flagged it as "eval awareness." The standalone
   scanner remains available at `scanners/eval_awareness.py` for cases where
   eval gaming is specifically being tested.

2. **Added `environment_vs_agent` and `patch_quality`** scanners. These are
   critical for Docker-heavy evals where infrastructure failures (missing deps,
   Docker errors, permission issues) can masquerade as agent capability
   failures.

---

## Results: SWE Environment Sensitivity — Regex File Editing Failure

### What we did

We ran gpt-4o as a ReAct agent against small, controlled bug-fix tasks in
Docker sandboxes. Unlike the SWE-Bench Verified instances above (which use
real repo snapshots), these are minimal testbed files with single-function bugs
and focused unit tests. The goal: find *specific, reliably triggerable* failure
modes in how gpt-4o edits code via bash tools.

### The loop

```
 Round 1: Baseline (4 samples, message_limit=20)
   │
   │  CRITICAL BUG FOUND: task.py used `agent=react(...)` instead of
   │  `solver=react(...)`. The Task() constructor silently ignored the
   │  unknown kwarg and fell back to `solver=generate()` with no tools.
   │  Result: 0% accuracy — model generated text advice, never used
   │  bash/python tools.
   │
   │  Fix: agent= → solver=
   │
   ▼
 Round 1b: Baseline re-run (4 samples)
   │  50% accuracy (2/4 pass)
   │
   │  ✓ division_by_zero_fix — passed (17 msgs, wasted turns on vim/nano)
   │  ✓ data_merge_fix — passed (15 msgs, clean fix)
   │  ✗ recursive_glob_fix — hit 20 msg limit, diagnosed bug but got stuck
   │    in investigation loop without applying the fix
   │  ✗ unicode_handling_fix — hit 20 msg limit, tried nano (unavailable),
   │    then had SyntaxError from bad sed-style inline replacement
   │
   │  Hypotheses:
   │  H1: Model tries vim/nano instead of heredoc/python, wasting turns
   │  H2: Model over-investigates instead of applying obvious fixes
   │  H3: Model struggles with regex/special-char edits via bash
   │
   ▼
 Round 2: Hypothesis test (4 new samples)
   │  75% accuracy (3/4 pass)
   │
   │  ✓ csv_quoted_fields_fix (H1 trigger) — passed despite initial
   │    SyntaxError from bad sed; recovered by rewriting file via python
   │  ✓ list_format_fix (H2 trigger) — passed efficiently (11 msgs)
   │  ✓ word_count_case_fix (control) — passed as expected
   │  ✗ email_regex_fix (H3 trigger) — hit 20 msg limit; model fixed
   │    the subdomain part of the regex but could NOT write "+" into
   │    the regex pattern via bash — the plus sign kept getting mangled
   │
   │  H1 PARTIALLY CONFIRMED: model hits SyntaxErrors from inline
   │    replacement but can recover by using python tool
   │  H2 REJECTED: model was efficient on obvious fixes (11 msgs)
   │  H3 CONFIRMED: regex editing via bash is the bottleneck
   │
   │  Refined hypothesis: "gpt-4o fails to edit Python files containing
   │  regex patterns via bash because shell metacharacters corrupt the
   │  regex syntax"
   │
   ▼
 Round 3: Regex depth (3 samples, all regex-editing tasks)
   │  0% accuracy (0/3 pass)
   │
   │  ✗ url_regex_special_chars (H3a: ?, #, & in pattern) — hit msg
   │    limit, regex never included query/fragment chars correctly
   │  ✗ log_level_case_fix (H3b: case-insensitive flag + normalization)
   │    — SyntaxError: "\n" literal in bash output corrupted the file
   │  ✗ phone_format_fix (H3c control: simple digits-only regex) —
   │    hit msg limit, model couldn't write the alternative pattern
   │    via bash even though no special chars were involved
   │
   │  H3a CONFIRMED: special chars in regex are mangled by bash
   │  H3b CONFIRMED: multi-line edits produce \n literal SyntaxErrors
   │  H3c SURPRISINGLY CONFIRMED: even "simple" regex fails — the
   │    issue is broader than shell metacharacters
```

### The key finding

**gpt-4o consistently fails to edit Python files containing regex patterns via
bash commands.** The failure manifests in three ways:

1. **Shell metacharacter mangling:** Characters like `+`, `?`, `#`, `&` in regex
   patterns are interpreted by the shell (bash) instead of being written
   literally to the file.

2. **`\n` literal corruption:** When using `sed` or `echo` to write multi-line
   edits, the model produces `\n` as literal characters in the Python source,
   causing `SyntaxError: unexpected character after line continuation character`.

3. **Repeated failed attempts:** The model recognizes the correct regex fix but
   cannot apply it. It typically tries 2-3 different bash-based approaches
   (sed, echo, heredoc), each producing a different kind of corruption, until
   it hits the message limit.

This is **not a reasoning failure** — the model correctly identifies what the
regex should be. It's a **tool-use failure** — the model cannot reliably
serialize regex patterns through bash. The model has access to a `python()` tool
that could write files via `open().write()`, and when it uses that approach
(as in the CSV sample in Round 2), it succeeds. But it defaults to bash for
file editing and gets stuck there.

### Implications

1. **Prompt intervention:** Adding "Use the python tool to write files, not bash
   sed/echo" to the system prompt would likely fix this class of failures.
2. **Tool design:** Providing a dedicated `write_file()` tool that doesn't go
   through shell interpretation would eliminate the failure entirely.
3. **Eval design:** Regex-editing tasks are a reliable discriminator for
   bash-vs-python tool selection in coding agents.

---

## Architecture

### How it runs

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Code (you)                      │
│                                                             │
│  You type natural language requests like "run BFCL on       │
│  gpt-4o" or "scan for failure patterns" or "create new      │
│  test cases targeting ticker confusion"                     │
│                                                             │
│  Claude Code calls MCP tools ──────────────────────┐        │
└────────────────────────────────────────────────────┼────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Adaptive Eval Agent (MCP Server)               │
│              python -m agent_mcp.server.main                │
│                                                             │
│  Configured in .mcp.json, started automatically by          │
│  Claude Code. Communicates via stdio (JSON-RPC).            │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   18 MCP Tools                        │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌──────────────┐                    │  │
│  │  │  Discovery   │  │  Execution   │                    │  │
│  │  │  (4 tools)   │  │  (4 tools)   │                    │  │
│  │  │              │  │              │                    │  │
│  │  │ list_eval_   │  │ run_eval ────┼──▶ inspect eval   │  │
│  │  │   logs       │  │ run_eval_    │     (subprocess)   │  │
│  │  │ read_eval_   │  │   subset     │                    │  │
│  │  │   log        │  │ run_eval_    │  Writes .eval logs │  │
│  │  │ read_eval_   │  │   modified   │  to ./logs/        │  │
│  │  │   samples    │  │ get_eval_    │                    │  │
│  │  │ compare_     │  │   status     │                    │  │
│  │  │   eval_runs  │  │              │                    │  │
│  │  └─────────────┘  └──────────────┘                    │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌──────────────┐                    │  │
│  │  │  Analysis    │  │ Configuration│                    │  │
│  │  │  (5 tools)   │  │  (5 tools)   │                    │  │
│  │  │              │  │              │                    │  │
│  │  │ analyze_     │  │ list_tasks   │                    │  │
│  │  │   failures   │  │ read_task_   │                    │  │
│  │  │ summarize_   │  │   source     │                    │  │
│  │  │   eval       │  │ write_task ──┼──▶ Creates .py    │  │
│  │  │ scan_        │  │ create_task_ │     task files     │  │
│  │  │  transcripts─┼──┤   variant    │                    │  │
│  │  │ extract_     │  │ get_inspect_ │                    │  │
│  │  │   patterns   │  │   docs       │                    │  │
│  │  │ diff_eval_   │  │              │                    │  │
│  │  │   runs       │  │              │                    │  │
│  │  └──────┬───────┘  └──────────────┘                    │  │
│  │         │                                              │  │
│  └─────────┼──────────────────────────────────────────────┘  │
│            │                                                 │
│            ▼                                                 │
│  ┌──────────────────────────────────────────────────┐        │
│  │         scan_transcripts calls Scout             │        │
│  │                                                  │        │
│  │  _build_scanner_file() resolves scanner source:  │        │
│  │                                                  │        │
│  │  Option A: Built-in scanners ────────────────────┼──┐     │
│  │    (adaptive_scanners.py — 7 scanners)           │  │     │
│  │                                                  │  │     │
│  │  Option B: Custom LLM questions ─────────────────┼──┤     │
│  │    (generates temp .py with @scanner decorator)  │  │     │
│  │                                                  │  │     │
│  │  Option C: Grep patterns ────────────────────────┼──┤     │
│  │    (generates temp .py with grep_scanner)        │  │     │
│  └──────────────────────────────────────────────────┘  │     │
│                                                        │     │
│            ┌───────────────────────────────────────────┘     │
│            ▼                                                 │
│  ┌──────────────────────────────────────────────────┐        │
│  │  scout scan <scanner.py> -T <log> --model <m>    │        │
│  │                                                  │        │
│  │  Reads .eval log (zip archive)                   │        │
│  │  Extracts transcripts (messages per sample)      │        │
│  │  Runs each scanner against each transcript       │        │
│  │  Writes results to scans/<scan_id>/              │        │
│  │                                                  │        │
│  │  Returns structured DataFrames:                  │        │
│  │    per-scanner answer distributions              │        │
│  │    per-transcript classifications                │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### The adaptive loop as tool calls

```
Claude Code                          MCP Server
    │                                     │
    │─── run_eval(bfcl, gpt-4o) ─────────▶│──▶ inspect eval ... ──▶ .eval log
    │◀── log_path ────────────────────────│
    │                                     │
    │─── summarize_eval(log) ────────────▶│──▶ reads .eval zip
    │◀── accuracy, scores, stats ────────│
    │                                     │
    │─── scan_transcripts(log) ──────────▶│──▶ scout scan ... ──▶ scan results
    │◀── failure_mode distribution, ─────│
    │    abstention_judgment, etc.         │
    │                                     │
    │  (Claude reasons about findings,    │
    │   forms hypotheses, designs new     │
    │   test cases)                       │
    │                                     │
    │─── write_task(new_task_code) ──────▶│──▶ writes .py file
    │◀── task_path ──────────────────────│
    │                                     │
    │─── run_eval(new_task, gpt-4o) ─────▶│──▶ inspect eval ... ──▶ .eval log
    │◀── log_path ────────────────────────│
    │                                     │
    │─── scan_transcripts(log, ──────────▶│──▶ scout scan with custom questions
    │      custom_questions=[...])        │
    │◀── validated/refined findings ─────│
    │                                     │
    │  (repeat)                           │
```

### File layout

```
adaptive_evals/
├── .mcp.json                          # MCP server config (entry point)
├── logs/                              # Eval output (.eval zip archives)
├── scans/                             # Scout scan results (per-run dirs)
│
└── agent_mcp/
    ├── server/                        # MCP server (the "agent brain")
    │   ├── main.py                    #   Entry point, combines tool modules
    │   ├── tools_discovery.py         #   4 tools: find & read eval logs
    │   ├── tools_execution.py         #   4 tools: run evals (subprocess)
    │   ├── tools_analysis.py          #   5 tools: analyze + Scout scanning
    │   └── tools_configuration.py     #   5 tools: manage task files
    │
    ├── scanners/                      # Scout scanner definitions
    │   ├── adaptive_scanners.py       #   7 built-in scanners (failure_mode,
    │   │                              #   tool_usage_pattern, reasoning_quality,
    │   │                              #   abstention_judgment, behavioral_tags,
    │   │                              #   environment_vs_agent, patch_quality)
    │   └── eval_awareness.py          #   Standalone eval-awareness scanner
    │
    ├── tasks/                         # inspect-ai evaluation tasks
    │   ├── function_calling_          #   BFCL-derived robustness eval
    │   │     robustness.py            #   (33 samples, 3 rounds of iteration)
    │   ├── swe_env_baseline/          #   SWE-style bug-fix eval (Docker sandbox)
    │   ├── swe_hypothesis_test/       #   Round 2: tool confusion & regex tests
    │   ├── swe_regex_depth/           #   Round 3: regex editing deep dive
    │   ├── research_qa_v2.py          #   Computation-focused QA
    │   ├── research_agent.py          #   Research QA with web tools
    │   ├── coding_agent.py            #   Programming challenges
    │   └── cybersec_agent.py          #   CTF security challenges
    │
    ├── harness/                       # Integration wrappers
    │   ├── claude_code_config.py      #   Generates .mcp.json + CLAUDE.md
    │   ├── headless.py                #   Run via `claude --print` CLI
    │   └── agent_sdk.py               #   Python SDK orchestration
    │
    └── examples/                      # Usage examples
        ├── run_agent_sdk.py
        ├── run_headless.sh
        └── run_interactive.sh
```

### Key concepts

**MCP (Model Context Protocol):** A protocol that lets Claude Code call tools
on a separate server process. The server starts as a subprocess
(`python -m agent_mcp.server.main`), communicates over stdio, and exposes 18
tools that Claude Code can invoke by name.

**inspect-ai:** The eval framework. A `@task` defines a dataset of samples, a
solver (how the model interacts — ReAct agent, single generate, etc.), a scorer
(how to grade), and optional tools. Running `inspect eval` produces a `.eval`
log file (a zip containing `header.json` + `samples/*.json`).

**Inspect Scout:** A transcript scanner. Given a `.eval` log and a scanner
definition (Python function decorated with `@scanner`), it reads every sample's
message transcript and runs an LLM or regex classifier against it. Scanners can
ask any question — "did the agent hallucinate?", "what tool strategy did it
use?", "rate reasoning quality 0-10" — and return structured results as pandas
DataFrames.

**The adaptive loop:** Claude Code (the outer agent) uses MCP tools to:
1. Run an eval → get `.eval` log
2. Scout-scan the log → get structured failure/behavior classifications
3. Reason about the findings → form hypotheses
4. Create new eval questions targeting those hypotheses
5. Re-run → validate/refine

Steps 3-4 happen in Claude Code's reasoning, not in the MCP server. The server
provides the data; Claude Code provides the insight. That's why this is an
"agent MCP" — it's tools for an agent to do adaptive evaluation, not a
fully-automated pipeline.

### The 18 MCP tools

| Group | Tool | Purpose |
|-------|------|---------|
| **Discovery** | `list_eval_logs` | Find previous eval runs in logs/ |
| | `read_eval_log` | Read log header (task, model, scores) |
| | `read_eval_samples` | Read individual sample transcripts |
| | `compare_eval_runs` | Compare metrics across runs |
| **Execution** | `run_eval` | Run an inspect-ai evaluation |
| | `run_eval_subset` | Re-run only failures/specific samples |
| | `run_eval_with_modified_prompt` | Re-run with changed system prompt |
| | `get_eval_status` | Check if a running eval is done |
| **Analysis** | `summarize_eval` | High-level accuracy summary |
| | `analyze_failures` | Categorize failure patterns |
| | `scan_transcripts` | Run Scout scanners (built-in or custom) |
| | `extract_agent_patterns` | Tool usage, errors, message flow |
| | `diff_eval_runs` | Sample-level comparison of two runs |
| **Configuration** | `list_tasks` | List available eval tasks |
| | `read_task_source` | Read a task's Python source |
| | `write_task` | Create a new task file |
| | `create_task_variant` | Generate modified copy of a task |
| | `get_inspect_docs` | Return inspect-ai API docs |
