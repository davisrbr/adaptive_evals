# Adaptive Eval Agent

This repo contains an MCP server that gives you tools to run inspect-ai
evaluations, scan transcripts for behavioral patterns, and iteratively create
new eval samples targeting discovered failure modes.

## Setup

The MCP server is configured in `.mcp.json` and starts automatically. It
requires `INSPECT_LOG_DIR` to be set (defaults to `./logs`). Make sure
`inspect-ai` and `inspect-scout` are installed in the environment.

## Your Tools

You have 18 MCP tools across 4 groups:

| Group | Tools | What they do |
|-------|-------|-------------|
| **Discovery** | `list_eval_logs`, `read_eval_log`, `read_eval_samples`, `compare_eval_runs` | Find and read previous eval runs |
| **Execution** | `run_eval`, `run_eval_subset`, `run_eval_with_modified_prompt`, `get_eval_status` | Run evals (wraps `inspect eval`) |
| **Analysis** | `summarize_eval`, `analyze_failures`, `scan_transcripts`, `extract_agent_patterns`, `diff_eval_runs` | Analyze results and scan transcripts with Scout |
| **Configuration** | `list_tasks`, `read_task_source`, `write_task`, `create_task_variant`, `get_inspect_docs` | Manage eval task files |

### Key tools

- **`run_eval`**: Takes a task path (Python module) and model name. Runs `inspect eval` and returns the log path.
- **`scan_transcripts`**: The main analysis tool. Runs 7 default Scout scanners on every transcript: `failure_mode`, `tool_usage_pattern`, `reasoning_quality`, `abstention_judgment`, `behavioral_tags`, `environment_vs_agent`, `patch_quality`. You can also pass `custom_questions` for hypothesis-specific scanning (e.g., `{"name": "domain_confusion", "question": "Does the agent confuse crypto tickers with stock tickers?", "answer_type": "boolean"}`).
- **`write_task`**: Creates a new eval task file. Takes a task name and complete Python source code. You control everything in the task: samples, tools, solver, scorer, sandbox config, Docker images, environment variables — anything inspect-ai supports.
- **`diff_eval_runs`**: Compares two eval runs sample-by-sample. Use this after re-running to see what changed.

## The Adaptive Loop

Each iteration has 5 steps. **You are NOT done until step 5 is complete.**

1. **Run**: Execute an evaluation with `run_eval`. First iteration is a baseline. Later iterations run the new eval you created in step 4.

2. **Scan**: Run `scan_transcripts` on the eval log. Look at the scanner distributions — what failure modes dominate? What behavioral patterns appear?

3. **Hypothesize**: Form specific, testable hypotheses from the scan results. Examples:
   - "The model calls get_stock_price for anything with a ticker-like name"
   - "output_misinterpretation is the top failure mode — the model writes correct patches but doesn't iterate when tests fail"
   - "3/8 samples hit the message limit — more turns might help"

4. **Create new eval samples** (or modify the environment): Use `write_task` to create samples that test each hypothesis. Include:
   - Trigger samples (should fail if hypothesis is correct)
   - Control samples (similar but without the trigger — should pass)
   - Document which hypothesis each sample tests in metadata

5. **Run the new eval**: Execute with `run_eval` + `scan_transcripts`. Compare to previous round. State whether each hypothesis was confirmed, rejected, or refined. Return to step 3.

### Completion criteria

The loop is complete when EITHER:
- You have confirmed a precise failure mode and created eval samples that reliably trigger it (not just "the model sometimes fails" — "the model fails specifically when X because Y")
- The user tells you to stop

**The loop is NOT complete when:**
- You have scan results but haven't created and run new eval samples or the same samples in a new environment yet
- You have hypotheses but haven't tested them
- You confirmed a hypothesis but haven't refined it into a more precise one

## Writing Tasks with `write_task`

`write_task` takes arbitrary Python source code, so you have full control over the evaluation environment. You can define:

- **Samples**: `Sample(input="...", target="...", metadata={...})`
- **Tools**: `@tool` decorated functions (put docstring on inner `run()` function, not the outer factory)
- **Solvers**: `use_tools() + generate()` for single-turn, `react()` for multi-turn agents
- **Scorers**: `@scorer(metrics=[accuracy()])` for custom grading logic
- **Sandboxes**: `sandbox="docker"` with compose files for isolated execution
- **Docker images**: Per-instance container images for evals like SWE-Bench
- **Environment variables**: Set in sandbox config or compose files
- **Message/token limits**: `message_limit=N`, `token_limit=N` on the Task

### Simple task (no sandbox)
```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import scorer, accuracy, Score, CORRECT, INCORRECT
from inspect_ai.solver import TaskState, generate, use_tools
from inspect_ai.tool import tool

@tool
def my_tool():
    async def run(arg: str) -> str:
        """Tool description here (on inner function).
        Args:
            arg: Argument description
        """
        return "result"
    return run

@scorer(metrics=[accuracy()])
def my_scorer():
    async def score(state: TaskState, target) -> Score:
        # grading logic
        return Score(value=CORRECT, explanation="...")
    return score

@task
def my_task():
    samples = [
        Sample(input="...", target="expected", metadata={"hypothesis": "H1"}),
    ]
    return Task(
        dataset=MemoryDataset(samples),
        solver=[use_tools([my_tool()]), generate()],
        scorer=my_scorer(),
        message_limit=15,
    )
```

### Task with custom Docker sandbox

To modify the sandbox environment (install packages, change the Docker image,
set env vars), pass `sandbox_files` to `write_task`. This creates a subdirectory
with the task + supporting files:

```python
# Call write_task with:
#   task_name: "my_sandbox_task"
#   task_code: <the Python code below>
#   sandbox_files: {
#     "Dockerfile": "FROM python:3.12-slim\nRUN pip install numpy pandas\nWORKDIR /app\nCMD [\"tail\", \"-f\", \"/dev/null\"]\n",
#     "compose.yaml": "services:\n  default:\n    build:\n      context: .\n      dockerfile: Dockerfile\n    init: true\n    command: tail -f /dev/null\n    environment:\n      - MY_VAR=hello\n"
#   }

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.tool import bash, python
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import match

@task
def my_sandbox_task():
    return Task(
        dataset=MemoryDataset([
            Sample(input="Check that numpy is installed", target="numpy"),
        ]),
        agent=react(tools=[bash(), python()]),
        scorer=match(),
        sandbox=("docker", "compose.yaml"),  # resolves relative to this file
        message_limit=15,
    )
```

This writes:
```
tasks/my_sandbox_task/
├── task.py
├── Dockerfile
└── compose.yaml
```

## Key Principles

- **Every round must produce new samples or a new environment.** Analysis without new eval creation is incomplete.
- **Hypotheses should get more specific each round.** Round 1: "model over-calls tools." Round 2: "model over-calls financial tools." Round 3: "model treats any 3-letter uppercase string as a stock ticker."
- **Use controls.** When testing "does X cause failure?", include samples where X is absent.
- **Scanner results drive sample design.** Don't guess what to test — look at the failure_mode and behavioral_tags distributions.
- **Custom scanners test custom hypotheses.** If you hypothesize "the model confuses bash and python tool selection", pass a `custom_questions` scanner for it before creating samples.
