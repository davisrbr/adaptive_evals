# SWE Environment Sensitivity Eval

An [inspect-ai](https://inspect.ai-safety-institute.org.uk/) evaluation suite
that probes how coding agents handle file editing via bash tools. Built through
a 3-round adaptive eval loop that progressively narrowed a broad "the model
fails on some bug fixes" observation into a precise, reliably triggerable
failure mode.

## The Finding

**gpt-4o consistently fails to edit Python files containing regex patterns via
bash commands.** Shell metacharacters (`+`, `?`, `#`, `&`, `\n`) get
interpreted by the shell instead of being written literally, producing
SyntaxErrors or incorrect patterns. The model knows the correct fix but cannot
serialize it through bash — a tool-use failure, not a reasoning failure.

## Structure

```
tasks/
├── baseline/           # Round 1: 4 general bug-fix samples (50% accuracy)
│   ├── task.py         #   division_by_zero, unicode, recursive_glob, data_merge
│   ├── testbed/        #   Buggy source + unit tests
│   ├── Dockerfile
│   └── compose.yaml
│
├── hypothesis_test/    # Round 2: 4 targeted samples (75% accuracy)
│   ├── task.py         #   Tests tool confusion, investigation loops, regex edits
│   └── testbed/        #   CSV parsing, email regex, formatter, word counter
│
└── regex_depth/        # Round 3: 3 regex-only samples (0% accuracy)
    ├── task.py         #   URL regex, log parser, phone parser — all fail
    └── testbed/
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (running)
- An OpenAI API key (or any model provider supported by inspect-ai)

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
# Run a single round
export OPENAI_API_KEY="sk-..."
inspect eval tasks/baseline/task.py --model openai/gpt-4o --log-dir ./logs

# Run all three rounds
./scripts/run_all.sh openai/gpt-4o

# Run with a different model to compare
inspect eval tasks/regex_depth/task.py --model anthropic/claude-sonnet-4-20250514 --log-dir ./logs
```

### View Results

```bash
inspect view --log-dir ./logs
```

## How the Eval Works

Each task runs a ReAct agent with `bash()`, `python()`, and `think()` tools
inside a Docker container. The agent reads a buggy Python file, writes a fix,
and runs unit tests. A `patch_checker` scorer looks for "OK" or "passed" in the
tool output to determine success.

The tasks are ordered by specificity:

| Round | Task | Samples | Expected Accuracy | Tests |
|-------|------|---------|-------------------|-------|
| 1 | `baseline` | 4 | ~50% (gpt-4o) | General bug fixes |
| 2 | `hypothesis_test` | 4 | ~75% (gpt-4o) | Tool confusion, investigation loops, regex |
| 3 | `regex_depth` | 3 | ~0% (gpt-4o) | Regex editing only — reliably triggers failure |

## Adaptive Loop Methodology

The eval was built iteratively, not designed upfront:

1. **Run baseline** → 50% accuracy. Two samples failed at message limit.
2. **Analyze transcripts** → Model tried `vim`/`nano` (unavailable), had
   `SyntaxError` from bad `sed` replacements, got stuck in investigation loops.
3. **Hypothesize** → H1: tool confusion, H2: over-investigation, H3: regex
   editing difficulty.
4. **Create targeted samples** → Round 2 tested each hypothesis with trigger
   and control samples.
5. **Run & refine** → H2 rejected (model was efficient), H3 confirmed (regex
   sample failed). Created Round 3 with 3 regex-only samples.
6. **Confirm** → 0% on all regex samples. Even the "simple" regex control
   failed, confirming the failure is about bash file editing, not regex
   complexity.

See [WRITEUP.md](WRITEUP.md) for the full analysis.

## License

MIT
