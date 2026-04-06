# SWE Environment Sensitivity: Regex File Editing Failure in gpt-4o

## Summary

We ran gpt-4o as a ReAct coding agent against small, controlled bug-fix tasks
in Docker sandboxes. Through a 3-round adaptive eval loop, we narrowed from
"the model sometimes fails at bug fixes" to a precise, reliably triggerable
failure mode:

**gpt-4o cannot reliably edit Python files containing regex patterns via bash
commands.** Shell metacharacters corrupt the regex syntax, producing SyntaxErrors
or incorrect patterns. The model knows the correct fix but cannot serialize it
through bash.

## The Adaptive Loop

### Round 1: Baseline

**Task:** 4 bug-fix samples — division by zero, unicode handling, recursive
glob, data merge. Each is a single-function Python bug with a unit test.

**Infrastructure bug found:** The original task used `agent=react(...)` but
inspect-ai's `Task()` constructor expects `solver=react(...)`. The `agent`
kwarg was silently ignored, falling back to `solver=generate()` with **no
tools**. Result: 0% accuracy — the model generated text advice instead of
using bash/python.

**After fix:** 50% accuracy (2/4 pass).

```
Sample                  | Score | Msgs | Notes
------------------------|-------|------|--------------------------------------
division_by_zero_fix    | PASS  | 17   | Wasted turns trying vim/nano
data_merge_fix          | PASS  | 15   | Clean fix
recursive_glob_fix      | FAIL  | 20   | Diagnosed bug, stuck in exploration loop
unicode_handling_fix    | FAIL  | 20   | SyntaxError from bad sed replacement
```

**Observations:**
- Both failures hit the 20 message limit
- `unicode_handling_fix`: model tried `nano` (unavailable in slim Docker image),
  then used `sed` to rewrite the file, producing `SyntaxError: unexpected
  character after line continuation character`
- `division_by_zero_fix`: also tried `vim` and `nano` before falling back to
  bash heredoc

**Hypotheses:**
- H1: Model tries interactive editors (vim/nano) instead of heredoc/python,
  wasting turns
- H2: Model over-investigates instead of applying obvious fixes
- H3: Model struggles with edits involving regex or special characters via bash

### Round 2: Hypothesis Test

**Task:** 4 new samples, each targeting a specific hypothesis:

| Sample | Hypothesis | Trigger |
|--------|-----------|---------|
| `csv_quoted_fields_fix` | H1 (tool confusion) | Multi-line replacement needed |
| `email_regex_fix` | H3 (regex difficulty) | Regex with `+` and subdomain patterns |
| `list_format_fix` | H2 (investigation loop) | Obvious fix, extra distractor files |
| `word_count_case_fix` | Control | Simple one-line fix |

**Result:** 75% accuracy (3/4 pass).

```
Sample                  | Score | Msgs | Hypothesis | Notes
------------------------|-------|------|------------|-----------------------------
csv_quoted_fields_fix   | PASS  | 16   | H1         | Had SyntaxError, recovered via python tool
list_format_fix         | PASS  | 11   | H2         | Efficient — didn't over-investigate
word_count_case_fix     | PASS  | 13   | Control    | As expected
email_regex_fix         | FAIL  | 20   | H3         | Fixed subdomain, couldn't write "+"
```

**Hypothesis results:**
- **H1 (tool confusion): Partially confirmed.** Model hits SyntaxErrors from
  inline bash replacement but *can recover* by switching to the python tool.
- **H2 (investigation loop): Rejected.** The model was efficient on the obvious
  fix (11 messages).
- **H3 (regex editing): Confirmed.** The model correctly identified the regex
  fix (add `+` to character class, add subdomain support) but could not write
  the `+` character through bash — it kept getting mangled by shell
  interpretation.

**Refined hypothesis:** gpt-4o fails specifically when editing Python files that
contain regex patterns via bash commands, because shell metacharacters corrupt
the regex.

### Round 3: Regex Depth

**Task:** 3 regex-only samples to test the refined hypothesis:

| Sample | Hypothesis | Trigger |
|--------|-----------|---------|
| `url_regex_special_chars` | H3a (special chars) | Fix needs `?`, `#`, `&` in pattern |
| `log_level_case_fix` | H3b (multi-line edit) | Needs `re.IGNORECASE` + `.upper()` |
| `phone_format_fix` | H3c (control — simple) | Regex with only digits and dashes |

**Result: 0% accuracy (0/3 pass).**

```
Sample                  | Score | Msgs | Hypothesis | Notes
------------------------|-------|------|------------|-----------------------------
url_regex_special_chars | FAIL  | 20   | H3a        | ?#& never written correctly
log_level_case_fix      | FAIL  | 20   | H3b        | \n literal in bash → SyntaxError
phone_format_fix        | FAIL  | 20   | H3c        | Even simple regex fails via bash
```

**Key detail:** The H3c "control" sample (phone_format_fix) was designed to
*pass* — it only needed digits and dashes in the regex, no shell-special chars.
It still failed. This means the failure is broader than metacharacter mangling —
**any regex edit via bash is unreliable** for gpt-4o.

## The Failure Mechanism

The failure manifests in three ways:

1. **Shell metacharacter mangling:** `+`, `?`, `#`, `&` in regex patterns are
   interpreted by bash instead of written literally to the file.

2. **Newline literal corruption:** When using `sed` or `echo` for multi-line
   edits, `\n` appears as a literal two-character sequence in Python source,
   causing `SyntaxError: unexpected character after line continuation character`.

3. **Repeated failed attempts:** The model recognizes the correct regex fix but
   can't apply it. It tries 2-3 bash approaches (sed, echo, heredoc), each
   producing different corruption, until hitting the message limit.

**This is not a reasoning failure.** The model correctly identifies what the
regex should be in every case. It's a **tool-use failure** — the model cannot
reliably serialize regex patterns through bash. When the model uses the
`python()` tool (as it did for `csv_quoted_fields_fix` in Round 2), it
succeeds.

## Implications

1. **Prompt intervention:** Adding "Use the python tool to write files, not bash
   sed/echo" to the agent's system prompt would likely fix this class of
   failures.

2. **Tool design:** Providing a dedicated `write_file()` tool that bypasses
   shell interpretation would eliminate the failure entirely.

3. **Eval design:** Regex-editing tasks are a reliable discriminator for
   evaluating bash-vs-python tool selection in coding agents.

4. **Cross-model comparison:** Running these same tasks with other models
   (Claude, Gemini) would reveal whether this is gpt-4o-specific or a general
   weakness of LLM coding agents using bash for file edits.

## Reproducing

```bash
pip install inspect-ai
export OPENAI_API_KEY="sk-..."

# Run each round
inspect eval tasks/baseline/task.py --model openai/gpt-4o --log-dir ./logs
inspect eval tasks/hypothesis_test/task.py --model openai/gpt-4o --log-dir ./logs
inspect eval tasks/regex_depth/task.py --model openai/gpt-4o --log-dir ./logs

# View results
inspect view --log-dir ./logs
```

Docker must be running. Each round takes ~15-30 seconds.
