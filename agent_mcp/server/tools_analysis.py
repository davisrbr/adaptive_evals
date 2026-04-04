"""
Analysis Tools

Tools for analyzing eval results, scanning transcripts with Inspect Scout,
and summarizing findings. These tools help the agent understand what happened
during an eval and identify patterns in failures.
"""

import json
import os
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcp.types import Tool, TextContent


def get_tools() -> list[Tool]:
    """Return analysis tool definitions."""
    return [
        Tool(
            name="analyze_failures",
            description=(
                "Analyze failure patterns in an evaluation run. Categorizes failures by type, "
                "extracts common patterns in agent behavior, and identifies the most informative "
                "failure cases. Returns a structured analysis that helps decide what to change."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "log_path": {
                        "type": "string",
                        "description": "Path to the eval log file to analyze",
                    },
                    "max_failures": {
                        "type": "integer",
                        "description": "Maximum number of failure cases to analyze in detail (default 10)",
                        "default": 10,
                    },
                },
                "required": ["log_path"],
            },
        ),
        Tool(
            name="summarize_eval",
            description=(
                "Generate a high-level summary of an evaluation run. Includes overall scores, "
                "sample counts, model info, and key statistics. Good first tool to call after "
                "running an eval."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "log_path": {
                        "type": "string",
                        "description": "Path to the eval log file",
                    },
                },
                "required": ["log_path"],
            },
        ),
        Tool(
            name="scan_transcripts",
            description=(
                "Run Inspect Scout scanners on eval transcripts. Scout can analyze transcripts "
                "for ANY pattern using LLM-based or regex-based scanning. Use built-in scanners "
                "(failure_mode, abstention_judgment, tool_usage_pattern, reasoning_quality, "
                "behavioral_tags, eval_awareness) or provide a custom question to scan for. "
                "Returns structured classification results across all transcripts. "
                "This is the primary tool for discovering eval patterns in the adaptive loop."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "log_path": {
                        "type": "string",
                        "description": "Path to the eval log file or log directory to scan",
                    },
                    "scanners": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of scanners to run. Built-in scanners: "
                            "'failure_mode' (classifies why a sample failed — wrong tool, missing call, etc), "
                            "'abstention_judgment' (evaluates tool use/abstain decisions), "
                            "'tool_usage_pattern' (characterizes tool strategy — strategic, reactive, confused), "
                            "'reasoning_quality' (0-10 numeric rating of reasoning), "
                            "'behavioral_tags' (multi-label tags: hallucination, domain_confusion, etc), "
                            "'eval_awareness' (detects gaming/eval awareness). "
                            "Default: runs all built-in scanners."
                        ),
                    },
                    "custom_questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Name for this custom scanner",
                                },
                                "question": {
                                    "type": "string",
                                    "description": "The question to ask about each transcript",
                                },
                                "answer_type": {
                                    "type": "string",
                                    "enum": ["boolean", "numeric", "labels"],
                                    "description": "Answer format: boolean (yes/no), numeric (0-10), or labels (provide labels list)",
                                    "default": "boolean",
                                },
                                "labels": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "For answer_type='labels': list of classification labels",
                                },
                            },
                            "required": ["name", "question"],
                        },
                        "description": (
                            "Custom LLM-based scanners. Each specifies a question to ask about "
                            "every transcript, with configurable answer types. Use this to scan "
                            "for any behavior pattern specific to your evaluation."
                        ),
                    },
                    "grep_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Regex patterns to grep for across transcripts (no LLM needed). "
                            "Useful for finding credential leaks, error messages, specific tool names, etc."
                        ),
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for LLM-based scanning (default: uses Scout default)",
                    },
                },
                "required": ["log_path"],
            },
        ),
        Tool(
            name="extract_agent_patterns",
            description=(
                "Extract patterns from agent transcripts across samples. Identifies common "
                "tool usage sequences, recurring errors, and behavioral patterns. Useful for "
                "understanding how the agent approaches tasks systematically."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "log_path": {
                        "type": "string",
                        "description": "Path to the eval log file",
                    },
                    "pattern_type": {
                        "type": "string",
                        "enum": ["tool_usage", "error_patterns", "message_flow", "all"],
                        "description": "Type of patterns to extract. Default 'all'.",
                        "default": "all",
                    },
                },
                "required": ["log_path"],
            },
        ),
        Tool(
            name="diff_eval_runs",
            description=(
                "Compare two evaluation runs in detail. Shows which samples improved, "
                "regressed, or stayed the same between runs. Essential for measuring the "
                "impact of prompt/config changes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "log_path_a": {
                        "type": "string",
                        "description": "Path to the first (baseline) eval log",
                    },
                    "log_path_b": {
                        "type": "string",
                        "description": "Path to the second (experimental) eval log",
                    },
                },
                "required": ["log_path_a", "log_path_b"],
            },
        ),
    ]


async def handle_call(name: str, arguments: dict) -> list[TextContent] | None:
    """Handle an analysis tool call. Returns None if tool name not recognized."""
    if name == "analyze_failures":
        return await _handle_analyze_failures(arguments)
    elif name == "summarize_eval":
        return await _handle_summarize_eval(arguments)
    elif name == "scan_transcripts":
        return await _handle_scan_transcripts(arguments)
    elif name == "extract_agent_patterns":
        return await _handle_extract_patterns(arguments)
    elif name == "diff_eval_runs":
        return await _handle_diff_eval_runs(arguments)
    return None


async def _handle_analyze_failures(args: dict) -> list[TextContent]:
    log_path = Path(args["log_path"])
    max_failures = args.get("max_failures", 10)

    data = _load_log(log_path)
    if isinstance(data, str):
        return [TextContent(type="text", text=data)]

    samples = data.get("samples", [])
    failures = [s for s in samples if not _sample_is_correct(s)]

    analysis = {
        "total_samples": len(samples),
        "total_failures": len(failures),
        "failure_rate": f"{len(failures)/max(len(samples),1)*100:.1f}%",
        "failure_details": [],
    }

    failure_categories: dict[str, list] = defaultdict(list)
    for s in failures[:max_failures]:
        detail = _extract_failure_detail(s)
        analysis["failure_details"].append(detail)

        category = detail.get("likely_cause", "unknown")
        failure_categories[category].append(detail["sample_id"])

    analysis["failure_categories"] = {k: {"count": len(v), "sample_ids": v} for k, v in failure_categories.items()}

    return [TextContent(type="text", text=json.dumps(analysis, indent=2, default=str))]


async def _handle_summarize_eval(args: dict) -> list[TextContent]:
    log_path = Path(args["log_path"])
    data = _load_log(log_path)
    if isinstance(data, str):
        return [TextContent(type="text", text=data)]

    eval_info = data.get("eval", {})
    results = data.get("results", {})
    stats = data.get("stats", {})
    samples = data.get("samples", [])

    correct = sum(1 for s in samples if _sample_is_correct(s))
    total = len(samples)

    summary = {
        "task": eval_info.get("task", ""),
        "model": eval_info.get("model", ""),
        "created": eval_info.get("created", ""),
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": f"{correct/max(total,1)*100:.1f}%",
        "scores": results.get("scores", []),
        "task_args": eval_info.get("task_args", {}),
        "stats": stats,
        "log_path": str(log_path),
    }

    return [TextContent(type="text", text=json.dumps(summary, indent=2, default=str))]


async def _handle_scan_transcripts(args: dict) -> list[TextContent]:
    log_path = args["log_path"]
    model = args.get("model", "")

    # Check scout is available
    try:
        result = subprocess.run(
            ["scout", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        scout_available = result.returncode == 0
    except Exception:
        scout_available = False

    if not scout_available:
        return [TextContent(
            type="text",
            text="Inspect Scout is not installed. Install with: pip install inspect-scout",
        )]

    # Build the scanner file
    scanner_file = _build_scanner_file(args)

    cmd = ["scout", "scan", scanner_file, "-T", log_path]
    if model:
        cmd.extend(["--model", model])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env={**os.environ},
        )
        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            output = f"Scout exited with code {result.returncode}:\n{output}"
    except subprocess.TimeoutExpired:
        output = "Scout scan timed out after 10 minutes"

    # Read structured results from the most recent scan
    results_summary = _read_latest_scan_results()
    if results_summary:
        output += f"\n\n{results_summary}"

    return [TextContent(type="text", text=output.strip())]


def _build_scanner_file(args: dict) -> str:
    """Build a scanner file from the request arguments.

    Supports three modes:
    1. Built-in named scanners from agent_mcp/scanners/
    2. Custom LLM-based questions (ad-hoc scanners)
    3. Grep patterns (no LLM needed)

    When multiple scanners/questions are requested, they are all combined
    into a single scanner file so Scout runs them in one pass.
    """
    import tempfile

    scanner_names = args.get("scanners", None)
    custom_questions = args.get("custom_questions", None)
    grep_patterns = args.get("grep_patterns", None)

    # If no specific scanners requested and no custom questions, use the
    # full adaptive scanner suite
    if not scanner_names and not custom_questions and not grep_patterns:
        scanners_dir = Path(__file__).parent.parent / "scanners"
        adaptive_path = scanners_dir / "adaptive_scanners.py"
        if adaptive_path.exists():
            return str(adaptive_path)
        # Fallback to eval_awareness only
        scanner_names = ["eval_awareness"]

    # If only built-in scanner names, check if they're all in one file
    if scanner_names and not custom_questions and not grep_patterns:
        scanners_dir = Path(__file__).parent.parent / "scanners"
        # adaptive_scanners.py contains all built-in scanners
        adaptive_path = scanners_dir / "adaptive_scanners.py"
        if adaptive_path.exists():
            return str(adaptive_path)
        # Try individual scanner files
        for name in scanner_names:
            path = scanners_dir / f"{name}.py"
            if path.exists():
                return str(path)

    # Generate a dynamic scanner file combining everything requested
    lines = [
        "from inspect_scout import llm_scanner, grep_scanner, scanner, AnswerMultiLabel",
        "",
    ]

    # Add built-in scanners by importing from our scanner modules
    if scanner_names:
        for name in scanner_names:
            scanners_dir = Path(__file__).parent.parent / "scanners"
            if (scanners_dir / "adaptive_scanners.py").exists():
                lines.append(
                    f"from agent_mcp.scanners.adaptive_scanners import {name}"
                )

    # Add custom LLM-based scanners
    if custom_questions:
        for i, q in enumerate(custom_questions):
            name = q.get("name", f"custom_{i}")
            question = q["question"].replace('"', '\\"').replace("\n", "\\n")
            answer_type = q.get("answer_type", "boolean")

            if answer_type == "labels" and q.get("labels"):
                labels_str = json.dumps(q["labels"])
                lines.append(f"""
@scanner(messages="all")
def {name}():
    return llm_scanner(question="{question}", answer={labels_str})
""")
            elif answer_type == "numeric":
                lines.append(f"""
@scanner(messages="all")
def {name}():
    return llm_scanner(question="{question}", answer="numeric")
""")
            else:
                lines.append(f"""
@scanner(messages="all")
def {name}():
    return llm_scanner(question="{question}", answer="boolean")
""")

    # Add grep scanners
    if grep_patterns:
        patterns_str = json.dumps(grep_patterns)
        lines.append(f"""
@scanner(messages="all")
def grep_patterns():
    return grep_scanner(patterns={patterns_str})
""")

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    f.write("\n".join(lines))
    f.close()
    return f.name


def _read_latest_scan_results() -> str:
    """Read and format results from the most recent Scout scan."""
    scans_dir = Path("scans")
    if not scans_dir.exists():
        return ""

    scan_dirs = sorted(
        [d for d in scans_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not scan_dirs:
        return ""

    latest = scan_dirs[0]
    summary_path = latest / "_summary.json"
    if not summary_path.exists():
        return ""

    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return ""

    parts = [f"Scan results ({latest.name}):"]

    for scanner_name, data in summary.get("scanners", {}).items():
        scans = data.get("scans", 0)
        results = data.get("results", 0)
        errors = data.get("errors", 0)
        parts.append(f"\n  {scanner_name}: {scans} transcripts scanned, {results} results, {errors} errors")

    # Try to load per-scanner dataframes for detailed results
    try:
        from inspect_scout import scan_results_df
        df = scan_results_df(str(latest))
        for scanner_name, scanner_df in df.scanners.items():
            if len(scanner_df) == 0:
                continue
            parts.append(f"\n  {scanner_name} distribution:")
            if "answer" in scanner_df.columns:
                counts = scanner_df["answer"].value_counts()
                for val, cnt in counts.items():
                    parts.append(f"    {val}: {cnt}/{len(scanner_df)}")
            elif "value" in scanner_df.columns:
                vals = scanner_df["value"].dropna()
                if len(vals) > 0:
                    parts.append(f"    mean={vals.mean():.2f}, min={vals.min()}, max={vals.max()}")
    except Exception:
        pass

    return "\n".join(parts)


async def _handle_extract_patterns(args: dict) -> list[TextContent]:
    log_path = Path(args["log_path"])
    pattern_type = args.get("pattern_type", "all")

    data = _load_log(log_path)
    if isinstance(data, str):
        return [TextContent(type="text", text=data)]

    samples = data.get("samples", [])
    patterns: dict[str, Any] = {}

    if pattern_type in ("tool_usage", "all"):
        tool_counts: Counter = Counter()
        tool_sequences: list[list[str]] = []
        for s in samples:
            seq = []
            for msg in s.get("messages", []):
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls", []):
                        tool_name = tc.get("function", {}).get("name", tc.get("name", "unknown"))
                        tool_counts[tool_name] += 1
                        seq.append(tool_name)
            if seq:
                tool_sequences.append(seq)
        patterns["tool_usage"] = {
            "tool_frequency": dict(tool_counts.most_common(20)),
            "avg_tools_per_sample": sum(len(s) for s in tool_sequences) / max(len(tool_sequences), 1),
            "example_sequences": tool_sequences[:5],
        }

    if pattern_type in ("error_patterns", "all"):
        error_counts: Counter = Counter()
        for s in samples:
            for msg in s.get("messages", []):
                if msg.get("role") == "tool" and "error" in str(msg.get("content", "")).lower():
                    content = str(msg.get("content", ""))
                    error_type = content[:100]
                    error_counts[error_type] += 1
        patterns["error_patterns"] = {
            "error_frequency": dict(error_counts.most_common(10)),
            "total_errors": sum(error_counts.values()),
        }

    if pattern_type in ("message_flow", "all"):
        msg_lengths = []
        for s in samples:
            msg_lengths.append(len(s.get("messages", [])))
        patterns["message_flow"] = {
            "avg_messages_per_sample": sum(msg_lengths) / max(len(msg_lengths), 1),
            "min_messages": min(msg_lengths) if msg_lengths else 0,
            "max_messages": max(msg_lengths) if msg_lengths else 0,
        }

    return [TextContent(type="text", text=json.dumps(patterns, indent=2, default=str))]


async def _handle_diff_eval_runs(args: dict) -> list[TextContent]:
    data_a = _load_log(Path(args["log_path_a"]))
    data_b = _load_log(Path(args["log_path_b"]))
    if isinstance(data_a, str):
        return [TextContent(type="text", text=data_a)]
    if isinstance(data_b, str):
        return [TextContent(type="text", text=data_b)]

    samples_a = {s.get("id", str(i)): s for i, s in enumerate(data_a.get("samples", []))}
    samples_b = {s.get("id", str(i)): s for i, s in enumerate(data_b.get("samples", []))}

    common_ids = set(samples_a.keys()) & set(samples_b.keys())
    improved = []
    regressed = []
    unchanged_correct = []
    unchanged_incorrect = []

    for sid in sorted(common_ids):
        a_correct = _sample_is_correct(samples_a[sid])
        b_correct = _sample_is_correct(samples_b[sid])
        if not a_correct and b_correct:
            improved.append(sid)
        elif a_correct and not b_correct:
            regressed.append(sid)
        elif a_correct and b_correct:
            unchanged_correct.append(sid)
        else:
            unchanged_incorrect.append(sid)

    total_a_correct = sum(1 for s in data_a.get("samples", []) if _sample_is_correct(s))
    total_b_correct = sum(1 for s in data_b.get("samples", []) if _sample_is_correct(s))

    diff = {
        "run_a": {
            "log_path": args["log_path_a"],
            "model": data_a.get("eval", {}).get("model", ""),
            "accuracy": f"{total_a_correct}/{len(data_a.get('samples', []))}",
        },
        "run_b": {
            "log_path": args["log_path_b"],
            "model": data_b.get("eval", {}).get("model", ""),
            "accuracy": f"{total_b_correct}/{len(data_b.get('samples', []))}",
        },
        "comparison": {
            "common_samples": len(common_ids),
            "improved": {"count": len(improved), "sample_ids": improved},
            "regressed": {"count": len(regressed), "sample_ids": regressed},
            "unchanged_correct": {"count": len(unchanged_correct)},
            "unchanged_incorrect": {"count": len(unchanged_incorrect), "sample_ids": unchanged_incorrect[:20]},
        },
        "only_in_a": list(set(samples_a.keys()) - common_ids)[:20],
        "only_in_b": list(set(samples_b.keys()) - common_ids)[:20],
    }

    return [TextContent(type="text", text=json.dumps(diff, indent=2, default=str))]


def _load_log(path: Path) -> dict | str:
    """Load an inspect-ai .eval log (zip archive with header.json and samples/)."""
    if not path.exists():
        return f"Log file not found: {path}"
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("header.json"))
            samples = []
            for name in z.namelist():
                if name.startswith("samples/") and name.endswith(".json"):
                    samples.append(json.loads(z.read(name)))
            data["samples"] = samples
            return data
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, OSError) as e:
        return f"Error reading log: {e}"


def _sample_is_correct(sample: dict) -> bool:
    scores = sample.get("scores", {})
    for scorer_name, score_data in scores.items():
        if isinstance(score_data, dict):
            val = score_data.get("value", "")
            if val in ("C", "correct", True, 1, "1", "CORRECT"):
                return True
    return False


def _extract_failure_detail(sample: dict) -> dict:
    """Extract relevant details from a failed sample for analysis."""
    messages = sample.get("messages", [])
    detail = {
        "sample_id": sample.get("id", ""),
        "input": _truncate(str(sample.get("input", "")), 500),
        "target": sample.get("target", ""),
        "scores": sample.get("scores", {}),
        "message_count": len(messages),
    }

    last_assistant = ""
    tool_errors = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_assistant = content
        if msg.get("role") == "tool" and "error" in str(msg.get("content", "")).lower():
            tool_errors.append(_truncate(str(msg.get("content", "")), 200))

    if tool_errors:
        detail["likely_cause"] = "tool_error"
        detail["tool_errors"] = tool_errors[:3]
    elif not messages:
        detail["likely_cause"] = "no_response"
    elif len(messages) <= 2:
        detail["likely_cause"] = "insufficient_reasoning"
    else:
        detail["likely_cause"] = "incorrect_answer"

    detail["last_assistant_message"] = _truncate(last_assistant, 300)
    return detail


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s
