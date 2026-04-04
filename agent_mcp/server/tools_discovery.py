"""
Eval Discovery & Log Reading Tools

Tools for listing, reading, and comparing inspect-ai evaluation logs.
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Any

from mcp.types import Tool, TextContent

# Default log directory for inspect-ai
DEFAULT_LOG_DIR = os.environ.get("INSPECT_LOG_DIR", "./logs")


def _read_eval_zip(path: Path, entry: str) -> dict | None:
    """Read a JSON entry from an .eval zip archive."""
    try:
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read(entry))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, OSError):
        return None


def _read_eval_samples(path: Path) -> list[dict]:
    """Read all sample files from an .eval zip archive."""
    samples = []
    try:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.startswith("samples/") and name.endswith(".json"):
                    samples.append(json.loads(z.read(name)))
    except (zipfile.BadZipFile, OSError):
        pass
    return samples


def _parse_eval_log_header(path: Path) -> dict[str, Any] | None:
    """Parse the header/metadata from an inspect-ai eval log."""
    data = _read_eval_zip(path, "header.json")
    if data is None:
        return None
    header = {}
    if "eval" in data:
        e = data["eval"]
        header["task"] = e.get("task", "")
        header["model"] = e.get("model", "")
        header["task_id"] = e.get("task_id", "")
        header["run_id"] = e.get("run_id", "")
        header["created"] = e.get("created", "")
        header["dataset"] = e.get("dataset", {})
        header["task_args"] = e.get("task_args", {})
    if "results" in data:
        r = data["results"]
        header["scores"] = r.get("scores", [])
        header["completed"] = r.get("completed", None)
        header["total_samples"] = r.get("total_samples", None)
    if "stats" in data:
        header["stats"] = data["stats"]
    header["log_path"] = str(path)
    return header


def _parse_eval_log_full(path: Path) -> dict[str, Any] | None:
    """Parse a full inspect-ai eval log including samples."""
    data = _read_eval_zip(path, "header.json")
    if data is None:
        return None
    data["samples"] = _read_eval_samples(path)
    return data


def _sample_is_correct(sample: dict) -> bool:
    """Check if a sample was scored as correct."""
    scores = sample.get("scores", {})
    for scorer_name, score_data in scores.items():
        if isinstance(score_data, dict):
            val = score_data.get("value", "")
            if val in ("C", "correct", True, 1, "1", "CORRECT"):
                return True
    return False


def get_tools() -> list[Tool]:
    """Return discovery tool definitions."""
    return [
        Tool(
            name="list_eval_logs",
            description=(
                "List previous inspect-ai evaluation runs. Returns metadata for each run "
                "including task name, model, scores, and timestamps. Use this to discover "
                "what evaluations have been run and find specific runs to examine."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "log_dir": {
                        "type": "string",
                        "description": "Directory to search for logs. Defaults to INSPECT_LOG_DIR or ./logs",
                    },
                    "task_filter": {
                        "type": "string",
                        "description": "Filter logs by task name (substring match)",
                    },
                    "model_filter": {
                        "type": "string",
                        "description": "Filter logs by model name (substring match)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of logs to return (default 20)",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="read_eval_log",
            description=(
                "Read a specific inspect-ai evaluation log in detail. Returns full metadata, "
                "scores, and optionally the sample-level results. Use after list_eval_logs "
                "to drill into a specific run."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "log_path": {
                        "type": "string",
                        "description": "Path to the eval log file",
                    },
                    "include_samples": {
                        "type": "boolean",
                        "description": "Include individual sample results (can be large). Default false.",
                        "default": False,
                    },
                    "sample_limit": {
                        "type": "integer",
                        "description": "Max samples to include when include_samples=true. Default 10.",
                        "default": 10,
                    },
                    "sample_filter": {
                        "type": "string",
                        "description": "Filter samples: 'all', 'correct', 'incorrect'. Default 'all'.",
                        "default": "all",
                    },
                },
                "required": ["log_path"],
            },
        ),
        Tool(
            name="read_eval_samples",
            description=(
                "Read individual sample results from an eval log. Shows the input, agent "
                "transcript (messages/tool calls), output, and score for each sample. "
                "Essential for understanding what the agent actually did during evaluation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "log_path": {
                        "type": "string",
                        "description": "Path to the eval log file",
                    },
                    "sample_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific sample indices to read. If omitted, returns first N samples.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max samples to return (default 5)",
                        "default": 5,
                    },
                    "include_transcript": {
                        "type": "boolean",
                        "description": "Include full message transcript for each sample. Default true.",
                        "default": True,
                    },
                },
                "required": ["log_path"],
            },
        ),
        Tool(
            name="compare_eval_runs",
            description=(
                "Compare metrics across multiple evaluation runs. Useful for seeing how "
                "different configurations (prompts, models, tools) affect performance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "log_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to eval log files to compare",
                    },
                },
                "required": ["log_paths"],
            },
        ),
    ]


async def handle_call(name: str, arguments: dict) -> list[TextContent] | None:
    """Handle a discovery tool call. Returns None if tool name not recognized."""
    if name == "list_eval_logs":
        return await _handle_list_eval_logs(arguments)
    elif name == "read_eval_log":
        return await _handle_read_eval_log(arguments)
    elif name == "read_eval_samples":
        return await _handle_read_eval_samples(arguments)
    elif name == "compare_eval_runs":
        return await _handle_compare_eval_runs(arguments)
    return None


async def _handle_list_eval_logs(args: dict) -> list[TextContent]:
    log_dir = Path(args.get("log_dir", DEFAULT_LOG_DIR))
    task_filter = args.get("task_filter", "")
    model_filter = args.get("model_filter", "")
    limit = args.get("limit", 20)

    if not log_dir.exists():
        return [TextContent(type="text", text=f"Log directory not found: {log_dir}")]

    log_files = sorted(log_dir.rglob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)

    results = []
    for path in log_files:
        if len(results) >= limit:
            break
        header = _parse_eval_log_header(path)
        if header is None:
            continue
        if task_filter and task_filter.lower() not in header.get("task", "").lower():
            continue
        if model_filter and model_filter.lower() not in header.get("model", "").lower():
            continue
        results.append(header)

    return [TextContent(
        type="text",
        text=json.dumps(results, indent=2, default=str),
    )]


async def _handle_read_eval_log(args: dict) -> list[TextContent]:
    log_path = Path(args["log_path"])
    include_samples = args.get("include_samples", False)
    sample_limit = args.get("sample_limit", 10)
    sample_filter = args.get("sample_filter", "all")

    data = _parse_eval_log_full(log_path)
    if data is None:
        return [TextContent(type="text", text=f"Could not read log: {log_path}")]

    result = {
        "eval": data.get("eval", {}),
        "plan": data.get("plan", {}),
        "results": data.get("results", {}),
        "stats": data.get("stats", {}),
    }

    if include_samples and "samples" in data:
        samples = data["samples"]
        if sample_filter == "correct":
            samples = [s for s in samples if _sample_is_correct(s)]
        elif sample_filter == "incorrect":
            samples = [s for s in samples if not _sample_is_correct(s)]
        result["samples"] = samples[:sample_limit]
        result["total_samples_available"] = len(data["samples"])

    return [TextContent(
        type="text",
        text=json.dumps(result, indent=2, default=str),
    )]


async def _handle_read_eval_samples(args: dict) -> list[TextContent]:
    log_path = Path(args["log_path"])
    sample_ids = args.get("sample_ids")
    limit = args.get("limit", 5)
    include_transcript = args.get("include_transcript", True)

    data = _parse_eval_log_full(log_path)
    if data is None:
        return [TextContent(type="text", text=f"Could not read log: {log_path}")]

    samples = data.get("samples", [])
    if sample_ids:
        selected = [samples[i] for i in sample_ids if i < len(samples)]
    else:
        selected = samples[:limit]

    results = []
    for s in selected:
        entry = {
            "id": s.get("id", ""),
            "input": s.get("input", ""),
            "target": s.get("target", ""),
            "scores": s.get("scores", {}),
            "metadata": s.get("metadata", {}),
        }
        if include_transcript:
            entry["messages"] = s.get("messages", [])
        else:
            msg_count = len(s.get("messages", []))
            entry["message_count"] = msg_count
        results.append(entry)

    return [TextContent(
        type="text",
        text=json.dumps(results, indent=2, default=str),
    )]


async def _handle_compare_eval_runs(args: dict) -> list[TextContent]:
    log_paths = args["log_paths"]
    comparison = []
    for p in log_paths:
        header = _parse_eval_log_header(Path(p))
        if header:
            comparison.append(header)

    return [TextContent(
        type="text",
        text=json.dumps(comparison, indent=2, default=str),
    )]
