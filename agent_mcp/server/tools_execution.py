"""
Eval Execution Tools

Tools for running inspect-ai evaluations, including full runs, subset re-runs,
and variant runs with modified configurations.
"""

import asyncio
import json
import os
import shlex
import zipfile
from pathlib import Path

from mcp.types import Tool, TextContent

DEFAULT_LOG_DIR = os.environ.get("INSPECT_LOG_DIR", "./logs")


def get_tools() -> list[Tool]:
    """Return execution tool definitions."""
    return [
        Tool(
            name="run_eval",
            description=(
                "Run an inspect-ai evaluation. Specify a task (Python module path or file), "
                "model, and optional configuration. Returns the log path of the completed run. "
                "This runs the eval as a subprocess and waits for completion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Task to evaluate. Can be a Python module path (e.g. 'agent_mcp.tasks.cybersec_agent') "
                            "or a file path (e.g. './agent_mcp/tasks/cybersec_agent.py')"
                        ),
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to evaluate (e.g. 'anthropic/claude-sonnet-4-20250514', 'openai/gpt-4o')",
                    },
                    "task_args": {
                        "type": "object",
                        "description": "Arguments to pass to the task function (as key-value pairs)",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Override the default system prompt for the agent being evaluated",
                    },
                    "max_messages": {
                        "type": "integer",
                        "description": "Maximum messages in agent conversation (default: task-specific)",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for model generation",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature for the model",
                    },
                    "sandbox": {
                        "type": "string",
                        "description": "Sandbox type: 'docker', 'local', or 'none'. Default depends on task.",
                    },
                    "sample_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Run only specific sample IDs (for targeted re-evaluation)",
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "Number of epochs to run (default 1)",
                        "default": 1,
                    },
                    "log_dir": {
                        "type": "string",
                        "description": "Directory to store logs. Defaults to INSPECT_LOG_DIR.",
                    },
                    "extra_args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional CLI arguments to pass to inspect eval",
                    },
                },
                "required": ["task", "model"],
            },
        ),
        Tool(
            name="run_eval_with_modified_prompt",
            description=(
                "Re-run an evaluation from a previous log but with a modified system prompt. "
                "This is the key tool for adaptive iteration: analyze failures, hypothesize "
                "a better prompt/configuration, and re-run to test the hypothesis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_log_path": {
                        "type": "string",
                        "description": "Path to the previous eval log to base this run on",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "New system prompt to use for the agent",
                    },
                    "sample_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only re-run specific samples (e.g. failures). If omitted, re-runs all.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optionally switch the model for this re-run",
                    },
                    "extra_task_args": {
                        "type": "object",
                        "description": "Additional task arguments to override",
                    },
                },
                "required": ["source_log_path"],
            },
        ),
        Tool(
            name="run_eval_subset",
            description=(
                "Re-run an evaluation on a subset of samples from a previous run. "
                "Useful for re-running only the failures, or re-running specific interesting cases. "
                "Preserves the original task configuration."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_log_path": {
                        "type": "string",
                        "description": "Path to the previous eval log",
                    },
                    "filter": {
                        "type": "string",
                        "enum": ["incorrect", "correct", "all"],
                        "description": "Which samples to re-run. Default 'incorrect'.",
                        "default": "incorrect",
                    },
                    "sample_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific sample IDs to re-run (overrides filter)",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optionally use a different model",
                    },
                },
                "required": ["source_log_path"],
            },
        ),
        Tool(
            name="get_eval_status",
            description=(
                "Check the status of a running evaluation. Returns progress information "
                "if the eval is still running, or the final results if complete."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "The run ID returned by run_eval",
                    },
                },
                "required": ["run_id"],
            },
        ),
    ]


async def handle_call(name: str, arguments: dict) -> list[TextContent] | None:
    """Handle an execution tool call. Returns None if tool name not recognized."""
    if name == "run_eval":
        return await _handle_run_eval(arguments)
    elif name == "run_eval_with_modified_prompt":
        return await _handle_run_eval_modified(arguments)
    elif name == "run_eval_subset":
        return await _handle_run_eval_subset(arguments)
    elif name == "get_eval_status":
        return await _handle_get_eval_status(arguments)
    return None


async def _handle_run_eval(args: dict) -> list[TextContent]:
    task = args["task"]
    model = args["model"]
    log_dir = args.get("log_dir", DEFAULT_LOG_DIR)

    cmd_parts = ["inspect", "eval", task, "--model", model, "--log-dir", log_dir]

    if args.get("epochs"):
        cmd_parts.extend(["--epochs", str(args["epochs"])])
    if args.get("max_messages"):
        cmd_parts.extend(["--max-messages", str(args["max_messages"])])
    if args.get("max_tokens"):
        cmd_parts.extend(["--max-tokens", str(args["max_tokens"])])
    if args.get("temperature") is not None:
        cmd_parts.extend(["--temperature", str(args["temperature"])])
    if args.get("sandbox"):
        cmd_parts.extend(["--sandbox", args["sandbox"]])

    # Task args passed as -T key=value
    for k, v in args.get("task_args", {}).items():
        cmd_parts.extend(["-T", f"{k}={v}"])

    # System prompt override via task arg
    if args.get("system_prompt"):
        cmd_parts.extend(["-T", f"system_prompt={args['system_prompt']}"])

    # Sample IDs filter
    if args.get("sample_ids"):
        ids = ",".join(args["sample_ids"])
        cmd_parts.extend(["--sample-id", ids])

    # Extra args
    for a in args.get("extra_args", []):
        cmd_parts.append(a)

    cmd = " ".join(shlex.quote(p) for p in cmd_parts)

    result = await _run_subprocess(cmd)
    return [TextContent(type="text", text=result)]


async def _handle_run_eval_modified(args: dict) -> list[TextContent]:
    source_log = Path(args["source_log_path"])
    if not source_log.exists():
        return [TextContent(type="text", text=f"Source log not found: {source_log}")]

    try:
        data = _read_eval_header(source_log)
    except Exception as e:
        return [TextContent(type="text", text=f"Error reading source log: {e}")]

    eval_info = data.get("eval", {})
    task = eval_info.get("task", "")
    model = args.get("model", eval_info.get("model", ""))
    task_args = eval_info.get("task_args", {})

    # Merge extra task args
    task_args.update(args.get("extra_task_args", {}))

    # Override system prompt
    if args.get("system_prompt"):
        task_args["system_prompt"] = args["system_prompt"]

    run_args = {
        "task": task,
        "model": model,
        "task_args": task_args,
    }

    if args.get("sample_ids"):
        run_args["sample_ids"] = args["sample_ids"]

    return await _handle_run_eval(run_args)


async def _handle_run_eval_subset(args: dict) -> list[TextContent]:
    source_log = Path(args["source_log_path"])
    if not source_log.exists():
        return [TextContent(type="text", text=f"Source log not found: {source_log}")]

    try:
        data = _read_eval_full(source_log)
    except Exception as e:
        return [TextContent(type="text", text=f"Error reading source log: {e}")]

    eval_info = data.get("eval", {})
    samples = data.get("samples", [])
    filter_type = args.get("filter", "incorrect")
    sample_ids = args.get("sample_ids")

    if not sample_ids:
        if filter_type == "incorrect":
            sample_ids = [s.get("id", str(i)) for i, s in enumerate(samples) if not _sample_is_correct(s)]
        elif filter_type == "correct":
            sample_ids = [s.get("id", str(i)) for i, s in enumerate(samples) if _sample_is_correct(s)]
        else:
            sample_ids = [s.get("id", str(i)) for i, s in enumerate(samples)]

    run_args = {
        "task": eval_info.get("task", ""),
        "model": args.get("model", eval_info.get("model", "")),
        "task_args": eval_info.get("task_args", {}),
        "sample_ids": sample_ids,
    }

    return await _handle_run_eval(run_args)


async def _handle_get_eval_status(args: dict) -> list[TextContent]:
    run_id = args["run_id"]
    log_dir = Path(DEFAULT_LOG_DIR)
    for path in log_dir.rglob("*.eval"):
        if run_id in path.name:
            try:
                data = _read_eval_header(path)
                status = data.get("status", "unknown")
                results = data.get("results", {})
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": status, "results": results, "log_path": str(path)}, indent=2, default=str),
                )]
            except Exception:
                pass

    return [TextContent(type="text", text=f"No log found for run_id: {run_id}")]


def _read_eval_header(path: Path) -> dict:
    """Read header.json from an .eval zip archive."""
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("header.json"))


def _read_eval_full(path: Path) -> dict:
    """Read full eval log (header + samples) from an .eval zip archive."""
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("header.json"))
        samples = []
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                samples.append(json.loads(z.read(name)))
        data["samples"] = samples
        return data


async def _run_subprocess(cmd: str) -> str:
    """Run a shell command and return combined stdout+stderr."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    output_parts = []
    if stdout:
        output_parts.append(f"STDOUT:\n{stdout.decode(errors='replace')}")
    if stderr:
        output_parts.append(f"STDERR:\n{stderr.decode(errors='replace')}")

    output_parts.append(f"\nReturn code: {proc.returncode}")
    return "\n".join(output_parts)


def _sample_is_correct(sample: dict) -> bool:
    scores = sample.get("scores", {})
    for scorer_name, score_data in scores.items():
        if isinstance(score_data, dict):
            val = score_data.get("value", "")
            if val in ("C", "correct", True, 1, "1", "CORRECT"):
                return True
    return False
