"""
Configuration & Task Management Tools

Tools for listing available tasks, creating task variants, and configuring
agent parameters for evaluations.
"""

import json
from pathlib import Path

from mcp.types import Tool, TextContent

TASKS_DIR = Path(__file__).parent.parent / "tasks"


def get_tools() -> list[Tool]:
    """Return configuration tool definitions."""
    return [
        Tool(
            name="list_tasks",
            description=(
                "List all available inspect-ai evaluation tasks. Shows task name, description, "
                "available parameters, and file location. Includes both built-in tasks from "
                "this project and any tasks in the configured task directories."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_dir": {
                        "type": "string",
                        "description": "Additional directory to search for tasks. Default searches agent_mcp/tasks/.",
                    },
                    "include_original": {
                        "type": "boolean",
                        "description": "Also list tasks from the original adaptive_evals tasks/ directory. Default true.",
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="create_task_variant",
            description=(
                "Create a modified variant of an existing eval task. Generates a new task file "
                "with the specified changes (different system prompt, tools, sandbox config, "
                "dataset, scoring, etc). The variant is saved to agent_mcp/tasks/ and can be "
                "run with run_eval."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "base_task_file": {
                        "type": "string",
                        "description": "Path to the base task file to modify",
                    },
                    "variant_name": {
                        "type": "string",
                        "description": "Name for the new task variant (used as filename)",
                    },
                    "modifications": {
                        "type": "object",
                        "description": (
                            "Modifications to apply. Keys can include: "
                            "'system_prompt' (new system prompt), "
                            "'tools' (list of tool names to add/remove), "
                            "'sandbox' (sandbox configuration), "
                            "'max_messages' (agent message limit), "
                            "'scorer' (scoring function to use), "
                            "'dataset_args' (dataset configuration)"
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what this variant tests",
                    },
                },
                "required": ["base_task_file", "variant_name"],
            },
        ),
        Tool(
            name="write_task",
            description=(
                "Write a complete new inspect-ai task file. Use this to create entirely new "
                "agentic evaluation tasks from scratch. The task should use inspect-ai's agent "
                "framework (ReAct agent, custom tools, sandboxing, etc)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "Name for the task (used as filename, e.g. 'my_custom_task')",
                    },
                    "task_code": {
                        "type": "string",
                        "description": "Complete Python source code for the task",
                    },
                },
                "required": ["task_name", "task_code"],
            },
        ),
        Tool(
            name="read_task_source",
            description=(
                "Read the source code of an existing task file. Useful for understanding "
                "how a task works before creating variants or modifications."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_path": {
                        "type": "string",
                        "description": "Path to the task file to read",
                    },
                },
                "required": ["task_path"],
            },
        ),
        Tool(
            name="get_inspect_docs",
            description=(
                "Get reference documentation for inspect-ai concepts. Helpful for writing "
                "new tasks or understanding the framework."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": [
                            "agents",
                            "tools",
                            "sandboxes",
                            "scorers",
                            "datasets",
                            "task_structure",
                            "mcp_tools",
                            "react_agent",
                        ],
                        "description": "Topic to get documentation for",
                    },
                },
                "required": ["topic"],
            },
        ),
    ]


async def handle_call(name: str, arguments: dict) -> list[TextContent] | None:
    """Handle a configuration tool call. Returns None if tool name not recognized."""
    if name == "list_tasks":
        return await _handle_list_tasks(arguments)
    elif name == "create_task_variant":
        return await _handle_create_variant(arguments)
    elif name == "write_task":
        return await _handle_write_task(arguments)
    elif name == "read_task_source":
        return await _handle_read_task_source(arguments)
    elif name == "get_inspect_docs":
        return await _handle_get_docs(arguments)
    return None


async def _handle_list_tasks(args: dict) -> list[TextContent]:
    tasks = []

    for py_file in TASKS_DIR.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        task_info = _extract_task_info(py_file)
        if task_info:
            tasks.extend(task_info)

    extra_dir = args.get("task_dir")
    if extra_dir:
        for py_file in Path(extra_dir).glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            task_info = _extract_task_info(py_file)
            if task_info:
                tasks.extend(task_info)

    if args.get("include_original", True):
        original_dir = Path(__file__).parent.parent.parent / "tasks"
        if original_dir.exists():
            for py_file in original_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                task_info = _extract_task_info(py_file)
                if task_info:
                    tasks.extend(task_info)

    return [TextContent(type="text", text=json.dumps(tasks, indent=2, default=str))]


async def _handle_create_variant(args: dict) -> list[TextContent]:
    base_path = Path(args["base_task_file"])
    variant_name = args["variant_name"]
    modifications = args.get("modifications", {})
    description = args.get("description", "")

    if not base_path.exists():
        return [TextContent(type="text", text=f"Base task not found: {base_path}")]

    base_code = base_path.read_text()
    variant_code = _generate_variant(base_code, variant_name, modifications, description)

    out_path = TASKS_DIR / f"{variant_name}.py"
    out_path.write_text(variant_code)

    return [TextContent(
        type="text",
        text=f"Created task variant at: {out_path}\nRun with: run_eval(task='{out_path}', model='...')",
    )]


async def _handle_write_task(args: dict) -> list[TextContent]:
    task_name = args["task_name"]
    task_code = args["task_code"]

    out_path = TASKS_DIR / f"{task_name}.py"
    out_path.write_text(task_code)

    return [TextContent(
        type="text",
        text=f"Wrote task to: {out_path}\nRun with: run_eval(task='{out_path}', model='...')",
    )]


async def _handle_read_task_source(args: dict) -> list[TextContent]:
    task_path = Path(args["task_path"])
    if not task_path.exists():
        return [TextContent(type="text", text=f"Task file not found: {task_path}")]
    return [TextContent(type="text", text=task_path.read_text())]


async def _handle_get_docs(args: dict) -> list[TextContent]:
    topic = args["topic"]
    docs = _INSPECT_DOCS.get(topic, f"No documentation available for: {topic}")
    return [TextContent(type="text", text=docs)]


def _extract_task_info(py_file: Path) -> list[dict]:
    """Extract task function info from a Python file by parsing it."""
    tasks = []
    try:
        source = py_file.read_text()
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "@task" in line:
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip().startswith("def "):
                        func_name = lines[j].strip().split("(")[0].replace("def ", "")
                        docstring = ""
                        if j + 1 < len(lines) and '"""' in lines[j + 1]:
                            doc_start = j + 1
                            if lines[doc_start].count('"""') >= 2:
                                docstring = lines[doc_start].strip().strip('"')
                            else:
                                for k in range(doc_start + 1, min(doc_start + 10, len(lines))):
                                    if '"""' in lines[k]:
                                        docstring = "\n".join(
                                            l.strip() for l in lines[doc_start:k + 1]
                                        ).strip('"').strip()
                                        break
                        tasks.append({
                            "name": func_name,
                            "file": str(py_file),
                            "description": docstring,
                        })
                        break
    except OSError:
        pass
    return tasks


def _generate_variant(base_code: str, variant_name: str, modifications: dict, description: str) -> str:
    """Generate a task variant with modifications applied."""
    header = f'"""\nTask variant: {variant_name}\n{description}\n\nAuto-generated variant. Modifications:\n'
    for k, v in modifications.items():
        header += f"  - {k}: {v}\n"
    header += '"""\n\n'

    code = header + base_code

    if "system_prompt" in modifications:
        prompt = modifications["system_prompt"].replace('"', '\\"').replace("\n", "\\n")
        code += f'\n\n# Variant system prompt override\n_VARIANT_SYSTEM_PROMPT = "{prompt}"\n'

    return code


_INSPECT_DOCS = {
    "agents": """
# Inspect AI Agents

Agents are the primary way to run agentic evaluations. They combine a model
with tools and execute multi-turn conversations.

## ReAct Agent (built-in)
```python
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.tool import bash, python

@task
def my_agent_task():
    return Task(
        dataset=my_dataset(),
        agent=react(tools=[bash(), python()]),
        scorer=my_scorer(),
        sandbox="docker",
    )
```

## Custom Agent
```python
from inspect_ai.agent import Agent, AgentState, agent

@agent
def my_agent(custom_param: str = "default"):
    async def execute(state: AgentState) -> AgentState:
        # Custom agent logic
        state = await state.generate()  # Call model
        return state
    return execute
```

## Agent Limits
```python
from inspect_ai import Task
from inspect_ai.agent import react

# Message and token limits are set on the Task, not the agent
task = Task(
    agent=react(tools=[bash(), python()]),
    message_limit=50,
    token_limit=100000,
    time_limit=300,  # seconds
)
```
""",
    "tools": """
# Inspect AI Tools

Tools give agents capabilities to interact with the environment.

## Built-in Tools
```python
from inspect_ai.tool import bash, python, web_search, computer

# Bash tool - execute shell commands
bash(timeout=60)

# Python tool - execute Python code
python(timeout=60)

# Web search
web_search()

# Computer use
computer()
```

## Custom Tools
```python
from inspect_ai.tool import Tool, tool

@tool
def my_tool():
    async def execute(query: str) -> str:
        \"\"\"Description of what the tool does.

        Args:
            query: The query to process
        \"\"\"
        return f"Result for {query}"
    return execute
```

## MCP Tools (from external MCP servers)
```python
from inspect_ai.tool import mcp_server_stdio, mcp_tools

server = mcp_server_stdio("npx", ["-y", "@example/mcp-server"])
tools = mcp_tools(server)
```
""",
    "sandboxes": """
# Inspect AI Sandboxes

Sandboxes provide isolated execution environments for agent tools.

## Docker Sandbox
```python
@task
def my_task():
    return Task(
        dataset=dataset,
        agent=react(tools=[bash()]),
        sandbox="docker",  # Uses default Dockerfile
    )
```

## Docker Compose
```python
@task
def my_task():
    return Task(
        dataset=dataset,
        agent=react(tools=[bash()]),
        sandbox=("docker", "compose.yaml"),
    )
```

## compose.yaml example:
```yaml
services:
  default:
    build: .
    init: true
    command: tail -f /dev/null
```

## Local Sandbox (no isolation)
```python
sandbox="local"
```
""",
    "scorers": """
# Inspect AI Scorers

Scorers evaluate agent outputs and assign scores.

## Built-in Scorers
```python
from inspect_ai.scorer import match, includes, model_graded_fact, model_graded_qa

# Exact match
match()

# Substring match
includes()

# LLM-graded factual accuracy
model_graded_fact()

# LLM-graded QA
model_graded_qa()
```

## Custom Scorer
```python
from inspect_ai.scorer import scorer, Score, Target, CORRECT, INCORRECT

@scorer
def my_scorer():
    async def score(state, target: Target) -> Score:
        answer = state.output.completion
        if target.text.lower() in answer.lower():
            return Score(value=CORRECT, explanation="Found target in output")
        return Score(value=INCORRECT, explanation="Target not found")
    return score
```
""",
    "datasets": """
# Inspect AI Datasets

Datasets provide the evaluation samples.

## Built-in Loaders
```python
from inspect_ai.dataset import json_dataset, csv_dataset, hf_dataset, MemoryDataset, Sample

# From JSON file
dataset = json_dataset("path/to/data.json")

# From CSV
dataset = csv_dataset("path/to/data.csv")

# From HuggingFace
dataset = hf_dataset("dataset_name", split="test")

# In-memory
dataset = MemoryDataset([
    Sample(input="question 1", target="answer 1", id="1"),
    Sample(input="question 2", target="answer 2", id="2"),
])
```

## Sample with files (for sandbox tasks)
```python
Sample(
    input="Find the flag in /tmp/secret.txt",
    target="FLAG{secret}",
    files={"secret.txt": "FLAG{secret}"},
    setup="chmod 600 /tmp/secret.txt",
)
```
""",
    "task_structure": """
# Inspect AI Task Structure

A complete task definition:

```python
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import match
from inspect_ai.tool import bash, python

@task
def my_evaluation(
    system_prompt: str = "You are a helpful assistant.",
    max_messages: int = 30,
):
    dataset = MemoryDataset([
        Sample(
            input="What is 2+2?",
            target="4",
            id="math_1",
        ),
    ])

    return Task(
        dataset=dataset,
        agent=react(
            tools=[bash(), python()],
            prompt=system_prompt,
        ),
        scorer=match(),
        sandbox="docker",
        message_limit=max_messages,
    )
```

Run with: `inspect eval my_file.py --model anthropic/claude-sonnet-4-20250514`
""",
    "mcp_tools": """
# Using MCP Servers as Tools in Inspect AI

Inspect AI can consume MCP server tools and make them available to agents.

## stdio transport
```python
from inspect_ai.tool import mcp_server_stdio, mcp_tools

server = mcp_server_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/path"])
tools = mcp_tools(server)

@task
def my_task():
    return Task(
        dataset=dataset,
        agent=react(tools=tools),
        scorer=scorer,
    )
```

## HTTP transport
```python
from inspect_ai.tool import mcp_server_http

server = mcp_server_http("https://mcp.example.com/sse")
```

## Sandbox transport (tools run inside sandbox)
```python
from inspect_ai.tool import mcp_server_sandbox

server = mcp_server_sandbox("npx", ["-y", "@example/mcp-server"])
```

## Filtering tools from a server
```python
# Only expose specific tools
tools = mcp_tools(server, tools=["read_file", "write_file"])
```
""",
    "react_agent": """
# ReAct Agent Details

The ReAct agent is inspect-ai's built-in general-purpose agent.

```python
from inspect_ai import Task
from inspect_ai.agent import react

agent = react(
    # Tools available to the agent
    tools=[bash(), python(), web_search()],

    # Prompt (prepended to conversation)
    prompt="You are a security researcher...",
)

# Limits are set on the Task
task = Task(
    agent=agent,
    message_limit=50,
    token_limit=100000,
    time_limit=300,
)
```

## Multi-agent with handoff
```python
from inspect_ai.agent import react, handoff

researcher = react(tools=[web_search()], name="researcher")
coder = react(tools=[bash(), python()], name="coder")

orchestrator = react(
    tools=[handoff(researcher), handoff(coder)],
    name="orchestrator",
)
```
""",
}
