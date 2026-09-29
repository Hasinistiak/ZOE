from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from backend.functions.project_creator.project_creator import (
    create_project,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

MODEL = os.getenv(
    "ZOE_PROJECT_CREATOR_MODEL",
    "openai/gpt-oss-20b",
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY3",
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY3 is not set."
    )

client = Groq(
    api_key=GROQ_API_KEY
)


# Maximum LLM reasoning rounds.
MAX_AGENT_STEPS = int(
    os.getenv(
        "ZOE_PROJECT_CREATOR_MAX_STEPS",
        "4",
    )
)


# Maximum project creation executions.
#
# A single project request should only ever create one project.
MAX_TOOL_CALLS = 1


# Maximum previous conversation messages.
MAX_CONTEXT_MESSAGES = int(
    os.getenv(
        "ZOE_PROJECT_CREATOR_MAX_CONTEXT_MESSAGES",
        "8",
    )
)


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = {
    "python",
    "cpp",
    "c++",
    "react",
    "reactnative",
    "react native",
    "rust",
    "tauri",
}


# ============================================================
# DESTINATIONS
# ============================================================

SUPPORTED_DESTINATIONS = {
    "local",
    "github",
}


# ============================================================
# TOOLS
# ============================================================

TOOLS = {
    "create_project": create_project,
}


# ============================================================
# GROQ TOOLS
# ============================================================

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": (
                "Create exactly one new software project using "
                "ZOE's project creator. "
                "A project requires THREE values explicitly "
                "determined from the user's request: "
                "project name, language/framework, and destination. "
                "Destination is REQUIRED and must be either "
                "'local' or 'github'. "
                "NEVER assume local when the user has not specified "
                "a destination. If destination is missing, ask the "
                "user before calling this tool. "
                "Supports Python, C++, React, React Native, Rust, "
                "and Tauri. "
                "If destination is github and repository visibility "
                "was not specified, github_private should be true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The project name. "
                            "Must come from the user's request "
                            "or an explicit contextual reference."
                        ),
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            "The project language or framework. "
                            "Supported values: python, cpp, c++, "
                            "react, reactnative, react native, "
                            "rust, tauri."
                        ),
                    },
                    "destination": {
                        "type": "string",
                        "enum": [
                            "local",
                            "github",
                        ],
                        "description": (
                            "REQUIRED. Where the project should be "
                            "created. 'local' creates the project "
                            "locally. 'github' creates the local "
                            "project and publishes it to GitHub. "
                            "Do not choose a destination unless the "
                            "user explicitly specified it or clearly "
                            "referred to an earlier destination."
                        ),
                    },
                    "github_private": {
                        "type": "boolean",
                        "description": (
                            "GitHub repository visibility. "
                            "True means private. False means public. "
                            "Only relevant when destination is github. "
                            "If the user chose GitHub but did not "
                            "specify visibility, use true."
                        ),
                    },
                },
                "required": [
                    "name",
                    "language",
                    "destination",
                ],
            },
        },
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's specialized Project Creator Agent.

Your job is to understand natural-language project creation
requests and use the create_project tool to create exactly
one project.

You are NOT the main ZOE personality.

You are a precise software project automation agent.

============================================================
SUPPORTED PROJECT TYPES
============================================================

The project creator supports:

- Python
- C++
- React
- React Native
- Rust
- Tauri

Accepted language values:

python
cpp
c++
react
reactnative
react native
rust
tauri


============================================================
REQUIRED INFORMATION
============================================================

EVERY project creation request requires THREE pieces of
information:

1. Project name
2. Language/framework
3. Destination

Destination is REQUIRED.

The user MUST specify whether the project should be:

- local
- github

NEVER assume local.

NEVER assume github.

NEVER interpret "create a project" as meaning local.

If destination is missing, DO NOT call the tool.

Ask the user for the destination.

Example:

User:
"Create a Python project called ARC."

Response:

"Should I create it locally or on GitHub?"

Do NOT call create_project.


============================================================
MISSING INFORMATION
============================================================

If the project name is missing:

Ask:

"What should I call the project?"

If the language/framework is missing:

Ask:

"Which language or framework should I use?"

If the destination is missing:

Ask:

"Should I create it locally or on GitHub?"

If multiple required values are missing, ask for all
missing values in one concise question.

Example:

User:
"Create a project."

Response:

"What should I call the project, which language or framework
should it use, and should I create it locally or on GitHub?"

Do not call the tool until all three required values are known.


============================================================
DESTINATION
============================================================

Explicit local examples:

"create it locally"
"create locally"
"local project"
"make it local"
"keep it local"
"on my computer"
"on my PC"
"only locally"
"don't put it on GitHub"

These mean:

destination = "local"


Explicit GitHub examples:

"put it on GitHub"
"create it on GitHub"
"publish it to GitHub"
"push it to GitHub"
"GitHub repo"
"GitHub repository"
"host it on GitHub"

These mean:

destination = "github"


IMPORTANT:

If neither local nor GitHub is specified, ASK.

Do NOT silently choose local.


============================================================
GITHUB VISIBILITY
============================================================

If destination is github:

Default repository visibility is PRIVATE.

Therefore:

"create it on GitHub"

means:

destination = "github"
github_private = true


If the user explicitly says:

"make it public"
"public repository"
"public GitHub repo"

use:

github_private = false


If the user explicitly says:

"private repository"
"private GitHub repo"

use:

github_private = true


If destination is local:

github_private is irrelevant.

The Python validation layer will normalize it to true.


============================================================
PROJECT NAME
============================================================

Use the project name exactly as supplied by the user.

Example:

"Create a project called ARC"

name = "ARC"


Do NOT convert:

"MyApp"

into:

"my-app"

unless the user explicitly requests that.


============================================================
LANGUAGE INTERPRETATION
============================================================

Map natural language to supported project types.

Examples:

"Python app"
-> python

"Python project"
-> python

"C++ project"
-> cpp

"C plus plus project"
-> cpp

"React app"
-> react

"React Native app"
-> reactnative

"Rust application"
-> rust

"Tauri desktop app"
-> tauri


============================================================
CORE RULES
============================================================

1. Never claim a project was created unless the tool reports
   success.

2. Never invent a project path.

3. Never invent a GitHub URL.

4. Always use create_project for actual creation.

5. Never manually reproduce project creation logic.

6. If the user explicitly specifies a project name, use it.

7. If the user explicitly specifies a language/framework,
   use it.

8. Destination is REQUIRED.

9. Never assume local.

10. Never assume GitHub.

11. Do not call create_project until name, language, and
    destination are known.

12. GitHub repositories are private by default.

13. Public GitHub repositories require an explicit public
    request.

14. Never silently change the user's project name.

15. Never silently change the requested language.

16. Do not create multiple projects unless the user explicitly
    asks for multiple projects.

17. Never greet the user.

18. Keep the final response concise.


============================================================
CONTEXT
============================================================

Conversation context may be supplied by ZOE.

Use context only when necessary to understand the current
request.

The current user message has priority over context.

Do NOT assume an old project name, language, or destination
applies to a new request.

Context may resolve an explicit reference such as:

"Create another one using the same language and destination."

In that case, the previous values may be reused.

But if the current request does not clearly refer to previous
values, do not reuse them.

Never use context to silently turn a missing destination into
local.


============================================================
TOOL EXECUTION
============================================================

Call create_project only when:

- name is known
- language is known
- destination is known

All three are mandatory.


============================================================
TOOL RESULT
============================================================

The create_project tool returns information such as:

success
name
language
destination
path
github_repo
message
error


If success is true:

For local:

"Created ARC locally at C:\\Users\\...\\Dev\\ARC."


For GitHub:

"Created ARC and published it to GitHub:
https://github.com/..."


If success is false:

Report the actual failure reason concisely.

Never claim success if creation or GitHub publication failed.


============================================================
FINAL RESPONSE STYLE
============================================================

Keep responses short.

Good:

"Created ARC locally at C:\\Users\\Hasin\\Dev\\ARC."

Good:

"Created ARC and published it to GitHub:
https://github.com/username/ARC"

Good:

"Project creation failed: ARC already exists."

Do not explain the internal reasoning or tool process.
"""


# ============================================================
# DESTINATION DETECTION
# ============================================================

def detect_explicit_destination(
    query: str,
) -> str | None:
    """
    Detect whether the user's current request explicitly
    specifies local or GitHub.

    IMPORTANT:
    This function intentionally returns None when no
    destination is clearly specified.

    There is NO local fallback.
    """

    if not isinstance(query, str):
        return None

    text = query.strip().lower()

    if not text:
        return None

    # --------------------------------------------------------
    # GitHub patterns
    # --------------------------------------------------------

    github_patterns = [
        r"\bon\s+github\b",
        r"\bto\s+github\b",
        r"\binto\s+github\b",
        r"\bput\s+it\s+on\s+github\b",
        r"\bpublish(?:ed|ing)?\s+(?:it\s+)?to\s+github\b",
        r"\bpush(?:ed|ing)?\s+(?:it\s+)?to\s+github\b",
        r"\bgithub\s+repo\b",
        r"\bgithub\s+repository\b",
        r"\bgithub\s+project\b",
        r"\bhost\s+(?:it\s+)?on\s+github\b",
        r"\bcreate\s+(?:it\s+)?on\s+github\b",
    ]

    for pattern in github_patterns:
        if re.search(pattern, text):
            return "github"

    # --------------------------------------------------------
    # Local patterns
    # --------------------------------------------------------

    local_patterns = [
        r"\blocally\b",
        r"\blocal\b",
        r"\bcreate\s+(?:it\s+)?locally\b",
        r"\bkeep\s+(?:it\s+)?local\b",
        r"\bon\s+my\s+(?:computer|pc|machine)\b",
        r"\bon\s+my\s+computer\b",
        r"\bon\s+my\s+pc\b",
        r"\bon\s+my\s+machine\b",
        r"\bon\s+this\s+(?:computer|pc|machine)\b",
        r"\bon\s+the\s+local\s+machine\b",
        r"\bon\s+the\s+computer\b",
        r"\bon\s+the\s+pc\b",
        r"\bon\s+the\s+machine\b",
        r"\bdon['’]?t\s+(?:put|publish|push)\s+(?:it\s+)?on\s+github\b",
        r"\bdo\s+not\s+(?:put|publish|push)\s+(?:it\s+)?on\s+github\b",
        r"\bwithout\s+github\b",
        r"\bonly\s+locally\b",
    ]

    for pattern in local_patterns:
        if re.search(pattern, text):
            return "local"

    # --------------------------------------------------------
    # No explicit destination.
    # --------------------------------------------------------

    return None


# ============================================================
# DESTINATION QUESTION
# ============================================================

def destination_question() -> str:
    """
    Standard question used whenever destination is missing.
    """

    return (
        "Should I create it locally or on GitHub?"
    )


# ============================================================
# ARGUMENT NORMALIZATION
# ============================================================

def normalize_arguments(
    tool_name: str,
    arguments: Any,
) -> dict[str, Any]:
    """
    Validate and sanitize model-generated project arguments.

    IMPORTANT:
    destination has NO default.
    """

    # --------------------------------------------------------
    # Parse JSON if necessary.
    # --------------------------------------------------------

    if isinstance(arguments, str):
        raw = arguments.strip()

        if not raw:
            arguments = {}

        else:
            try:
                arguments = json.loads(raw)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON arguments for tool "
                    f"'{tool_name}'."
                ) from exc

    # --------------------------------------------------------
    # None -> empty object.
    # --------------------------------------------------------

    if arguments is None:
        arguments = {}

    # --------------------------------------------------------
    # Arguments must be a dictionary.
    # --------------------------------------------------------

    if not isinstance(arguments, dict):
        raise ValueError(
            f"Arguments for tool '{tool_name}' "
            "must be an object."
        )

    # --------------------------------------------------------
    # Validate tool.
    # --------------------------------------------------------

    if tool_name not in TOOLS:
        raise ValueError(
            f"Unknown project creator tool: {tool_name}"
        )

    # --------------------------------------------------------
    # Required name.
    # --------------------------------------------------------

    name = arguments.get("name")

    if not isinstance(name, str):
        raise ValueError(
            "Project name must be a string."
        )

    name = name.strip()

    if not name:
        raise ValueError(
            "Project name cannot be empty."
        )

    # --------------------------------------------------------
    # Required language.
    # --------------------------------------------------------

    language = arguments.get("language")

    if not isinstance(language, str):
        raise ValueError(
            "Project language must be a string."
        )

    language = language.strip().lower()

    if not language:
        raise ValueError(
            "Project language cannot be empty."
        )

    # --------------------------------------------------------
    # Validate language.
    # --------------------------------------------------------

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language or framework: "
            f"{language}"
        )

    # --------------------------------------------------------
    # Normalize C++.
    # --------------------------------------------------------

    if language == "c++":
        language = "cpp"

    # --------------------------------------------------------
    # Normalize React Native.
    # --------------------------------------------------------

    if language == "react native":
        language = "reactnative"

    # --------------------------------------------------------
    # REQUIRED DESTINATION.
    #
    # There is intentionally NO:
    #
    # arguments.get("destination", "local")
    #
    # --------------------------------------------------------

    destination = arguments.get("destination")

    if not isinstance(destination, str):
        raise ValueError(
            "Project destination is required. "
            "Choose either 'local' or 'github'."
        )

    destination = destination.strip().lower()

    if not destination:
        raise ValueError(
            "Project destination is required. "
            "Choose either 'local' or 'github'."
        )

    if destination not in SUPPORTED_DESTINATIONS:
        raise ValueError(
            "Destination must be 'local' or 'github'."
        )

    # --------------------------------------------------------
    # GitHub visibility.
    #
    # Private is the only default because visibility is not
    # required when creating a GitHub project.
    # --------------------------------------------------------

    github_private = arguments.get(
        "github_private",
        True,
    )

    if not isinstance(
        github_private,
        bool,
    ):
        raise ValueError(
            "github_private must be a boolean."
        )

    # --------------------------------------------------------
    # Local projects don't use GitHub visibility.
    # --------------------------------------------------------

    if destination == "local":
        github_private = True

    return {
        "name": name,
        "language": language,
        "destination": destination,
        "github_private": github_private,
    }


# ============================================================
# VALIDATE CURRENT QUERY DESTINATION
# ============================================================

def validate_query_destination(
    query: str,
    arguments: Any,
) -> str | None:
    """
    Ensure that the model did not invent a destination.

    The current user request must explicitly specify local
    or GitHub before create_project can execute.

    Returns:
        None if valid.
        Error message if invalid.
    """

    explicit_destination = (
        detect_explicit_destination(query)
    )

    if explicit_destination is None:
        return (
            "Project destination was not specified. "
            "The user must choose local or GitHub."
        )

    # --------------------------------------------------------
    # Parse model arguments.
    # --------------------------------------------------------

    try:
        if isinstance(arguments, str):
            arguments = json.loads(
                arguments.strip() or "{}"
            )

    except Exception:
        return (
            "Invalid project creation arguments."
        )

    if not isinstance(arguments, dict):
        return (
            "Invalid project creation arguments."
        )

    model_destination = arguments.get(
        "destination"
    )

    if not isinstance(
        model_destination,
        str,
    ):
        return (
            "Project destination is required."
        )

    model_destination = (
        model_destination.strip().lower()
    )

    # --------------------------------------------------------
    # Prevent the model from choosing a different destination
    # than the one the user actually requested.
    # --------------------------------------------------------

    if (
        model_destination
        != explicit_destination
    ):
        return (
            "The requested destination does not match "
            "the destination specified by the user."
        )

    return None


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: Any,
    query: str | None = None,
) -> Any:
    """
    Execute the project creator safely.
    """

    function = TOOLS.get(tool_name)

    if function is None:
        return {
            "success": False,
            "error": (
                f"Unknown project creator tool: "
                f"{tool_name}"
            ),
        }

    # --------------------------------------------------------
    # Destination enforcement.
    # --------------------------------------------------------

    if query is not None:
        destination_error = (
            validate_query_destination(
                query,
                arguments,
            )
        )

        if destination_error:
            return {
                "success": False,
                "error": destination_error,
                "requires_user_input": True,
                "missing": [
                    "destination"
                ],
            }

    # --------------------------------------------------------
    # Normalize arguments.
    # --------------------------------------------------------

    try:
        cleaned_arguments = normalize_arguments(
            tool_name,
            arguments,
        )

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # Execute project creator.
    # --------------------------------------------------------

    try:
        result = function(
            **cleaned_arguments
        )

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # Handle None.
    # --------------------------------------------------------

    if result is None:
        return {
            "success": False,
            "error": (
                "Project creator returned no result."
            ),
        }

    # --------------------------------------------------------
    # Convert dataclass result to dictionary.
    # --------------------------------------------------------

    if hasattr(
        result,
        "__dataclass_fields__",
    ):
        try:
            from dataclasses import asdict

            return asdict(result)

        except Exception:
            pass

    # --------------------------------------------------------
    # Already a dictionary.
    # --------------------------------------------------------

    if isinstance(result, dict):
        return result

    # --------------------------------------------------------
    # Fallback.
    # --------------------------------------------------------

    return {
        "success": True,
        "result": str(result),
    }


# ============================================================
# RESULT STATUS
# ============================================================

def result_succeeded(
    result: Any,
) -> bool:
    """
    Determine whether project creation succeeded.
    """

    if result is None:
        return False

    if isinstance(result, dict):
        return (
            result.get(
                "success",
                False,
            )
            is True
        )

    return False


# ============================================================
# CONTEXT CLEANING
# ============================================================

def clean_context(
    context: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """
    Keep only safe user/assistant conversation messages.
    """

    if not context:
        return []

    cleaned: list[
        dict[str, str]
    ] = []

    for message in context[
        -MAX_CONTEXT_MESSAGES:
    ]:
        if not isinstance(
            message,
            dict,
        ):
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not isinstance(
            content,
            str,
        ):
            continue

        content = content.strip()

        if not content:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content,
            }
        )

    return cleaned


# ============================================================
# BUILD MESSAGES
# ============================================================

def build_messages(
    query: str,
    context: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build the model conversation.
    """

    messages: list[
        dict[str, Any]
    ] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
    ]

    messages.extend(
        clean_context(context)
    )

    messages.append(
        {
            "role": "user",
            "content": query.strip(),
        }
    )

    return messages


# ============================================================
# SERIALIZE TOOL RESULT
# ============================================================

def serialize_tool_result(
    result: Any,
) -> str:
    """
    Serialize project creator output for the model.
    """

    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )

    except Exception:
        return str(result)


# ============================================================
# TOOL CALL SIGNATURE
# ============================================================

def tool_call_signature(
    tool_name: str,
    arguments: Any,
) -> str:
    """
    Create a stable signature for duplicate-call detection.
    """

    try:
        normalized = normalize_arguments(
            tool_name,
            arguments,
        )

        return (
            tool_name
            + ":"
            + json.dumps(
                normalized,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
        )

    except Exception:
        return (
            tool_name
            + ":"
            + str(arguments)
        )


# ============================================================
# PROJECT CREATOR AGENT
# ============================================================

class ProjectCreatorAgent:
    """
    Specialized project creation reasoning agent.

    Required user information:

    - project name
    - language/framework
    - destination

    Destination MUST be explicitly chosen by the user.

    The agent:

    - receives the user's project request
    - determines project name
    - determines language/framework
    - determines destination
    - determines GitHub visibility
    - executes the existing project_creator.py logic
    - analyzes the result
    - returns a concise response
    """

    def __init__(
        self,
        model: str = MODEL,
    ):
        self.model = model

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        query: str,
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Execute a project creation request.
        """

        # ----------------------------------------------------
        # Validate query.
        # ----------------------------------------------------

        if (
            not query
            or not query.strip()
        ):
            return (
                "I need a project creation request."
            )

        query = query.strip()

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Check destination BEFORE invoking the LLM.
        #
        # This prevents the model from inventing "local".
        # ----------------------------------------------------

        explicit_destination = (
            detect_explicit_destination(query)
        )

        if explicit_destination is None:

            # ------------------------------------------------
            # We deliberately do not let the model decide
            # whether the user meant local.
            # ------------------------------------------------

            return destination_question()

        messages = build_messages(
            query=query,
            context=context,
        )

        tool_calls_used = 0
        steps_used = 0

        previous_calls: set[str] = set()

        last_tool_error: str | None = None

        # ====================================================
        # AGENT LOOP
        # ====================================================

        while (
            steps_used
            < MAX_AGENT_STEPS
        ):
            steps_used += 1

            print()
            print(
                "[Project Creator Agent]"
                f" Step {steps_used}/"
                f"{MAX_AGENT_STEPS}"
            )

            # ------------------------------------------------
            # Model request.
            # ------------------------------------------------

            try:
                response = (
                    client
                    .chat
                    .completions
                    .create(
                        model=self.model,
                        messages=messages,
                        tools=GROQ_TOOLS,
                        tool_choice="auto",
                        temperature=0.1,
                        max_completion_tokens=1000,
                    )
                )

            except Exception as exc:
                print(
                    "[Project Creator Agent]"
                    f" Model error: {exc}"
                )

                return (
                    "I couldn't process the "
                    "project creation request "
                    "right now."
                )

            # ------------------------------------------------
            # Validate response.
            # ------------------------------------------------

            try:
                message = (
                    response
                    .choices[0]
                    .message
                )

            except Exception as exc:
                print(
                    "[Project Creator Agent]"
                    f" Invalid response: {exc}"
                )

                return (
                    "I couldn't process the "
                    "project creator response."
                )

            tool_calls = (
                message.tool_calls
                or []
            )

            # =================================================
            # FINAL ANSWER
            # =================================================

            if not tool_calls:

                content = (
                    message.content
                    or ""
                ).strip()

                if content:
                    return content

                if last_tool_error:
                    return (
                        "Project creation failed: "
                        f"{last_tool_error}"
                    )

                return (
                    "I couldn't determine "
                    "the project creation request."
                )

            # =================================================
            # TOOL BUDGET
            # =================================================

            if (
                tool_calls_used
                >= MAX_TOOL_CALLS
            ):
                return (
                    "I couldn't complete "
                    "the project creation request."
                )

            # =================================================
            # PRESERVE ASSISTANT TOOL CALL
            # =================================================

            assistant_message: dict[
                str,
                Any,
            ] = {
                "role": "assistant",
                "content": (
                    message.content
                    or ""
                ),
                "tool_calls": [],
            }

            for tool_call in tool_calls:

                assistant_message[
                    "tool_calls"
                ].append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": (
                                tool_call
                                .function
                                .name
                            ),
                            "arguments": (
                                tool_call
                                .function
                                .arguments
                            ),
                        },
                    }
                )

            messages.append(
                assistant_message
            )

            # =================================================
            # EXECUTE TOOL CALLS
            # =================================================

            for tool_call in tool_calls:

                # --------------------------------------------
                # Global budget.
                # --------------------------------------------

                if (
                    tool_calls_used
                    >= MAX_TOOL_CALLS
                ):
                    return (
                        "I couldn't complete "
                        "the project creation request."
                    )

                tool_name = (
                    tool_call
                    .function
                    .name
                )

                raw_arguments = (
                    tool_call
                    .function
                    .arguments
                )

                print(
                    "[Project Creator Agent]"
                    f" Tool: {tool_name}"
                )

                print(
                    "[Project Creator Agent]"
                    f" Arguments: "
                    f"{raw_arguments}"
                )

                # --------------------------------------------
                # Destination enforcement.
                # --------------------------------------------

                destination_error = (
                    validate_query_destination(
                        query,
                        raw_arguments,
                    )
                )

                if destination_error:

                    last_tool_error = (
                        destination_error
                    )

                    print(
                        "[Project Creator Agent]"
                        " Tool call blocked:"
                        f" {destination_error}"
                    )

                    return destination_question()

                # --------------------------------------------
                # Duplicate detection.
                #
                # Do this BEFORE consuming the tool budget.
                # --------------------------------------------

                signature = (
                    tool_call_signature(
                        tool_name,
                        raw_arguments,
                    )
                )

                if signature in previous_calls:

                    result = {
                        "success": False,
                        "error": (
                            "The same project creation "
                            "request was already executed "
                            "during this request."
                        ),
                    }

                    last_tool_error = (
                        result["error"]
                    )

                    print(
                        "[Project Creator Agent]"
                        " Duplicate tool call blocked."
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": (
                                tool_call.id
                            ),
                            "content": (
                                serialize_tool_result(
                                    result
                                )
                            ),
                        }
                    )

                    continue

                # --------------------------------------------
                # Consume actual execution budget.
                # --------------------------------------------

                tool_calls_used += 1

                previous_calls.add(
                    signature
                )

                # --------------------------------------------
                # Execute project creator.
                # --------------------------------------------

                result = execute_tool(
                    tool_name,
                    raw_arguments,
                    query=query,
                )

                if not result_succeeded(
                    result
                ):
                    if isinstance(
                        result,
                        dict,
                    ):
                        last_tool_error = str(
                            result.get(
                                "error",
                                result.get(
                                    "message",
                                    "Project creation failed.",
                                ),
                            )
                        )

                    else:
                        last_tool_error = (
                            "Project creation failed."
                        )

                else:
                    last_tool_error = None

                print(
                    "[Project Creator Agent]"
                    f" Result: {result}"
                )

                # --------------------------------------------
                # Return result to model.
                # --------------------------------------------

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.id
                        ),
                        "content": (
                            serialize_tool_result(
                                result
                            )
                        ),
                    }
                )

                # ============================================
                # SUCCESS
                # ============================================

                if result_succeeded(
                    result
                ):

                    success_data = result

                    project_name = (
                        success_data.get(
                            "name",
                            "project",
                        )
                    )

                    destination = (
                        success_data.get(
                            "destination",
                            explicit_destination,
                        )
                    )

                    project_path = (
                        success_data.get(
                            "path"
                        )
                    )

                    github_repo = (
                        success_data.get(
                            "github_repo"
                        )
                    )

                    # ----------------------------------------
                    # GitHub
                    # ----------------------------------------

                    if destination == "github":

                        if github_repo:
                            return (
                                f"Created {project_name} "
                                f"and published it to GitHub: "
                                f"{github_repo}"
                            )

                        return (
                            f"Created {project_name}, "
                            "but no GitHub URL was returned."
                        )

                    # ----------------------------------------
                    # Local
                    # ----------------------------------------

                    if destination == "local":

                        if project_path:
                            return (
                                f"Created {project_name} "
                                f"locally at "
                                f"{project_path}."
                            )

                        return (
                            f"Created {project_name} "
                            "locally."
                        )

                    # ----------------------------------------
                    # Unknown destination returned by tool.
                    # ----------------------------------------

                    return (
                        f"Created {project_name} "
                        "successfully."
                    )

        # ====================================================
        # STEP LIMIT
        # ====================================================

        if last_tool_error:
            return (
                "Project creation failed: "
                f"{last_tool_error}"
            )

        return (
            "I couldn't complete the "
            "project creation request."
        )


# ============================================================
# SINGLETON
# ============================================================

project_creator_agent = ProjectCreatorAgent()


# ============================================================
# PUBLIC ZOE INTERFACE
# ============================================================

def run_project_creator_agent(
    query: str,
    context: list[dict[str, Any]] | None = None,
) -> str:
    """
    Public interface used by ZOE Runtime.
    """

    return project_creator_agent.run(
        query=query,
        context=context,
    )


# ============================================================
# STANDALONE CLI
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "              ZOE PROJECT CREATOR AGENT"
    )
    print("=" * 70)
    print()

    print(
        f"Model: {MODEL}"
    )

    print(
        "Groq API: key 3"
    )

    print(
        f"Max steps: "
        f"{MAX_AGENT_STEPS}"
    )

    print(
        f"Max tool calls: "
        f"{MAX_TOOL_CALLS}"
    )

    print()

    print("Examples:")

    print(
        "  Create a Python project called ARC locally"
    )

    print(
        "  Create a Tauri app called AegRCr locally"
    )

    print(
        "  Create a React project called Dashboard "
        "and publish it to GitHub"
    )

    print(
        "  Create a private Rust project called Sentinel "
        "on GitHub"
    )

    print(
        "  Create a public C++ project called Victor "
        "on GitHub"
    )

    print()

    print(
        "If destination is omitted, ZOE will ask:"
    )

    print(
        '  "Should I create it locally or on GitHub?"'
    )

    print()

    print(
        "Type 'exit' or 'quit' to stop."
    )

    print()

    while True:

        try:
            query = input(
                "You: "
            ).strip()

        except KeyboardInterrupt:
            print(
                "\nExiting..."
            )
            break

        except EOFError:
            print(
                "\nExiting..."
            )
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        try:

            response = (
                run_project_creator_agent(
                    query
                )
            )

            print(
                f"ZOE: {response}"
            )

            print()

        except Exception as exc:

            print(
                f"ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print()