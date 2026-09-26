from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from groq import Groq

from backend.functions.reminder import (
    create_reminder,
    delete_reminder,
    get_reminders,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.getenv(
    "ZOE_REMINDER_MODEL",
    "openai/gpt-oss-120b",
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY1"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY1 is not set."
    )

client = Groq(
    api_key=GROQ_API_KEY
)

ZOE_TIMEZONE = ZoneInfo(
    "Asia/Dhaka"
)

MAX_AGENT_STEPS = int(
    os.getenv(
        "ZOE_REMINDER_MAX_STEPS",
        "6",
    )
)

MAX_TOOL_CALLS = int(
    os.getenv(
        "ZOE_REMINDER_MAX_TOOL_CALLS",
        "30",
    )
)

MAX_IDENTICAL_FAILURES = int(
    os.getenv(
        "ZOE_REMINDER_MAX_IDENTICAL_FAILURES",
        "1",
    )
)


# ============================================================
# NUMBER WORDS
# ============================================================

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOLS = {
    "create_reminder": create_reminder,
    "get_reminders": get_reminders,
    "delete_reminder": delete_reminder,
}


# ============================================================
# GROQ TOOLS
# ============================================================

GROQ_TOOLS = [

    # ========================================================
    # CREATE
    # ========================================================

    {
        "type": "function",
        "function": {
            "name": "create_reminder",

            "description": (
                "Create exactly ONE Todoist reminder. "
                "The date/time must already be resolved before "
                "calling this function. "
                "For relative expressions such as "
                "'in 5 minutes' or 'in 2 hours', resolve them "
                "using the CURRENT ZOE LOCAL TIME supplied in "
                "the system context. "
                "Use an exact ISO-8601 datetime with +06:00 "
                "when an exact time is required. "
                "For recurring reminders, preserve the "
                "natural-language Todoist recurrence expression. "
                "Never invent missing date or time information."
            ),

            "parameters": {
                "type": "object",

                "properties": {

                    "name": {
                        "type": "string",

                        "description": (
                            "Reminder description. "
                            "Preserve the user's wording "
                            "as much as practical."
                        ),
                    },

                    "due_string": {
                        "type": "string",

                        "description": (
                            "The COMPLETE resolved Todoist "
                            "due expression. "
                            "For exact one-time reminders use "
                            "an ISO-8601 datetime with +06:00. "
                            "Example: "
                            "'2026-09-20T18:27:00+06:00'. "
                            "For recurring reminders use a "
                            "Todoist recurrence expression such "
                            "as 'every Monday at 7 PM'. "
                            "For date-only reminders use the "
                            "appropriate natural-language date."
                        ),
                    },
                },

                "required": [
                    "name",
                ],
            },
        },
    },

    # ========================================================
    # GET
    # ========================================================

    {
        "type": "function",
        "function": {
            "name": "get_reminders",

            "description": (
                "Retrieve all current active Todoist reminders. "
                "Use this whenever existing Todoist state is "
                "required, especially deletion, updates, "
                "relative references, categories, dates, or "
                "phrases such as 'that one', 'those', "
                "'all of them', 'the last one', or "
                "'the ones I just created'."
            ),

            "parameters": {
                "type": "object",

                "properties": {},

                "required": [],
            },
        },
    },

    # ========================================================
    # DELETE
    # ========================================================

    {
        "type": "function",
        "function": {
            "name": "delete_reminder",

            "description": (
                "Delete exactly ONE existing Todoist reminder. "
                "For destructive requests, inspect Todoist "
                "first when the target is not unquestionably "
                "known. "
                "Never invent an existing reminder. "
                "Use allow_partial_match=true only when the "
                "user intentionally identified the reminder "
                "using a partial description."
                "checking of, marking as done means deletion"
            ),

            "parameters": {
                "type": "object",

                "properties": {

                    "name": {
                        "type": "string",

                        "description": (
                            "Exact reminder name when known, "
                            "or a specific identifying fragment "
                            "when intentional partial matching "
                            "is required."
                        ),
                    },

                    "allow_partial_match": {
                        "type": "boolean",

                        "description": (
                            "True only when the target is "
                            "intentionally identified using "
                            "a partial description."
                        ),
                    },
                },

                "required": [
                    "name",
                ],
            },
        },
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's autonomous Todoist Reminder Action Agent.

Your only responsibility is executing reminder objectives.

You are NOT a conversational assistant.

Do not produce conversational replies.

The runtime owns the final user-facing response.


============================================================
CURRENT TIME
============================================================

ZOE operates in:

    Asia/Dhaka
    UTC+06:00

The runtime supplies the CURRENT ZOE LOCAL TIME in the
execution context.

Use that time when resolving relative expressions.

Examples:

If current time is:

    2026-09-20T18:22:00+06:00

Then:

    "in 5 minutes"
    -> 2026-09-20T18:27:00+06:00

    "in 10 minutes"
    -> 2026-09-20T18:32:00+06:00

    "in 1 hour"
    -> 2026-09-20T19:22:00+06:00

Do NOT use UTC as the user's clock.

Do NOT ask Todoist to interpret relative expressions when an
exact datetime can be calculated.

For exact one-time reminders, pass the resolved ISO datetime
as `due_string`.


============================================================
CORE RULES
============================================================

1. ACT when reminder intent is clear.

2. Use Todoist as the authoritative source for existing
   reminders.

3. Never invent existing reminders, IDs, names, dates, or
   times.

4. Create every reminder requested by the user.

5. For bulk operations, complete every independent operation.

6. Continue independent operations after an individual
   failure.

7. Do not repeatedly execute the same failed operation.

8. Do not perform unrelated actions.

9. Do not claim objective completion unless the requested
   objective was actually completed.


============================================================
CREATION
============================================================

Create exactly one reminder per requested reminder.

Examples:

"Remind me to study."

    create_reminder(
        name="study"
    )

"Remind me to study tomorrow."

    create_reminder(
        name="study",
        due_string="tomorrow"
    )

"Remind me to study at 8 PM."

    create_reminder(
        name="study",
        due_string="today at 8 PM"
    )

"Remind me to study every Monday at 7 PM."

    create_reminder(
        name="study",
        due_string="every Monday at 7 PM"
    )


============================================================
RELATIVE TIME
============================================================

For expressions such as:

    in 5 minutes
    in five minutes
    in 20 minutes
    in one hour
    in 2 hours

calculate the exact datetime using the supplied current
Asia/Dhaka time.

Do not pass:

    "in 5 minutes"

to Todoist when an exact datetime can be produced.

Instead pass something like:

    "2026-09-20T18:27:00+06:00"


============================================================
EXPLICIT TIME
============================================================

For:

    tomorrow at 8 PM
    Friday at 7:30 PM
    September 25 at 9 AM

resolve the date/time appropriately.

Use +06:00 for exact one-time datetimes.

Do not convert Bangladesh local time into UTC manually.


============================================================
RECURRING REMINDERS
============================================================

For recurring requests, Todoist's natural-language recurrence
syntax may be passed directly.

Examples:

    every Monday at 7 PM
    every day at 8 AM
    every Friday at 6 PM

Do not turn recurring expressions into a single timestamp.


============================================================
DELETION SAFETY
============================================================

Deletion is destructive.

If the user refers to existing reminders by category,
partial description, date, relative reference, or ambiguous
language, inspect Todoist first.

Examples:

    delete all my study reminders
    delete my physics reminders
    remove everything for tomorrow
    delete that
    delete those
    delete the last one
    delete the ones I just created

After inspection, identify the intended target(s).

If one exact target is known:

    delete_reminder(
        name=<exact name>,
        allow_partial_match=false
    )

If the user's wording intentionally identifies a target using
a partial description:

    delete_reminder(
        name=<specific fragment>,
        allow_partial_match=true
    )

Do not broaden a deletion request unnecessarily.

If several reminders match equally and the request does not
clearly establish which one should be deleted, do not guess.


============================================================
BULK DELETION
============================================================

For:

    delete all my physics reminders

first call get_reminders.

Identify every matching reminder.

Delete each intended reminder independently.

Do not stop after deleting the first match.

Do not delete unrelated reminders.


============================================================
UPDATES
============================================================

There is no direct update operation.

If an existing reminder must be changed:

1. Find the existing reminder.
2. Delete the old reminder.
3. Create the corrected reminder.
4. Only treat the operation as successful if both mutations
   succeed.


============================================================
NO CONVERSATION HISTORY
============================================================

You receive only:

1. the current request
2. the current execution context
3. tool results generated during this execution

There is no previous conversation history.

Do not pretend to remember previous requests.


============================================================
TOOL DISCIPLINE
============================================================

Use get_reminders when existing Todoist state is required.

Do not repeatedly call get_reminders when the returned state
already provides the necessary information and no mutation has
occurred.

After destructive or bulk mutations, verification may be used
when it materially improves confidence.

Do not call tools merely to look busy.


============================================================
AMBIGUITY
============================================================

If the target cannot be safely determined from:

    current request
    current execution context
    current Todoist state

then stop rather than randomly modifying a reminder.

Never manufacture certainty.
"""


# ============================================================
# ARGUMENT RULES
# ============================================================

ALLOWED_ARGUMENTS = {

    "create_reminder": {
        "name",
        "due_string",
    },

    "get_reminders": set(),

    "delete_reminder": {
        "name",
        "allow_partial_match",
    },
}


REQUIRED_ARGUMENTS = {

    "create_reminder": {
        "name",
    },

    "get_reminders": set(),

    "delete_reminder": {
        "name",
    },
}


# ============================================================
# TIME RESOLUTION
# ============================================================

def _local_now() -> datetime:
    """
    Return the actual current ZOE time.
    """

    return datetime.now(
        ZOE_TIMEZONE
    )


def _parse_number(
    value: str,
) -> int | None:
    """
    Parse numeric or simple English number words.
    """

    value = (
        value
        .strip()
        .lower()
    )

    if value.isdigit():
        return int(value)

    return NUMBER_WORDS.get(
        value
    )


def _resolve_relative_time(
    text: str,
) -> str | None:
    """
    Resolve simple relative expressions into exact ISO
    datetimes.

    This belongs to the AGENT layer.

    Examples:

        in 5 minutes
        in five minutes
        in 2 hours
        in one hour

    Result:

        2026-09-20T18:27:00+06:00
    """

    if not text:
        return None

    value = (
        str(text)
        .strip()
        .lower()
    )

    value = re.sub(
        r"[,.!?]+$",
        "",
        value,
    )

    # --------------------------------------------------------
    # Minutes
    # --------------------------------------------------------

    match = re.fullmatch(
        r"in\s+([a-z0-9]+)\s+minutes?",
        value,
    )

    if match:

        amount = _parse_number(
            match.group(1)
        )

        if amount is not None:

            resolved = (
                _local_now()
                + timedelta(
                    minutes=amount
                )
            )

            return resolved.isoformat(
                timespec="seconds"
            )

    # --------------------------------------------------------
    # Hours
    # --------------------------------------------------------

    match = re.fullmatch(
        r"in\s+([a-z0-9]+)\s+hours?",
        value,
    )

    if match:

        amount = _parse_number(
            match.group(1)
        )

        if amount is not None:

            resolved = (
                _local_now()
                + timedelta(
                    hours=amount
                )
            )

            return resolved.isoformat(
                timespec="seconds"
            )

    return None


def _normalize_create_arguments(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize model-generated create arguments.

    The important behavior here is that relative durations are
    resolved in the agent before reaching the Todoist function.
    """

    cleaned = dict(
        arguments
    )

    name = cleaned.get(
        "name"
    )

    if name is not None:

        cleaned["name"] = str(
            name
        ).strip()

    due_string = cleaned.get(
        "due_string"
    )

    if due_string is not None:

        due_string = str(
            due_string
        ).strip()

        relative = _resolve_relative_time(
            due_string
        )

        if relative is not None:

            print(
                "[Reminder Agent] "
                f"Resolved relative time: "
                f"{due_string} -> {relative}"
            )

            due_string = relative

        cleaned[
            "due_string"
        ] = due_string

    return cleaned


# ============================================================
# NORMALIZE TOOL ARGUMENTS
# ============================================================

def normalize_arguments(
    name: str,
    arguments: Any,
) -> dict[str, Any]:
    """
    Validate and sanitize model-generated tool arguments.
    """

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    if isinstance(
        arguments,
        str,
    ):

        raw = arguments.strip()

        if not raw:

            arguments = {}

        else:

            try:

                arguments = json.loads(
                    raw
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON arguments "
                    f"for tool '{name}'."
                ) from exc

    # --------------------------------------------------------
    # None
    # --------------------------------------------------------

    if arguments is None:
        arguments = {}

    # --------------------------------------------------------
    # Object validation
    # --------------------------------------------------------

    if not isinstance(
        arguments,
        dict,
    ):

        raise ValueError(
            f"Arguments for tool '{name}' "
            "must be an object."
        )

    # --------------------------------------------------------
    # Tool validation
    # --------------------------------------------------------

    if name not in ALLOWED_ARGUMENTS:

        raise ValueError(
            f"Unknown reminder tool: {name!r}"
        )

    # --------------------------------------------------------
    # Remove unsupported arguments
    # --------------------------------------------------------

    cleaned = {
        key: value
        for key, value in arguments.items()
        if key in ALLOWED_ARGUMENTS[name]
    }

    # --------------------------------------------------------
    # Required arguments
    # --------------------------------------------------------

    missing = []

    for key in REQUIRED_ARGUMENTS[name]:

        value = cleaned.get(
            key
        )

        if value is None:

            missing.append(
                key
            )

            continue

        if (
            isinstance(
                value,
                str,
            )
            and not value.strip()
        ):

            missing.append(
                key
            )

    if missing:

        raise ValueError(
            f"Missing required argument(s) "
            f"for '{name}': "
            f"{', '.join(missing)}"
        )

    # --------------------------------------------------------
    # Strings
    # --------------------------------------------------------

    for key in (
        "name",
        "due_string",
    ):

        if key in cleaned:

            value = cleaned[key]

            if value is not None:

                cleaned[key] = str(
                    value
                ).strip()

    # --------------------------------------------------------
    # Boolean
    # --------------------------------------------------------

    if (
        name == "delete_reminder"
        and "allow_partial_match"
        in cleaned
    ):

        value = cleaned[
            "allow_partial_match"
        ]

        if isinstance(
            value,
            str,
        ):

            cleaned[
                "allow_partial_match"
            ] = (
                value.lower().strip()
                in {
                    "true",
                    "1",
                    "yes",
                    "y",
                }
            )

        elif not isinstance(
            value,
            bool,
        ):

            cleaned[
                "allow_partial_match"
            ] = bool(
                value
            )

    # --------------------------------------------------------
    # Agent-level time resolution
    # --------------------------------------------------------

    if name == "create_reminder":

        cleaned = (
            _normalize_create_arguments(
                cleaned
            )
        )

    return cleaned


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    name: str,
    arguments: Any,
) -> tuple[Any, dict[str, Any]]:

    if name not in TOOLS:

        raise ValueError(
            f"Unknown reminder tool: {name}"
        )

    cleaned_arguments = (
        normalize_arguments(
            name,
            arguments,
        )
    )

    result = TOOLS[name](
        **cleaned_arguments
    )

    return (
        result,
        cleaned_arguments,
    )


# ============================================================
# RESULT SUCCESS
# ============================================================

def tool_result_succeeded(
    result: Any,
) -> bool:

    if result is None:
        return False

    if isinstance(
        result,
        dict,
    ):

        if result.get(
            "success"
        ) is False:

            return False

        if result.get(
            "success"
        ) is True:

            return True

    return True


# ============================================================
# SERIALIZE TOOL RESULT
# ============================================================

def serialize_tool_result(
    result: Any,
) -> str:

    try:

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    except Exception:

        return json.dumps(
            {
                "success": False,
                "error": (
                    "Tool result could not "
                    "be serialized."
                ),
            },
            ensure_ascii=False,
        )


# ============================================================
# BUILD MESSAGES
# ============================================================

def build_messages(
    user_request: str,
) -> list[dict[str, Any]]:
    """
    Build the stateless model context.

    The actual current local time is explicitly supplied to
    the model so relative expressions have a reliable temporal
    reference.
    """

    now = _local_now()

    execution_context = (
        "CURRENT ZOE LOCAL TIME:\n"
        f"{now.isoformat(timespec='seconds')}\n\n"
        "TIMEZONE:\n"
        "Asia/Dhaka (UTC+06:00)"
    )

    return [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },

        {
            "role": "system",
            "content": execution_context,
        },

        {
            "role": "user",
            "content": user_request.strip(),
        },

    ]


# ============================================================
# TOOL CALL SIGNATURE
# ============================================================

def tool_call_signature(
    name: str,
    arguments: Any,
) -> str:

    try:

        cleaned = normalize_arguments(
            name,
            arguments,
        )

        return (
            name
            + ":"
            + json.dumps(
                cleaned,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
        )

    except Exception:

        return (
            name
            + ":"
            + str(arguments)
        )


# ============================================================
# REMINDER AGENT
# ============================================================

class ReminderAgent:

    def __init__(
        self,
        model: str = MODEL,
    ):

        self.model = model

    # ========================================================
    # RESULT
    # ========================================================

    @staticmethod
    def build_result(
        executed_operations: list[
            dict[str, Any]
        ],
        *,
        success: bool,
        action: str | None = None,
        error: str | None = None,
        last_result: Any = None,
    ) -> dict[str, Any]:

        result: dict[str, Any] = {
            "success": success,
            "action": action,
            "operations": executed_operations,
            "tool_calls": len(
                executed_operations
            ),
        }

        if last_result is not None:

            result[
                "last_result"
            ] = last_result

        if error is not None:

            result[
                "error"
            ] = error

        return result

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        user_request: str,
    ) -> Any:

        if (
            not user_request
            or not user_request.strip()
        ):

            return {
                "success": False,
                "action": None,
                "error": (
                    "Reminder request "
                    "cannot be empty."
                ),
                "operations": [],
                "tool_calls": 0,
            }

        messages = build_messages(
            user_request
        )

        total_tool_calls = 0

        executed_operations: list[
            dict[str, Any]
        ] = []

        last_tool_result: Any = None

        identical_failures: dict[
            str,
            int,
        ] = {}

        # ====================================================
        # AGENT LOOP
        # ====================================================

        for step in range(
            1,
            MAX_AGENT_STEPS + 1,
        ):

            print()
            print(
                "[Reminder Agent]"
                f" Step {step}/{MAX_AGENT_STEPS}"
            )

            # ------------------------------------------------
            # Current time diagnostic
            # ------------------------------------------------

            print(
                "[Reminder Agent]"
                f" Local time: "
                f"{_local_now().isoformat(
                    timespec='seconds'
                )}"
            )

            # ------------------------------------------------
            # Model call
            # ------------------------------------------------

            try:

                response = (
                    client.chat.completions.create(

                        model=self.model,

                        messages=messages,

                        tools=GROQ_TOOLS,

                        tool_choice="auto",

                        temperature=0,

                        max_completion_tokens=1200,
                    )
                )

            except Exception as exc:

                return self.build_result(
                    executed_operations,
                    success=False,
                    action="model_error",
                    error=(
                        "Groq request failed: "
                        f"{exc}"
                    ),
                    last_result=(
                        last_tool_result
                    ),
                )

            # ------------------------------------------------
            # Response validation
            # ------------------------------------------------

            try:

                message = (
                    response
                    .choices[0]
                    .message
                )

            except Exception as exc:

                return self.build_result(
                    executed_operations,
                    success=False,
                    action="model_error",
                    error=(
                        "Invalid Groq response: "
                        f"{exc}"
                    ),
                    last_result=(
                        last_tool_result
                    ),
                )

            tool_calls = (
                message.tool_calls
                or []
            )

            # ------------------------------------------------
            # No tool calls
            # ------------------------------------------------

            if not tool_calls:

                if executed_operations:

                    mutating_operations = [
                        operation
                        for operation
                        in executed_operations
                        if operation.get(
                            "mutating",
                            False,
                        )
                    ]

                    if mutating_operations:

                        objective_success = all(
                            operation.get(
                                "success",
                                False,
                            )
                            for operation
                            in mutating_operations
                        )

                    else:

                        objective_success = all(
                            operation.get(
                                "success",
                                False,
                            )
                            for operation
                            in executed_operations
                        )

                    return self.build_result(
                        executed_operations,
                        success=objective_success,
                        action="completed",
                        last_result=(
                            last_tool_result
                        ),
                    )

                return self.build_result(
                    executed_operations,
                    success=False,
                    action="no_tool_call",
                    error=(
                        "Reminder Agent did not "
                        "produce a tool call."
                    ),
                    last_result=(
                        message.content
                        or ""
                    ).strip(),
                )

            # ------------------------------------------------
            # Preserve assistant message
            # ------------------------------------------------

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

                function_name = (
                    tool_call
                    .function
                    .name
                )

                raw_arguments = (
                    tool_call
                    .function
                    .arguments
                )

                assistant_message[
                    "tool_calls"
                ].append(
                    {
                        "id": tool_call.id,

                        "type": "function",

                        "function": {
                            "name": function_name,
                            "arguments": (
                                raw_arguments
                            ),
                        },
                    }
                )

            messages.append(
                assistant_message
            )

            # ------------------------------------------------
            # Execute every tool call
            # ------------------------------------------------

            for tool_call in tool_calls:

                if (
                    total_tool_calls
                    >= MAX_TOOL_CALLS
                ):

                    return self.build_result(
                        executed_operations,
                        success=False,
                        action="tool_limit",
                        error=(
                            "Reminder Agent reached "
                            "the maximum number of "
                            "tool executions."
                        ),
                        last_result=(
                            last_tool_result
                        ),
                    )

                total_tool_calls += 1

                function_name = (
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
                    "[Reminder Agent]"
                    f" Tool #{total_tool_calls}: "
                    f"{function_name}"
                )

                print(
                    "[Reminder Agent]"
                    f" Raw arguments: "
                    f"{raw_arguments}"
                )

                # --------------------------------------------
                # Repeated failure detection
                # --------------------------------------------

                signature = (
                    tool_call_signature(
                        function_name,
                        raw_arguments,
                    )
                )

                previous_failures = (
                    identical_failures.get(
                        signature,
                        0,
                    )
                )

                if (
                    previous_failures
                    >= MAX_IDENTICAL_FAILURES
                ):

                    result = {
                        "success": False,
                        "error": (
                            "Identical tool call "
                            "already failed; "
                            "execution stopped "
                            "to prevent repeated "
                            "failure."
                        ),
                    }

                    operation_record = {
                        "tool": function_name,

                        "arguments": raw_arguments,

                        "success": False,

                        "mutating": (
                            function_name
                            in {
                                "create_reminder",
                                "delete_reminder",
                            }
                        ),

                        "error": result[
                            "error"
                        ],

                        "skipped": True,
                    }

                    executed_operations.append(
                        operation_record
                    )

                    last_tool_result = result

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
                # Execute
                # --------------------------------------------

                try:

                    (
                        result,
                        cleaned_arguments,
                    ) = execute_tool(
                        function_name,
                        raw_arguments,
                    )

                    operation_success = (
                        tool_result_succeeded(
                            result
                        )
                    )

                    if not operation_success:

                        identical_failures[
                            signature
                        ] = (
                            previous_failures
                            + 1
                        )

                    operation_record = {
                        "tool": function_name,

                        "arguments": (
                            cleaned_arguments
                        ),

                        "success": (
                            operation_success
                        ),

                        "mutating": (
                            function_name
                            in {
                                "create_reminder",
                                "delete_reminder",
                            }
                        ),

                        "result": result,
                    }

                    executed_operations.append(
                        operation_record
                    )

                    last_tool_result = result

                    print(
                        "[Reminder Agent]"
                        f" Cleaned arguments: "
                        f"{cleaned_arguments}"
                    )

                    print(
                        "[Reminder Agent]"
                        f" Result: {result}"
                    )

                except Exception as exc:

                    identical_failures[
                        signature
                    ] = (
                        previous_failures
                        + 1
                    )

                    print(
                        "[Reminder Agent]"
                        f" Tool error: {exc}"
                    )

                    try:

                        cleaned_arguments = (
                            normalize_arguments(
                                function_name,
                                raw_arguments,
                            )
                        )

                    except Exception:

                        cleaned_arguments = (
                            raw_arguments
                        )

                    result = {
                        "success": False,
                        "error": str(exc),
                    }

                    operation_record = {
                        "tool": function_name,

                        "arguments": (
                            cleaned_arguments
                        ),

                        "success": False,

                        "mutating": (
                            function_name
                            in {
                                "create_reminder",
                                "delete_reminder",
                            }
                        ),

                        "error": str(exc),
                    }

                    executed_operations.append(
                        operation_record
                    )

                    last_tool_result = result

                # --------------------------------------------
                # Send result back to model
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

        # ====================================================
        # MAX STEPS
        # ====================================================

        return self.build_result(
            executed_operations,
            success=False,
            action="max_steps",
            error=(
                "Reminder Agent reached "
                "the maximum number of "
                "reasoning steps."
            ),
            last_result=(
                last_tool_result
            ),
        )


# ============================================================
# SINGLETON
# ============================================================

reminder_agent = ReminderAgent()


# ============================================================
# PUBLIC ZOE INTERFACE
# ============================================================

def run_reminder_agent(
    query: str,
) -> Any:
    """
    Public interface used by ZOE Runtime.

    The agent is stateless.
    """

    return reminder_agent.run(
        user_request=query
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "          ZOE AUTONOMOUS REMINDER AGENT"
    )
    print("=" * 70)
    print()

    print(
        f"Model: {MODEL}"
    )

    print(
        "Groq API: key 1"
    )

    print(
        "Timezone: Asia/Dhaka (UTC+06:00)"
    )

    print(
        f"Current time: "
        f"{_local_now().isoformat(
            timespec='seconds'
        )}"
    )

    print(
        f"Max agent steps: "
        f"{MAX_AGENT_STEPS}"
    )

    print(
        f"Max tool calls: "
        f"{MAX_TOOL_CALLS}"
    )

    print(
        f"Max identical failures: "
        f"{MAX_IDENTICAL_FAILURES}"
    )

    print()

    print(
        "Examples:"
    )

    print(
        "  Remind me to study in 5 minutes"
    )

    print(
        "  Remind me to study tomorrow at 8 PM"
    )

    print(
        "  Remind me to study every Monday at 7 PM"
    )

    print(
        "  Delete all my study reminders"
    )

    print(
        "  Show my reminders"
    )

    print()

    print(
        "Type 'exit' or 'quit' to stop."
    )

    while True:

        try:

            request = input(
                "\nYou > "
            ).strip()

            if not request:
                continue

            if request.lower() in {
                "exit",
                "quit",
            }:

                break

            result = run_reminder_agent(
                query=request
            )

            print()
            print(
                "=" * 70
            )

            print(
                "RESULT:"
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

            print(
                "=" * 70
            )

        except KeyboardInterrupt:

            print(
                "\nExiting..."
            )

            break

        except Exception as exc:

            print(
                "\nERROR: "
                f"{type(exc).__name__}: {exc}"
            )