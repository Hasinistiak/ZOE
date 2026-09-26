from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from backend.functions.mail import check_emails


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

MODEL = os.getenv(
    "ZOE_EMAIL_MODEL",
    "openai/gpt-oss-120b",
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY2",
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY2 is not set."
    )

client = Groq(
    api_key=GROQ_API_KEY,
)

MAX_AGENT_STEPS = int(
    os.getenv(
        "ZOE_EMAIL_MAX_STEPS",
        "6",
    )
)

MAX_TOOL_CALLS = int(
    os.getenv(
        "ZOE_EMAIL_MAX_TOOL_CALLS",
        "10",
    )
)

MAX_EMAIL_RESULTS = int(
    os.getenv(
        "ZOE_EMAIL_MAX_RESULTS",
        "25",
    )
)


# ============================================================
# EMAIL DISPLAY
# ============================================================

def email_summary(
    email: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a safe compact representation of an email.

    Only expose the fields the Email Agent actually needs.
    """

    return {
        "id": email.get("id"),
        "thread_id": email.get("thread_id"),
        "from": email.get("from"),
        "to": email.get("to"),
        "subject": email.get("subject"),
        "date": email.get("date"),
        "snippet": email.get("snippet"),
        "unread": email.get(
            "unread",
            False,
        ),
    }


# ============================================================
# TOOL: CHECK EMAILS
# ============================================================

def tool_check_emails(
    max_results: int = 10,
    unread_only: bool = False,
) -> dict[str, Any]:
    """
    Retrieve recent Gmail emails.

    Args:
        max_results:
            Maximum number of emails to retrieve.

        unread_only:
            If True, retrieve only unread emails.
    """

    # --------------------------------------------------------
    # Validate max_results
    # --------------------------------------------------------

    try:
        max_results = int(
            max_results
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "max_results must be an integer."
        ) from exc

    if max_results < 1:

        raise ValueError(
            "max_results must be at least 1."
        )

    max_results = min(
        max_results,
        MAX_EMAIL_RESULTS,
    )

    # --------------------------------------------------------
    # Normalize unread_only
    # --------------------------------------------------------

    if isinstance(
        unread_only,
        str,
    ):

        unread_only = (
            unread_only.lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

    unread_only = bool(
        unread_only
    )

    # --------------------------------------------------------
    # Gmail
    # --------------------------------------------------------

    result = check_emails(
        max_results=max_results,
        unread_only=unread_only,
    )

    # --------------------------------------------------------
    # Empty result
    # --------------------------------------------------------

    if result is None:

        return {
            "success": True,
            "count": 0,
            "emails": [],
            "unread_only": unread_only,
        }

    # --------------------------------------------------------
    # Unexpected result type
    # --------------------------------------------------------

    if not isinstance(
        result,
        dict,
    ):

        return {
            "success": True,
            "count": 0,
            "emails": [],
            "unread_only": unread_only,
            "raw_result": str(result),
        }

    # --------------------------------------------------------
    # Gmail/API failure
    # --------------------------------------------------------

    if result.get("success") is False:

        return result

    # --------------------------------------------------------
    # Extract emails
    # --------------------------------------------------------

    raw_emails = result.get(
        "emails",
        [],
    )

    if not isinstance(
        raw_emails,
        list,
    ):

        raw_emails = []

    emails = [
        email_summary(email)
        for email in raw_emails
        if isinstance(
            email,
            dict,
        )
    ]

    return {
        "success": True,
        "count": len(emails),
        "emails": emails,
        "unread_only": unread_only,
    }


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOLS = {
    "check_emails": tool_check_emails,
}


# ============================================================
# GROQ TOOLS
# ============================================================

GROQ_TOOLS = [
    {
        "type": "function",

        "function": {
            "name": "check_emails",

            "description": (
                "Retrieve recent Gmail messages. "
                "Use unread_only=true when the user asks "
                "for unread or new emails. Use unread_only=false "
                "for recent or general email requests. "
                "The result contains sender, recipient, subject, "
                "date, snippet, Gmail message ID, thread ID, "
                "and unread status. "
                "This tool is read-only."
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "max_results": {
                        "type": "integer",

                        "description": (
                            "Maximum number of emails to "
                            "retrieve. Use a small number "
                            "such as 5 or 10 unless the "
                            "user explicitly asks for more."
                        ),

                        "minimum": 1,

                        "maximum": MAX_EMAIL_RESULTS,
                    },

                    "unread_only": {
                        "type": "boolean",

                        "description": (
                            "Set true when the user asks "
                            "for unread or new emails. "
                            "Set false for general recent "
                            "email requests."
                        ),
                    },
                },

                "required": [
                    "max_results",
                    "unread_only",
                ],
            },
        },
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's autonomous Email Action Agent.

You are an EXECUTION AGENT, not a conversational assistant.

Your job is to accomplish the user's email-inspection
objective using the available Gmail tool.


============================================================
AVAILABLE CAPABILITY
============================================================

You can:

- inspect recent emails
- inspect unread emails
- see sender
- see recipient
- see subject
- see date
- see snippet
- see Gmail message ID
- see thread ID
- see unread status
- summarize retrieved results

The Gmail capability is READ-ONLY.

You CANNOT:

- send email
- reply to email
- delete email
- archive email
- mark email as read
- mark email as unread
- modify email
- create drafts
- download attachments
- retrieve arbitrary full message bodies


============================================================
CHECKING EMAIL
============================================================

For:

"Check my email."

Use:

check_emails(
    max_results=10,
    unread_only=false
)


For:

"Check my unread emails."

Use:

check_emails(
    max_results=10,
    unread_only=true
)


For:

"Show my latest 5 emails."

Use:

check_emails(
    max_results=5,
    unread_only=false
)


For:

"What's new in my inbox?"

Use:

check_emails(
    max_results=10,
    unread_only=true
)


For:

"Are there any unread emails?"

Use:

check_emails(
    max_results=10,
    unread_only=true
)


============================================================
EMAIL INTERPRETATION
============================================================

After receiving email data:

- identify relevant messages
- preserve sender information
- preserve subject information
- use snippets when useful
- distinguish unread from read
- do not invent message contents
- do not invent sender identities
- do not invent dates
- do not infer unsupported information


============================================================
IMPORTANT EMAILS
============================================================

You may identify emails that appear potentially important
based on their actual metadata and snippet.

Useful signals can include:

- sender
- subject
- snippet
- unread status
- deadline language
- verification language
- payment language
- appointment language
- application language
- security language
- confirmation language
- interview language
- urgent language

Do not claim certainty when the available information
is insufficient.

Use:

"appears important"

or:

"may require your attention"

when appropriate.


============================================================
NEWSLETTERS
============================================================

You may identify obvious newsletters or automated emails
when the sender, subject, or snippet makes that clear.

Do not invent classifications.


============================================================
SEARCH LIMITATION
============================================================

The current tool only exposes recent/unread retrieval.

It does NOT provide arbitrary Gmail search.

Therefore do NOT pretend that you searched for:

- a specific sender
- a specific subject
- a specific keyword
- a specific date

unless the tool actually supports it.

For example:

"Find the email from NASA."

Do not fabricate a search result.

The current tool only retrieves recent or unread messages.


============================================================
FULL EMAIL CONTENT
============================================================

The current tool returns metadata and Gmail snippets.

A snippet is NOT necessarily the complete email.

Never present a snippet as the full email.

If the user asks to read the full email and the required
tool is unavailable, state that only the available preview
was retrieved.


============================================================
NO CHAT HISTORY
============================================================

You are stateless.

You do not receive previous conversation history.

Do not assume that:

"that email"
"the previous email"
"the one I mentioned earlier"

refers to an earlier conversation.

Use only the current request and current Gmail results.


============================================================
ERROR HANDLING
============================================================

If Gmail authentication fails:

→ report that Gmail access failed.

If Gmail returns an API error:

→ report the actual failure.

If zero emails are returned:

→ report that no emails were returned.

Never claim that the entire mailbox is empty unless the
tool actually establishes that.


============================================================
EXECUTION
============================================================

When the user's request is clear:

ACT.

Do not unnecessarily ask for confirmation.

Do not repeatedly call the same tool when one call already
provides enough information.


============================================================
OUTPUT
============================================================

You are NOT the conversational layer.

Do not produce filler such as:

"Sure."
"Of course."
"Let me check."
"I'll take a look."

Execute the objective and return the result.

The Python runtime handles the final conversational response.
"""


# ============================================================
# BUILD MESSAGES
# ============================================================

def build_messages(
    user_request: str,
) -> list[dict[str, Any]]:
    """
    Build the stateless LLM conversation.
    """

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_request.strip(),
        },
    ]


# ============================================================
# ARGUMENT NORMALIZATION
# ============================================================

def normalize_arguments(
    name: str,
    arguments: Any,
) -> dict[str, Any]:
    """
    Parse and validate model-generated tool arguments.

    Unknown arguments are rejected rather than silently
    discarded.
    """

    # --------------------------------------------------------
    # Validate tool
    # --------------------------------------------------------

    if name not in TOOLS:

        raise ValueError(
            f"Unknown email tool: {name!r}"
        )

    # --------------------------------------------------------
    # Parse JSON string
    # --------------------------------------------------------

    if arguments is None:

        arguments = {}

    if isinstance(
        arguments,
        str,
    ):

        arguments = arguments.strip()

        if not arguments:

            arguments = {}

        else:

            try:

                arguments = json.loads(
                    arguments
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON arguments "
                    f"for tool {name!r}."
                ) from exc

    # --------------------------------------------------------
    # Must be object
    # --------------------------------------------------------

    if not isinstance(
        arguments,
        dict,
    ):

        raise ValueError(
            f"Arguments for tool {name!r} "
            "must be a JSON object."
        )

    # --------------------------------------------------------
    # Allowed arguments
    # --------------------------------------------------------

    allowed_arguments = {
        "check_emails": {
            "max_results",
            "unread_only",
        },
    }

    allowed = allowed_arguments[name]

    unknown = [
        key
        for key in arguments
        if key not in allowed
    ]

    if unknown:

        raise ValueError(
            f"Unknown argument(s) for "
            f"{name!r}: "
            f"{', '.join(map(str, unknown))}"
        )

    cleaned = dict(
        arguments
    )

    # --------------------------------------------------------
    # Required arguments
    # --------------------------------------------------------

    required_arguments = {
        "check_emails": {
            "max_results",
            "unread_only",
        },
    }

    missing = [
        key
        for key in required_arguments[name]
        if (
            key not in cleaned
            or cleaned[key] is None
        )
    ]

    if missing:

        raise ValueError(
            f"Missing required argument(s) "
            f"for {name!r}: "
            f"{', '.join(missing)}"
        )

    # --------------------------------------------------------
    # max_results
    # --------------------------------------------------------

    try:

        cleaned["max_results"] = int(
            cleaned["max_results"]
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "max_results must be an integer."
        ) from exc

    if cleaned["max_results"] < 1:

        raise ValueError(
            "max_results must be at least 1."
        )

    cleaned["max_results"] = min(
        cleaned["max_results"],
        MAX_EMAIL_RESULTS,
    )

    # --------------------------------------------------------
    # unread_only
    # --------------------------------------------------------

    unread_only = cleaned[
        "unread_only"
    ]

    if isinstance(
        unread_only,
        str,
    ):

        unread_only = (
            unread_only.lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

    cleaned["unread_only"] = bool(
        unread_only
    )

    return cleaned


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    name: str,
    arguments: Any,
) -> Any:
    """
    Execute one registered email tool.
    """

    if name not in TOOLS:

        raise ValueError(
            f"Unknown email tool: {name!r}"
        )

    cleaned_arguments = (
        normalize_arguments(
            name,
            arguments,
        )
    )

    return TOOLS[name](
        **cleaned_arguments
    )


# ============================================================
# RESULT SUCCESS
# ============================================================

def result_is_successful(
    result: Any,
) -> bool:
    """
    Determine whether a tool explicitly reported failure.
    """

    if isinstance(
        result,
        dict,
    ):

        return (
            result.get("success")
            is not False
        )

    return True


# ============================================================
# EMAIL AGENT
# ============================================================

class EmailAgent:
    """
    Autonomous Gmail Email Agent.

    Current capabilities:

    - recent email inspection
    - unread email inspection
    - email metadata extraction
    - bounded execution
    - stateless operation
    - safe error handling
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
        user_request: str,
    ) -> Any:
        """
        Execute the complete email-inspection objective.
        """

        # ----------------------------------------------------
        # Validate request
        # ----------------------------------------------------

        if (
            not user_request
            or not user_request.strip()
        ):

            return {
                "success": False,
                "action": None,
                "error": (
                    "Email request "
                    "cannot be empty."
                ),
            }

        messages = build_messages(
            user_request=user_request,
        )

        total_tool_calls = 0

        executed_operations: list[
            dict[str, Any]
        ] = []

        # ====================================================
        # AGENT LOOP
        # ====================================================

        for step in range(
            1,
            MAX_AGENT_STEPS + 1,
        ):

            print()
            print(
                "[Email Agent]"
                f" Step {step}/{MAX_AGENT_STEPS}"
            )

            # ------------------------------------------------
            # LLM REQUEST
            # ------------------------------------------------

            try:

                response = (
                    client.chat.completions.create(

                        model=self.model,

                        messages=messages,

                        tools=GROQ_TOOLS,

                        tool_choice="auto",

                        temperature=0,

                        max_completion_tokens=1600,
                    )
                )

            except Exception as exc:

                return {
                    "success": False,
                    "action": None,
                    "error": (
                        "Groq request failed: "
                        f"{exc}"
                    ),
                    "operations":
                        executed_operations,
                }

            # ------------------------------------------------
            # Validate response
            # ------------------------------------------------

            if (
                not response
                or not getattr(
                    response,
                    "choices",
                    None,
                )
            ):

                return {
                    "success": False,
                    "action": None,
                    "error": (
                        "Email Agent received "
                        "an invalid model response."
                    ),
                    "operations":
                        executed_operations,
                }

            message = (
                response.choices[0].message
            )

            tool_calls = (
                getattr(
                    message,
                    "tool_calls",
                    None,
                )
                or []
            )

            # ------------------------------------------------
            # No tool calls
            # ------------------------------------------------

            if not tool_calls:

                if executed_operations:

                    return {
                        "success": all(
                            operation.get(
                                "success",
                                False,
                            )
                            for operation
                            in executed_operations
                        ),

                        "action": (
                            "single_step"
                            if len(
                                executed_operations
                            ) == 1
                            else "multi_step"
                        ),

                        "operations":
                            executed_operations,
                    }

                return {
                    "success": False,
                    "action": None,
                    "error": (
                        "Email Agent did not "
                        "produce a tool call."
                    ),
                    "model_response": (
                        (
                            message.content
                            or ""
                        ).strip()
                    ),
                }

            # ------------------------------------------------
            # Append assistant tool-call message
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

                function = (
                    tool_call.function
                )

                assistant_message[
                    "tool_calls"
                ].append(
                    {
                        "id":
                            tool_call.id,

                        "type":
                            "function",

                        "function": {
                            "name":
                                function.name,

                            "arguments":
                                function.arguments,
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

                    return {
                        "success": False,

                        "action":
                            "tool_limit",

                        "error": (
                            "Email Agent reached "
                            "the maximum number of "
                            "tool executions."
                        ),

                        "operations":
                            executed_operations,
                    }

                total_tool_calls += 1

                function_name = (
                    tool_call.function.name
                )

                raw_arguments = (
                    tool_call.function.arguments
                )

                print(
                    "[Email Agent]"
                    f" Tool #{total_tool_calls}: "
                    f"{function_name}"
                )

                print(
                    "[Email Agent]"
                    f" Arguments: "
                    f"{raw_arguments}"
                )

                # --------------------------------------------
                # Execute
                # --------------------------------------------

                try:

                    result = execute_tool(
                        function_name,
                        raw_arguments,
                    )

                    successful = (
                        result_is_successful(
                            result
                        )
                    )

                    operation = {
                        "tool":
                            function_name,

                        "arguments":
                            normalize_arguments(
                                function_name,
                                raw_arguments,
                            ),

                        "success":
                            successful,

                        "result":
                            result,
                    }

                    executed_operations.append(
                        operation
                    )

                    print(
                        "[Email Agent]"
                        f" Result: {result}"
                    )

                except Exception as exc:

                    print(
                        "[Email Agent]"
                        f" Tool error: {exc}"
                    )

                    result = {
                        "success": False,
                        "error": str(exc),
                    }

                    executed_operations.append(
                        {
                            "tool":
                                function_name,

                            "arguments":
                                raw_arguments,

                            "success":
                                False,

                            "error":
                                str(exc),
                        }
                    )

                # --------------------------------------------
                # Send tool result back to model
                # --------------------------------------------

                try:

                    serialized_result = json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )

                except Exception:

                    serialized_result = str(
                        result
                    )

                messages.append(
                    {
                        "role": "tool",

                        "tool_call_id":
                            tool_call.id,

                        "content":
                            serialized_result,
                    }
                )

        # ====================================================
        # MAX STEPS
        # ====================================================

        return {
            "success": False,

            "action":
                "max_steps",

            "error": (
                "Email Agent reached "
                "the maximum number of "
                "reasoning steps."
            ),

            "operations":
                executed_operations,
        }


# ============================================================
# SINGLETON
# ============================================================

email_agent = EmailAgent()


# ============================================================
# PUBLIC ZOE INTERFACE
# ============================================================

def run_email_agent(
    query: str,
) -> Any:
    """
    Public interface used by ZOE Runtime.
    """

    return email_agent.run(
        user_request=query,
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("                ZOE AUTONOMOUS EMAIL AGENT")
    print("=" * 70)
    print()

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Max agent steps: {MAX_AGENT_STEPS}"
    )

    print(
        f"Max tool calls: {MAX_TOOL_CALLS}"
    )

    print(
        f"Max emails per call: {MAX_EMAIL_RESULTS}"
    )

    print()

    print("Try:")

    print(
        "  Check my email"
    )

    print(
        "  Check my unread emails"
    )

    print(
        "  Show me my latest 5 emails"
    )

    print(
        "  What's new in my inbox?"
    )

    print(
        "  Are there any unread emails?"
    )

    print(
        "  Give me a summary of my recent emails"
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

            result = run_email_agent(
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