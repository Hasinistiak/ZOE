
from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from backend.functions.notes import (
    delete_note,
    list_notes,
    open_note,
    save_note,
    search_notes,
    update_note,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = os.getenv(
    "ZOE_NOTES_MODEL",
    "openai/gpt-oss-120b",
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY3"
)

MAX_COMPLETION_TOKENS = 300

VALID_OPERATIONS = {
    "save",
    "open",
    "list",
    "search",
    "update",
    "delete",
}


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY3 is not set."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's persistent notes operation parser.

Your ONLY job is to convert the user's request into EXACTLY
ONE notes operation.

You MUST return ONLY valid JSON.

Do not answer the user.
Do not explain anything.
Do not include markdown.
Do not include code fences.
Do not invent note IDs.
Do not invent note contents.
Do not perform the operation yourself.

Available operations:

SAVE
OPEN
LIST
SEARCH
UPDATE
DELETE


============================================================
SAVE
============================================================

Use SAVE when the user wants to create, write, store, save,
record, or make a new note.

Return:

{
  "operation": "save",
  "title": "...",
  "tags": ["..."]
}

The actual note content is supplied separately by the caller.

DO NOT put note content in the JSON.

If the user explicitly gives a title, preserve that title.

If the user does not give a title, create a short descriptive
title based on the subject of the note.

Do not make the title unnecessarily long.

Examples:

"Save a note titled Linux Boot Process"
→
{
  "operation": "save",
  "title": "Linux Boot Process",
  "tags": ["linux", "boot"]
}

"Make a note about operating systems"
→
{
  "operation": "save",
  "title": "Operating Systems",
  "tags": ["operating-systems"]
}


============================================================
OPEN
============================================================

Use OPEN when the user wants to read, show, open, display,
retrieve, access, or bring up ONE specific existing note.

Return:

{
  "operation": "open",
  "identifier": "..."
}

The identifier may be:
- a note ID
- an exact title
- a distinctive title fragment

Preserve the user's requested identifier.

Examples:

"Open my Operating Systems note"
→
{
  "operation": "open",
  "identifier": "Operating Systems"
}

"Read note_12345"
→
{
  "operation": "open",
  "identifier": "note_12345"
}


============================================================
LIST
============================================================

Use LIST when the user wants to see the collection of notes.

Examples:

"Show my notes"
"What notes do I have?"
"List my notes"
"What have I saved?"
"Show all my notes"
"What are my previous notes?"

Return:

{
  "operation": "list"
}


============================================================
SEARCH
============================================================

Use SEARCH when the user wants to FIND notes based on a
topic, subject, phrase, keyword, or concept.

Use SEARCH when the user does NOT clearly identify one
specific note.

Return:

{
  "operation": "search",
  "query": "..."
}

Examples:

"Find my notes about operating systems"
→
{
  "operation": "search",
  "query": "operating systems"
}

"Do I have anything about Linux memory management?"
→
{
  "operation": "search",
  "query": "Linux memory management"
}

"Find notes mentioning BIOS"
→
{
  "operation": "search",
  "query": "BIOS"
}


============================================================
OPEN VS SEARCH
============================================================

This distinction is important.

If the user refers to ONE specific note:
→ OPEN

If the user wants to discover notes matching a subject:
→ SEARCH

Examples:

"Open my operating systems note"
→ OPEN

"Find my notes about operating systems"
→ SEARCH

"Show me the operating systems note"
→ OPEN

"Do I have any notes about operating systems?"
→ SEARCH


============================================================
UPDATE
============================================================

Use UPDATE when the user explicitly wants to modify an
existing note.

This includes:
- edit
- update
- modify
- rewrite
- change
- replace
- append
- add to

Return:

{
  "operation": "update",
  "identifier": "...",
  "title": null,
  "tags": null
}

The caller supplies the updated content separately.

IMPORTANT:

Only provide "title" if the user explicitly asks to change
the title.

Only provide "tags" if the user explicitly asks to change
the tags.

Otherwise use null.

The identifier must come from the user's request.

Never invent an identifier.

Examples:

"Update my Operating Systems note"
→
{
  "operation": "update",
  "identifier": "Operating Systems",
  "title": null,
  "tags": null
}

"Rename my Linux note to Linux Internals"
→
{
  "operation": "update",
  "identifier": "Linux",
  "title": "Linux Internals",
  "tags": null
}


============================================================
DELETE
============================================================

Use DELETE when the user explicitly wants to delete, remove,
erase, discard, or permanently delete an existing note.

Return:

{
  "operation": "delete",
  "identifier": "..."
}

The identifier must come from the user's request.

Never invent an identifier.

Examples:

"Delete my Linux note"
→
{
  "operation": "delete",
  "identifier": "Linux"
}

"Remove note_12345"
→
{
  "operation": "delete",
  "identifier": "note_12345"
}


============================================================
AMBIGUOUS REQUESTS
============================================================

Do not invent missing information.

If the request does not identify a note for OPEN, UPDATE,
or DELETE, return the operation with an empty identifier.

The caller will handle the missing identifier.

For SAVE, a title may be generated if one is not explicitly
provided.

For SEARCH, extract the actual subject being searched for.


============================================================
GENERAL RULES
============================================================

- Return exactly ONE operation.
- Return ONLY JSON.
- Never answer the user.
- Never invent note IDs.
- Never invent note content.
- Never fabricate identifiers.
- Never perform filesystem operations.
- Never include additional keys unless specified above.
- Preserve user-provided identifiers.
- Keep generated titles short and descriptive.
- Keep tags concise and relevant.
- Avoid duplicate tags.
- Tags should normally be lowercase.
- Do not create excessive tags.
"""


# ============================================================
# LLM PARSER
# ============================================================

def _run_llm(
    query: str,
    model: str,
) -> dict[str, Any]:
    """
    Parse a user request into one structured notes operation.

    This function ONLY performs LLM parsing.
    It does not execute any notes operation.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        response_format={
            "type": "json_object"
        },
    )

    if not response.choices:
        return {}

    message = response.choices[0].message

    content = (
        message.content
        if message is not None
        else None
    )

    if not content:
        return {}

    content = content.strip()

    if not content:
        return {}

    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(result, dict):
        return {}

    return result


# ============================================================
# VALUE HELPERS
# ============================================================

def _string_or_none(
    value: Any,
) -> str | None:
    """
    Convert a value to a clean string.

    Empty strings become None.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (dict, list, tuple, set)):
        return None

    value = str(value).strip()

    return value or None


def _required_string(
    value: Any,
) -> str | None:
    """
    Extract a required string value.

    Currently equivalent to _string_or_none but kept separate
    to make intent explicit at call sites.
    """

    return _string_or_none(value)


def _tags_or_none(
    value: Any,
) -> list[str] | None:
    """
    Normalize a tag list.

    - Rejects non-lists.
    - Removes empty tags.
    - Removes duplicates.
    - Normalizes whitespace.
    - Limits pathological tag counts.
    """

    if value is None:
        return None

    if not isinstance(value, list):
        return None

    tags: list[str] = []

    for raw_tag in value:
        if not isinstance(raw_tag, str):
            continue

        tag = raw_tag.strip().lower()

        if not tag:
            continue

        if tag not in tags:
            tags.append(tag)

        if len(tags) >= 10:
            break

    return tags


def _normalize_operation(
    value: Any,
) -> str | None:
    """
    Normalize and validate the operation name.
    """

    if not isinstance(value, str):
        return None

    operation = value.strip().lower()

    if operation not in VALID_OPERATIONS:
        return None

    return operation


# ============================================================
# RESULT HELPERS
# ============================================================

def _success_result(
    result: Any,
) -> dict[str, Any]:
    """
    Normalize a successful notes-function result and attach
    the agent identifier.
    """

    if isinstance(result, dict):
        result["agent"] = "notes"
        return result

    return {
        "success": True,
        "agent": "notes",
        "result": result,
    }


def _error(
    message: str,
) -> dict[str, Any]:
    """
    Standard notes-agent error response.
    """

    return {
        "success": False,
        "agent": "notes",
        "error": message,
    }


# ============================================================
# AGENT
# ============================================================

class NotesAgent:
    """
    ZOE notes operation agent.

    Responsibilities:
        1. Parse the user's notes request with the LLM.
        2. Validate the parsed operation.
        3. Execute exactly one notes operation.

    The agent does NOT generate note content.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
    ) -> None:

        model = str(model).strip()

        if not model:
            raise ValueError(
                "Notes agent model cannot be empty."
            )

        self.model = model

    def run(
        self,
        user_request: str,
        content: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute one notes operation.

        Parameters
        ----------
        user_request:
            Natural-language request describing the operation.

        content:
            Content supplied separately by the caller.

            For SAVE:
                This is the note content.

            For UPDATE:
                This is the replacement/updated content.

            For OPEN/LIST/SEARCH/DELETE:
                Normally None.
        """

        if not isinstance(user_request, str):
            return _error(
                "Notes request must be a string."
            )

        query = user_request.strip()

        if not query:
            return _error(
                "Empty notes request."
            )

        # ----------------------------------------------------
        # Normalize supplied content
        # ----------------------------------------------------

        normalized_content: str | None

        if content is None:
            normalized_content = None

        elif isinstance(content, str):
            normalized_content = content.strip()

        else:
            return _error(
                "Note content must be a string."
            )

        # ----------------------------------------------------
        # PARSE REQUEST
        # ----------------------------------------------------

        try:
            parsed = _run_llm(
                query=query,
                model=self.model,
            )

        except Exception as exc:
            return _error(
                f"Notes parser failed: {exc}"
            )

        if not parsed:
            return _error(
                "The notes parser returned no valid operation."
            )

        operation = _normalize_operation(
            parsed.get("operation")
        )

        if operation is None:
            return _error(
                "Unsupported or missing notes operation."
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if operation == "save":

            title = _string_or_none(
                parsed.get("title")
            )

            if not title:
                title = "ZOE Note"

            # Content MUST come from the caller.
            #
            # This prevents a request such as:
            #
            # "Save a note about Linux"
            #
            # from accidentally saving the user's instruction
            # itself as the note.
            if normalized_content is None:
                return _error(
                    "Note content was not supplied."
                )

            if not normalized_content:
                return _error(
                    "Note content cannot be empty."
                )

            tags = (
                _tags_or_none(
                    parsed.get("tags")
                )
                or []
            )

            try:
                result = save_note(
                    title=title,
                    content=normalized_content,
                    tags=tags,
                )

            except Exception as exc:
                return _error(
                    f"Failed to save note: {exc}"
                )

            return _success_result(result)

        # ----------------------------------------------------
        # OPEN
        # ----------------------------------------------------

        if operation == "open":

            identifier = _required_string(
                parsed.get("identifier")
            )

            if not identifier:
                return _error(
                    "No note was specified."
                )

            try:
                result = open_note(
                    identifier
                )

            except Exception as exc:
                return _error(
                    f"Failed to open note: {exc}"
                )

            return _success_result(result)

        # ----------------------------------------------------
        # LIST
        # ----------------------------------------------------

        if operation == "list":

            try:
                result = list_notes()

            except Exception as exc:
                return _error(
                    f"Failed to list notes: {exc}"
                )

            return _success_result(result)

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        if operation == "search":

            search_query = _required_string(
                parsed.get("query")
            )

            if not search_query:
                return _error(
                    "No search query was specified."
                )

            try:
                result = search_notes(
                    search_query
                )

            except Exception as exc:
                return _error(
                    f"Failed to search notes: {exc}"
                )

            return _success_result(result)

        # ----------------------------------------------------
        # UPDATE
        # ----------------------------------------------------

        if operation == "update":

            identifier = _required_string(
                parsed.get("identifier")
            )

            if not identifier:
                return _error(
                    "No note was specified for update."
                )

            if normalized_content is None:
                return _error(
                    "Updated note content was not supplied."
                )

            if not normalized_content:
                return _error(
                    "Updated note content cannot be empty."
                )

            title = _string_or_none(
                parsed.get("title")
            )

            tags = _tags_or_none(
                parsed.get("tags")
            )

            try:
                result = update_note(
                    identifier=identifier,
                    content=normalized_content,
                    title=title,
                    tags=tags,
                )

            except Exception as exc:
                return _error(
                    f"Failed to update note: {exc}"
                )

            return _success_result(result)

        # ----------------------------------------------------
        # DELETE
        # ----------------------------------------------------

        if operation == "delete":

            identifier = _required_string(
                parsed.get("identifier")
            )

            if not identifier:
                return _error(
                    "No note was specified for deletion."
                )

            try:
                result = delete_note(
                    identifier
                )

            except Exception as exc:
                return _error(
                    f"Failed to delete note: {exc}"
                )

            return _success_result(result)

        # ----------------------------------------------------
        # SHOULD NEVER REACH HERE
        # ----------------------------------------------------

        return _error(
            "Unsupported notes operation."
        )


# ============================================================
# PUBLIC API
# ============================================================

notes_agent = NotesAgent()


def run_notes_agent(
    query: str,
    content: str | None = None,
) -> dict[str, Any]:
    """
    Public entry point for ZOE's notes agent.
    """

    return notes_agent.run(
        user_request=query,
        content=content,
    )


# ============================================================
# CLI
# ============================================================

def _run_cli() -> None:
    """
    Simple manual testing interface.

    NOTE:
    The CLI does not have a separate content pipeline, so SAVE
    and UPDATE requests should be tested through the actual
    ZOE caller when content is required.
    """

    print(
        f"ZOE Notes Agent"
        f"\nModel: {notes_agent.model}"
        f"\nType 'exit' or 'quit' to stop."
    )

    while True:

        try:
            query = input(
                "\nNotes > "
            ).strip()

        except EOFError:
            break

        except KeyboardInterrupt:
            print()
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        try:

            result = run_notes_agent(
                query
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        except Exception as exc:

            print(
                json.dumps(
                    _error(str(exc)),
                    indent=2,
                    ensure_ascii=False,
                )
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    _run_cli()
