from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


# ============================================================
# CONFIGURATION
# ============================================================

TIMEZONE = "Asia/Dhaka"

ZOE_TIMEZONE = ZoneInfo(
    TIMEZONE
)

NOTES_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "notes"
)

NOTES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TIME
# ============================================================

def now() -> datetime:
    return datetime.now(
        ZOE_TIMEZONE
    )


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize(
    value: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        value.strip().lower(),
    )


def _slugify(
    value: str,
) -> str:

    value = value.strip().lower()

    value = re.sub(
        r"[^\w\s-]",
        "",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"[\s_-]+",
        "-",
        value,
    )

    value = value.strip("-")

    return value or "untitled"


# ============================================================
# NOTE ID
# ============================================================

def _generate_note_id() -> str:

    return (
        "note_"
        + uuid.uuid4().hex[:12]
    )


# ============================================================
# NOTE PATH
# ============================================================

def _note_path(
    note_id: str,
) -> Path:

    if not re.fullmatch(
        r"note_[a-f0-9]{12}",
        note_id,
    ):
        raise ValueError(
            "Invalid note ID."
        )

    return (
        NOTES_DIR
        / f"{note_id}.md"
    )


# ============================================================
# FRONTMATTER
# ============================================================

def _build_frontmatter(
    *,
    note_id: str,
    title: str,
    created_at: str,
    updated_at: str,
    tags: list[str],
) -> str:

    lines = [
        "---",
        f"id: {note_id}",
        f"title: {title}",
        f"created_at: {created_at}",
        f"updated_at: {updated_at}",
        "tags:",
    ]

    for tag in tags:

        lines.append(
            f"  - {tag}"
        )

    lines.extend(
        [
            "---",
            "",
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# FRONTMATTER PARSER
# ============================================================

def _parse_frontmatter(
    text: str,
) -> tuple[dict[str, Any], str]:

    if not text.startswith(
        "---\n"
    ):

        return {}, text

    end = text.find(
        "\n---",
        4,
    )

    if end == -1:

        return {}, text

    raw = text[
        4:end
    ]

    content = text[
        end + 4:
    ].lstrip("\n")

    metadata: dict[str, Any] = {}

    current_key: str | None = None

    for line in raw.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(
            "- "
        ):

            if current_key == "tags":

                metadata.setdefault(
                    "tags",
                    [],
                ).append(
                    stripped[2:].strip()
                )

            continue

        if ":" not in line:
            continue

        key, value = line.split(
            ":",
            1,
        )

        key = key.strip()
        value = value.strip()

        if value:

            metadata[key] = value
            current_key = key

        else:

            metadata[key] = []
            current_key = key

    return (
        metadata,
        content,
    )


# ============================================================
# READ NOTE FILE
# ============================================================

def _read_note_file(
    path: Path,
) -> dict[str, Any]:

    text = path.read_text(
        encoding="utf-8"
    )

    metadata, content = (
        _parse_frontmatter(text)
    )

    note_id = metadata.get(
        "id"
    )

    title = metadata.get(
        "title"
    )

    if not note_id:

        note_id = path.stem

    if not title:

        title = path.stem

    return {
        "id": note_id,
        "title": title,
        "created_at": metadata.get(
            "created_at"
        ),
        "updated_at": metadata.get(
            "updated_at"
        ),
        "tags": metadata.get(
            "tags",
            [],
        ),
        "content": content,
        "path": str(path),
    }


# ============================================================
# LIST NOTE FILES
# ============================================================

def _note_files() -> list[Path]:

    return sorted(
        NOTES_DIR.glob(
            "note_*.md"
        )
    )


# ============================================================
# SAVE NOTE
# ============================================================

def save_note(
    title: str,
    content: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a new persistent ZOE note.

    The actual content is stored exactly as supplied.
    """

    title = title.strip()

    if not title:

        return {
            "success": False,
            "error": "Note title is required.",
        }

    if not content.strip():

        return {
            "success": False,
            "error": "Note content is required.",
        }

    note_id = _generate_note_id()

    timestamp = now().isoformat(
        timespec="seconds"
    )

    clean_tags: list[str] = []

    for tag in tags or []:

        tag = str(tag).strip()

        if tag and tag not in clean_tags:

            clean_tags.append(
                tag
            )

    frontmatter = _build_frontmatter(
        note_id=note_id,
        title=title,
        created_at=timestamp,
        updated_at=timestamp,
        tags=clean_tags,
    )

    document = (
        frontmatter
        + content.rstrip()
        + "\n"
    )

    path = _note_path(
        note_id
    )

    path.write_text(
        document,
        encoding="utf-8",
    )

    return {
        "success": True,
        "operation": "save",
        "note": {
            "id": note_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "tags": clean_tags,
            "content": content,
            "path": str(path),
        },
    }


# ============================================================
# OPEN NOTE
# ============================================================

def open_note(
    identifier: str,
) -> dict[str, Any]:
    """
    Open a note by exact note ID or title.

    Title matching is case-insensitive.
    """

    identifier = identifier.strip()

    if not identifier:

        return {
            "success": False,
            "error": "Note identifier is required.",
        }

    # --------------------------------------------------------
    # DIRECT ID
    # --------------------------------------------------------

    if re.fullmatch(
        r"note_[a-f0-9]{12}",
        identifier,
        flags=re.IGNORECASE,
    ):

        path = _note_path(
            identifier.lower()
        )

        if not path.exists():

            return {
                "success": False,
                "error":
                    f"No note found with ID '{identifier}'.",
            }

        return {
            "success": True,
            "operation": "open",
            "note": _read_note_file(
                path
            ),
        }

    # --------------------------------------------------------
    # TITLE MATCH
    # --------------------------------------------------------

    normalized_identifier = _normalize(
        identifier
    )

    matches: list[dict[str, Any]] = []

    for path in _note_files():

        try:

            note = _read_note_file(
                path
            )

        except (
            OSError,
            UnicodeDecodeError,
        ):

            continue

        if _normalize(
            str(note["title"])
        ) == normalized_identifier:

            matches.append(
                note
            )

    if len(matches) == 1:

        return {
            "success": True,
            "operation": "open",
            "note": matches[0],
        }

    if len(matches) > 1:

        return {
            "success": False,
            "error":
                "Multiple notes have that title.",
            "matches": [
                {
                    "id": note["id"],
                    "title": note["title"],
                    "updated_at": note["updated_at"],
                }
                for note in matches
            ],
        }

    # --------------------------------------------------------
    # PARTIAL TITLE MATCH
    # --------------------------------------------------------

    partial_matches: list[dict[str, Any]] = []

    for path in _note_files():

        try:

            note = _read_note_file(
                path
            )

        except (
            OSError,
            UnicodeDecodeError,
        ):

            continue

        title = _normalize(
            str(note["title"])
        )

        if (
            normalized_identifier in title
            or title in normalized_identifier
        ):

            partial_matches.append(
                note
            )

    if len(partial_matches) == 1:

        return {
            "success": True,
            "operation": "open",
            "note": partial_matches[0],
        }

    if partial_matches:

        return {
            "success": False,
            "error":
                "Multiple notes match that title.",
            "matches": [
                {
                    "id": note["id"],
                    "title": note["title"],
                    "updated_at": note["updated_at"],
                }
                for note in partial_matches
            ],
        }

    return {
        "success": False,
        "error":
            f"No note found matching '{identifier}'.",
    }


# ============================================================
# LIST NOTES
# ============================================================

def list_notes(
    limit: int = 50,
) -> dict[str, Any]:
    """
    Return the most recently updated notes.
    """

    limit = max(
        1,
        min(
            int(limit),
            200,
        ),
    )

    notes: list[dict[str, Any]] = []

    for path in _note_files():

        try:

            note = _read_note_file(
                path
            )

        except (
            OSError,
            UnicodeDecodeError,
        ):

            continue

        notes.append(
            {
                "id": note["id"],
                "title": note["title"],
                "created_at": note["created_at"],
                "updated_at": note["updated_at"],
                "tags": note["tags"],
            }
        )

    notes.sort(
        key=lambda note: (
            note.get(
                "updated_at"
            )
            or ""
        ),
        reverse=True,
    )

    notes = notes[:limit]

    return {
        "success": True,
        "operation": "list",
        "count": len(notes),
        "notes": notes,
    }


# ============================================================
# SEARCH NOTES
# ============================================================

def search_notes(
    query: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search note titles, tags and content.
    """

    query = query.strip()

    if not query:

        return {
            "success": False,
            "error": "Search query is required.",
        }

    limit = max(
        1,
        min(
            int(limit),
            100,
        ),
    )

    normalized_query = _normalize(
        query
    )

    results: list[dict[str, Any]] = []

    for path in _note_files():

        try:

            note = _read_note_file(
                path
            )

        except (
            OSError,
            UnicodeDecodeError,
        ):

            continue

        title = _normalize(
            str(note["title"])
        )

        tags = " ".join(
            str(tag)
            for tag in note.get(
                "tags",
                [],
            )
        )

        tags = _normalize(
            tags
        )

        content = _normalize(
            str(note.get(
                "content",
                "",
            ))
        )

        score = 0

        if normalized_query in title:
            score += 10

        if normalized_query in tags:
            score += 7

        if normalized_query in content:
            score += 3

        if score == 0:
            continue

        results.append(
            {
                "id": note["id"],
                "title": note["title"],
                "created_at": note["created_at"],
                "updated_at": note["updated_at"],
                "tags": note["tags"],
                "score": score,
            }
        )

    results.sort(
        key=lambda item: (
            item["score"],
            item.get(
                "updated_at"
            ) or "",
        ),
        reverse=True,
    )

    results = results[:limit]

    return {
        "success": True,
        "operation": "search",
        "query": query,
        "count": len(results),
        "notes": results,
    }


# ============================================================
# UPDATE NOTE
# ============================================================

def update_note(
    identifier: str,
    content: str | None = None,
    title: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Update an existing note.
    """

    result = open_note(
        identifier
    )

    if not result.get(
        "success"
    ):

        return result

    note = result["note"]

    new_title = (
        title.strip()
        if title is not None
        else note["title"]
    )

    new_content = (
        content
        if content is not None
        else note["content"]
    )

    new_tags = (
        tags
        if tags is not None
        else note.get(
            "tags",
            [],
        )
    )

    if not new_title:

        return {
            "success": False,
            "error": "Note title cannot be empty.",
        }

    if not str(
        new_content
    ).strip():

        return {
            "success": False,
            "error": "Note content cannot be empty.",
        }

    clean_tags: list[str] = []

    for tag in new_tags:

        tag = str(tag).strip()

        if tag and tag not in clean_tags:

            clean_tags.append(
                tag
            )

    timestamp = now().isoformat(
        timespec="seconds"
    )

    frontmatter = _build_frontmatter(
        note_id=note["id"],
        title=new_title,
        created_at=note["created_at"],
        updated_at=timestamp,
        tags=clean_tags,
    )

    document = (
        frontmatter
        + str(new_content).rstrip()
        + "\n"
    )

    path = _note_path(
        note["id"]
    )

    path.write_text(
        document,
        encoding="utf-8",
    )

    return {
        "success": True,
        "operation": "update",
        "note": _read_note_file(
            path
        ),
    }


# ============================================================
# DELETE NOTE
# ============================================================

def delete_note(
    identifier: str,
) -> dict[str, Any]:
    """
    Permanently delete a note.
    """

    result = open_note(
        identifier
    )

    if not result.get(
        "success"
    ):

        return result

    note = result["note"]

    path = _note_path(
        note["id"]
    )

    if not path.exists():

        return {
            "success": False,
            "error": "Note file no longer exists.",
        }

    path.unlink()

    return {
        "success": True,
        "operation": "delete",
        "deleted": {
            "id": note["id"],
            "title": note["title"],
        },
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import json

    print(
        json.dumps(
            list_notes(),
            indent=2,
            ensure_ascii=False,
        )
    )