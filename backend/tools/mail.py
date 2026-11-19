
from __future__ import annotations

from pathlib import Path
from typing import Any
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import json

from google.auth.exceptions import RefreshError

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CREDENTIALS_FILE = BASE_DIR / "backend" / "credentials.json"
TOKEN_FILE = BASE_DIR / "backend" / "gmail_token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]


# ============================================================
# GMAIL AUTHENTICATION
# ============================================================

def _delete_token() -> None:
    """Remove the stored token so a fresh login can happen."""
    try:
        TOKEN_FILE.unlink()
        print(f"[ZOE GMAIL] Deleted invalid token: {TOKEN_FILE}")
    except FileNotFoundError:
        pass


def _save_token(creds: Credentials) -> None:
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")


def _get_gmail_service():
    """
    Authenticate with Gmail and return an authenticated API service.

    If the stored token is corrupt, expired without a refresh token,
    or can no longer be refreshed (revoked/expired refresh token),
    it is deleted and a fresh authentication flow is started.
    """

    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    creds: Credentials | None = None

    # --- Load stored token ---
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                str(TOKEN_FILE),
                SCOPES,
            )
        except (ValueError, KeyError, json.JSONDecodeError):
            _delete_token()
            creds = None

    # --- Validate / refresh ---
    if creds and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                _save_token(creds)  # persist the refreshed token
            except RefreshError as exc:
                print(f"[ZOE GMAIL] Token refresh failed: {exc}")
                _delete_token()
                creds = None
        else:
            _delete_token()
            creds = None

    # --- Re-authenticate if needed ---
    if not creds or not creds.valid:
        if not CREDENTIALS_FILE.exists():
            raise FileNotFoundError(
                f"Gmail credentials file not found: {CREDENTIALS_FILE}"
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            SCOPES,
        )

        creds = flow.run_local_server(
            port=0,
            access_type="offline",
            prompt="consent",
        )
        _save_token(creds)

    return build(
        "gmail",
        "v1",
        credentials=creds,
        cache_discovery=False,
    )

# ============================================================
# EMAIL HEADER HELPER
# ============================================================

def _get_header(
    headers: list[dict[str, str]],
    name: str,
) -> str:
    """
    Extract a Gmail message header.
    """

    name = name.lower()

    for header in headers:
        if header.get("name", "").lower() == name:
            return header.get("value", "")

    return ""


# ============================================================
# CHECK PRIMARY EMAILS
# ============================================================

def check_emails(
    max_results: int = 25,
    unread_only: bool = True,
    since_seconds: int | None = None,
) -> dict[str, Any]:
 

    try:
        service = _get_gmail_service()

        # ----------------------------------------------------
        # Build Gmail search query
        # ----------------------------------------------------

        query_parts = [
            "category:primary",
        ]

        if unread_only:
            query_parts.append(
                "is:unread"
            )

        if since_seconds is not None:
            since_seconds = max(
                1,
                int(since_seconds),
            )

            cutoff = int(time.time()) - since_seconds

            query_parts.append(
                f"after:{cutoff}"
            )

        query = " ".join(query_parts)

        # ----------------------------------------------------
        # Search Gmail
        # ----------------------------------------------------

        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                maxResults=max_results,
                q=query,
            )
            .execute()
        )

        messages = response.get(
            "messages",
            [],
        )

        emails: list[dict[str, Any]] = []

        # ----------------------------------------------------
        # Fetch individual messages
        # ----------------------------------------------------

        for message in messages:

            message_id = message.get("id")

            if not message_id:
                continue

            email = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="metadata",
                    metadataHeaders=[
                        "From",
                        "To",
                        "Subject",
                        "Date",
                    ],
                )
                .execute()
            )

            payload = email.get(
                "payload",
                {},
            )

            headers = payload.get(
                "headers",
                [],
            )

            label_ids = email.get(
                "labelIds",
                [],
            )

            # ------------------------------------------------
            # Extra protection for exact time filtering
            # ------------------------------------------------

            if since_seconds is not None:

                internal_date = email.get(
                    "internalDate"
                )

                if internal_date:

                    message_time = int(
                        internal_date
                    ) / 1000

                    cutoff_time = (
                        time.time()
                        - since_seconds
                    )

                    if message_time < cutoff_time:
                        continue

            # ------------------------------------------------
            # Build normalized email object
            # ------------------------------------------------

            emails.append(
                {
                    "id": message_id,

                    "thread_id": email.get(
                        "threadId"
                    ),

                    "from": _get_header(
                        headers,
                        "From",
                    ),

                    "to": _get_header(
                        headers,
                        "To",
                    ),

                    "subject": _get_header(
                        headers,
                        "Subject",
                    ),

                    "date": _get_header(
                        headers,
                        "Date",
                    ),

                    "snippet": email.get(
                        "snippet",
                        "",
                    ),

                    "unread": (
                        "UNREAD"
                        in label_ids
                    ),

                    "labels": label_ids,

                    "internal_date": email.get(
                        "internalDate"
                    ),
                }
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return {
            "success": True,
            "count": len(emails),
            "emails": emails,
        }

    except Exception as exc:

        return {
            "success": False,
            "count": 0,
            "emails": [],
            "error": str(exc),
        }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    # Default behavior:
    # No time limit + unread only.
    result = check_emails(
        max_results=25,
    )

    print("\n" + "=" * 70)
    print("ZOE PRIMARY EMAIL CHECK")
    print("=" * 70)

    print("\nMode: UNREAD ONLY")
    print("Time limit: NONE")

    if not result["success"]:

        print(
            f"\nERROR: {result['error']}"
        )

        raise SystemExit(1)

    print(
        f"\nUnread emails found: "
        f"{result['count']}\n"
    )

    for email in result["emails"]:

        status = (
            "UNREAD"
            if email["unread"]
            else "READ"
        )

        print("-" * 70)

        print(f"Status : {status}")
        print(f"From   : {email['from']}")
        print(f"To     : {email['to']}")
        print(f"Subject: {email['subject']}")
        print(f"Date   : {email['date']}")
        print(f"Snippet: {email['snippet']}")
        print(f"ID     : {email['id']}")
        print(f"Thread : {email['thread_id']}")
