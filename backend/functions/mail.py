from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


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

def _get_gmail_service():
    """
    Authenticate with Gmail and return an authenticated API service.

    First run:
        Opens Google's OAuth consent page.

    Later runs:
        Reuses the saved refresh token.
    """

    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    creds: Credentials | None = None

    # --------------------------------------------------------
    # Load existing token
    # --------------------------------------------------------

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            SCOPES,
        )

    # --------------------------------------------------------
    # Refresh expired credentials
    # --------------------------------------------------------

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    # --------------------------------------------------------
    # First-time authentication
    # --------------------------------------------------------

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

        TOKEN_FILE.write_text(
            creds.to_json(),
            encoding="utf-8",
        )

    # --------------------------------------------------------
    # Build Gmail service
    # --------------------------------------------------------

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
# CHECK EMAILS
# ============================================================

def check_emails(
    max_results: int = 10,
    unread_only: bool = False,
) -> dict[str, Any]:
    """
    Check recent Gmail messages.

    Args:
        max_results:
            Maximum number of emails to return.

        unread_only:
            If True, only return unread emails.

    Returns:
        Structured dictionary containing email information.
    """

    try:
        service = _get_gmail_service()

        # ----------------------------------------------------
        # Build Gmail query
        # ----------------------------------------------------

        query = "is:unread" if unread_only else ""

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

        messages = response.get("messages", [])

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

            payload = email.get("payload", {})

            headers = payload.get("headers", [])

            label_ids = email.get("labelIds", [])

            emails.append(
                {
                    "id": message_id,
                    "thread_id": email.get("threadId"),
                    "from": _get_header(headers, "From"),
                    "to": _get_header(headers, "To"),
                    "subject": _get_header(headers, "Subject"),
                    "date": _get_header(headers, "Date"),
                    "snippet": email.get("snippet", ""),
                    "unread": "UNREAD" in label_ids,
                    "labels": label_ids,
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

    result = check_emails(
        max_results=10,
        unread_only=False,
    )

    print("\n" + "=" * 70)
    print("ZOE EMAIL CHECK")
    print("=" * 70)

    if not result["success"]:
        print(f"\nERROR: {result['error']}")
        raise SystemExit(1)

    print(f"\nEmails found: {result['count']}\n")

    for email in result["emails"]:

        status = "UNREAD" if email["unread"] else "READ"

        print("-" * 70)
        print(f"Status : {status}")
        print(f"From   : {email['from']}")
        print(f"Subject: {email['subject']}")
        print(f"Date   : {email['date']}")
        print(f"Snippet: {email['snippet']}")
        print(f"ID     : {email['id']}")