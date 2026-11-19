import os
import requests

from datetime import datetime
from zoneinfo import ZoneInfo


FOOTBALL_DATA_TOKEN = "c33c95360511430987c34b23c9bb78fe"

REAL_MADRID_ID = 86

BASE_URL = "https://api.football-data.org/v4"

BD_TIMEZONE = ZoneInfo(
    "Asia/Dhaka"
)

headers = {
    "X-Auth-Token": FOOTBALL_DATA_TOKEN,
    "Accept": "application/json"
}


def format_bd_time(utc_date: str | None) -> str | None:

    if not utc_date:
        return None

    try:

        dt = datetime.fromisoformat(
            utc_date.replace(
                "Z",
                "+00:00"
            )
        )

        bd_time = dt.astimezone(
            BD_TIMEZONE
        )

        return bd_time.strftime(
            "%d %B %Y, %I:%M %p"
        )

    except Exception as e:

        print(
            "Football time conversion error:",
            e
        )

        return utc_date


def get_next_real_madrid_match():

    response = requests.get(
        f"{BASE_URL}/teams/{REAL_MADRID_ID}/matches",
        headers=headers,
        params={
            "status": "SCHEDULED",
            "limit": 10
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    matches = data.get(
        "matches",
        []
    )

    if not matches:
        return None

    matches.sort(
        key=lambda match:
            match.get(
                "utcDate",
                ""
            )
    )

    match = matches[0]

    home = match.get(
        "homeTeam",
        {}
    )

    away = match.get(
        "awayTeam",
        {}
    )

    competition = match.get(
        "competition",
        {}
    )

    utc_date = match.get(
        "utcDate"
    )

    return {
        "id": match.get("id"),

        "timestamp": format_bd_time(
            utc_date
        ),

        "tournament": competition.get(
            "name"
        ),

        "homeTeam": (
            home.get("shortName")
            or home.get("name")
        ),

        "awayTeam": (
            away.get("shortName")
            or away.get("name")
        )
    }


def get_last_real_madrid_match():

    response = requests.get(
        f"{BASE_URL}/teams/{REAL_MADRID_ID}/matches",
        headers=headers,
        params={
            "status": "FINISHED",
            "limit": 20
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    matches = data.get(
        "matches",
        []
    )

    if not matches:
        return None

    matches.sort(
        key=lambda match:
            match.get(
                "utcDate",
                ""
            ),
        reverse=True
    )

    match = matches[0]

    home = match.get(
        "homeTeam",
        {}
    )

    away = match.get(
        "awayTeam",
        {}
    )

    competition = match.get(
        "competition",
        {}
    )

    score = match.get(
        "score",
        {}
    ).get(
        "fullTime",
        {}
    )

    utc_date = match.get(
        "utcDate"
    )

    return {
        "id": match.get("id"),

        "timestamp": format_bd_time(
            utc_date
        ),

        "tournament": competition.get(
            "name"
        ),

        "homeTeam": (
            home.get("shortName")
            or home.get("name")
        ),

        "awayTeam": (
            away.get("shortName")
            or away.get("name")
        ),

        "homeScore": score.get(
            "home"
        ),

        "awayScore": score.get(
            "away"
        )
    }

if __name__ == "__main__":
    print(get_next_real_madrid_match())
    print(get_last_real_madrid_match())