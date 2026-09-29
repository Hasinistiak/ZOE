
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.agents.calendar_agent import run_calendar_agent
from backend.agents.football_calendar_agent import run_football_agent
from backend.agents.f1_calendar_agent import run_f1_agent
from backend.agents.mail_agent import run_email_agent
from backend.agents.f1news_agent import run_f1_news_agent
from backend.agents.footballnews_agent import run_football_news_agent
from backend.agents.reminder_agent import run_reminder_agent
from backend.agents.spotify_agent import run_spotify_agent
from backend.agents.weather_agent import run_weather_agent
from backend.agents.news_agent import run_world_news_agent
from backend.agents.project_agent import run_project_creator_agent


# ============================================================
# TYPES
# ============================================================

AgentFunction = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """
    Immutable definition of a ZOE agent.

    The Brain only needs:
        - name
        - description

    Runtime needs:
        - function
    """

    name: str
    description: str
    function: AgentFunction


# ============================================================
# AGENT REGISTRY
# ============================================================

AGENTS: dict[str, AgentSpec] = {

    # ========================================================
    # PERSONAL CALENDAR
    # ========================================================

    "calendar": AgentSpec(
        name="calendar",
        description=(
            "Personal Google Calendar: view, create, update, or delete "
            "personal events and handle scheduling. "
            "Use this for the user's own calendar."
        ),
        function=run_calendar_agent,
    ),

    # ========================================================
    # FOOTBALL CALENDAR
    # ========================================================

    "football_calendar": AgentSpec(
        name="football_calendar",
        description=(
            "Real Madrid schedule calendar: retrieve Real Madrid matches. "
            "Read-only. Cannot create, update, or delete events."
        ),
        function=run_football_agent,
    ),

    # ========================================================
    # FORMULA 1 CALENDAR
    # ========================================================

    "f1_calendar": AgentSpec(
        name="f1_calendar",
        description=(
            "Formula 1 schedule calendar: retrieve F1 races. "
            "Read-only. Cannot create, update, or delete events."
        ),
        function=run_f1_agent,
    ),

    # ========================================================
    # REMINDERS
    # ========================================================

    "reminder": AgentSpec(
        name="reminder",
        description=(
            "Reminders and tasks: create, view, update, or delete "
            "reminders and tasks."
        ),
        function=run_reminder_agent,
    ),

    # ========================================================
    # EMAIL
    # ========================================================

    "email": AgentSpec(
        name="email",
        description=(
            "Gmail: inspect recent or unread emails, including "
            "senders, recipients, subjects, dates, snippets, "
            "and unread status."
        ),
        function=run_email_agent,
    ),

    # ========================================================
    # SPOTIFY
    # ========================================================

    "spotify": AgentSpec(
        name="spotify",
        description=(
            "Spotify and music: search, play, pause, skip, "
            "control playback, and manage music."
        ),
        function=run_spotify_agent,
    ),

    # ========================================================
    # WEATHER
    # ========================================================

    "weather": AgentSpec(
        name="weather",
        description=(
            "Weather and forecasts for the user's configured "
            "locations."
        ),
        function=run_weather_agent,
    ),

    # ========================================================
    # WORLD NEWS
    # ========================================================

    "news": AgentSpec(
        name="news",
        description=(
            "Latest world and international news, including "
            "current events, countries, politics, business, "
            "technology, and major global developments."
        ),
        function=run_world_news_agent,
    ),

    # ========================================================
    # FOOTBALL NEWS
    # ========================================================

    "football_news": AgentSpec(
        name="football_news",
        description=(
            "Latest football news."
        ),
        function=run_football_news_agent,
    ),

    # ========================================================
    # FORMULA 1 NEWS
    # ========================================================

    "f1_news": AgentSpec(
        name="f1_news",
        description=(
            "Latest Formula 1 news."
        ),
        function=run_f1_news_agent,
    ),

    # ========================================================
    # PROJECT CREATOR
    # ========================================================

    "project_creator": AgentSpec(
        name="project_creator",
        description=(
            "Project creation and setup: create new software projects, "
            "initialize project structure, configure development tooling, "
            "and set up Git repositories or GitHub repositories."
        ),
        function=run_project_creator_agent,
    ),
}


# ============================================================
# AGENT DISCOVERY
# ============================================================

def get_agent_descriptions() -> list[dict[str, str]]:
    """
    Return the minimal agent information required by the Brain.

    This intentionally excludes Python functions and implementation
    details to keep the Brain context small.
    """

    return [
        {
            "name": agent.name,
            "description": agent.description,
        }
        for agent in AGENTS.values()
    ]


def get_agent_names() -> tuple[str, ...]:
    """
    Return all registered agent names.
    """

    return tuple(AGENTS.keys())


def has_agent(
    agent_name: str,
) -> bool:
    """
    Return True if an agent is registered.
    """

    return agent_name in AGENTS


# ============================================================
# AGENT LOOKUP
# ============================================================

def get_agent(
    agent_name: str,
) -> AgentSpec:
    """
    Return the complete specification for an agent.
    """

    try:
        return AGENTS[agent_name]

    except KeyError:
        available = ", ".join(AGENTS.keys())

        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Available agents: {available}"
        ) from None


def get_agent_function(
    agent_name: str,
) -> AgentFunction:
    """
    Return the executable function for an agent.
    """

    return get_agent(
        agent_name
    ).function


# ============================================================
# AGENT EXECUTION
# ============================================================

def execute_agent(
    agent_name: str,
    **kwargs: Any,
) -> Any:
    """
    Execute a registered agent.

    Agent-specific arguments are passed through unchanged.
    """

    function = get_agent_function(
        agent_name
    )

    return function(
        **kwargs
    )

