from __future__ import annotations

import requests
from datetime import datetime
from typing import Any


BASE_URL = "https://wttr.in"


# ============================================================
# HELPERS
# ============================================================

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _weather_description(data: dict) -> str:
    try:
        return data["weatherDesc"][0]["value"]
    except (KeyError, IndexError, TypeError):
        return "Unknown"


def _parse_hour(hour: dict) -> dict:
    """
    Normalize an hourly wttr.in forecast block.
    """

    return {
        "time": _safe_int(hour.get("time")),
        "temperature_c": _safe_float(hour.get("tempC")),
        "feels_like_c": _safe_float(hour.get("FeelsLikeC")),
        "condition": _weather_description(hour),

        "chance_of_rain": _safe_int(hour.get("chanceofrain")),
        "chance_of_snow": _safe_int(hour.get("chanceofsnow")),
        "chance_of_thunder": _safe_int(hour.get("chanceofthunder")),
        "chance_of_fog": _safe_int(hour.get("chanceoffog")),
        "chance_of_frost": _safe_int(hour.get("chanceoffrost")),

        "precipitation_mm": _safe_float(hour.get("precipMM")),

        "humidity": _safe_int(hour.get("humidity")),

        "wind_kph": _safe_float(hour.get("windspeedKmph")),
        "wind_mph": _safe_float(hour.get("windspeedMiles")),
        "wind_direction": hour.get("winddir16Point", ""),
        "wind_degree": _safe_int(hour.get("winddirDegree")),

        "visibility_km": _safe_float(hour.get("visibility")),
        "pressure_mb": _safe_float(hour.get("pressure")),
        "cloud_cover": _safe_int(hour.get("cloudcover")),

        "uv_index": _safe_int(hour.get("uvIndex")),

        "dew_point_c": _safe_float(hour.get("DewPointC")),
        "heat_index_c": _safe_float(hour.get("HeatIndexC")),
        "wind_chill_c": _safe_float(hour.get("WindChillC")),
        "wind_gust_kph": _safe_float(hour.get("WindGustKmph")),

        "is_day": hour.get("isdaytime") == "yes",
    }


def _parse_day(day: dict) -> dict:
    """
    Normalize one daily forecast.
    """

    hourly = [
        _parse_hour(hour)
        for hour in day.get("hourly", [])
    ]

    return {
        "date": day.get("date"),

        "max_temperature_c": _safe_float(day.get("maxtempC")),
        "min_temperature_c": _safe_float(day.get("mintempC")),
        "average_temperature_c": _safe_float(day.get("avgtempC")),

        "sunrise": day.get("astronomy", [{}])[0].get("sunrise", ""),
        "sunset": day.get("astronomy", [{}])[0].get("sunset", ""),
        "moonrise": day.get("astronomy", [{}])[0].get("moonrise", ""),
        "moonset": day.get("astronomy", [{}])[0].get("moonset", ""),
        "moon_phase": day.get("astronomy", [{}])[0].get("moon_phase", ""),
        "moon_illumination": _safe_int(
            day.get("astronomy", [{}])[0].get("moon_illumination")
        ),

        "total_snow_cm": _safe_float(day.get("totalSnow_cm")),

        "sun_hour": _safe_float(day.get("sunHour")),
        "uv_index": _safe_int(day.get("uvIndex")),

        "chance_of_rain": _safe_int(day.get("hourly", [{}])[0].get("chanceofrain"))
        if hourly
        else 0,

        "hourly": hourly,
    }


# ============================================================
# CURRENT WEATHER
# ============================================================

def get_current_weather(location: str = "Chittagong") -> dict:
    """
    Return detailed current weather conditions.
    """

    url = f"{BASE_URL}/{location}?format=j1"

    response = requests.get(
        url,
        timeout=10,
        headers={
            "User-Agent": "ZOE-Weather-Agent/1.0"
        },
    )

    response.raise_for_status()

    data = response.json()

    current = data["current_condition"][0]

    return {
        "location": location,

        "temperature_c": _safe_float(current.get("temp_C")),
        "feels_like_c": _safe_float(current.get("FeelsLikeC")),
        "condition": _weather_description(current),

        "humidity": _safe_int(current.get("humidity")),

        "wind_kph": _safe_float(current.get("windspeedKmph")),
        "wind_mph": _safe_float(current.get("windspeedMiles")),
        "wind_direction": current.get("winddir16Point", ""),
        "wind_degree": _safe_int(current.get("winddirDegree")),

        "visibility_km": _safe_float(current.get("visibility")),
        "pressure_mb": _safe_float(current.get("pressure")),
        "cloud_cover": _safe_int(current.get("cloudcover")),

        "uv_index": _safe_int(current.get("uvIndex")),

        "precipitation_mm": _safe_float(current.get("precipMM")),

        "dew_point_c": _safe_float(current.get("DewPointC")),
        "heat_index_c": _safe_float(current.get("HeatIndexC")),
        "wind_chill_c": _safe_float(current.get("WindChillC")),
        "wind_gust_kph": _safe_float(current.get("WindGustKmph")),

        "observation_time": current.get("observation_time"),
    }


# ============================================================
# FULL WEATHER
# ============================================================

def get_weather(
    location: str = "Chittagong",
    days: int = 7,
) -> dict:
    """
    Fetch detailed current weather + multi-day + hourly forecast.

    Returns structured data for the Weather Agent.
    """

    days = max(1, min(days, 7))

    url = f"{BASE_URL}/{location}?format=j1"

    response = requests.get(
        url,
        timeout=10,
        headers={
            "User-Agent": "ZOE-Weather-Agent/1.0"
        },
    )

    response.raise_for_status()

    data = response.json()

    current_raw = data["current_condition"][0]

    current = {
        "temperature_c": _safe_float(current_raw.get("temp_C")),
        "feels_like_c": _safe_float(current_raw.get("FeelsLikeC")),
        "condition": _weather_description(current_raw),

        "humidity": _safe_int(current_raw.get("humidity")),

        "wind_kph": _safe_float(current_raw.get("windspeedKmph")),
        "wind_mph": _safe_float(current_raw.get("windspeedMiles")),
        "wind_direction": current_raw.get("winddir16Point", ""),
        "wind_degree": _safe_int(current_raw.get("winddirDegree")),

        "visibility_km": _safe_float(current_raw.get("visibility")),
        "pressure_mb": _safe_float(current_raw.get("pressure")),
        "cloud_cover": _safe_int(current_raw.get("cloudcover")),

        "uv_index": _safe_int(current_raw.get("uvIndex")),

        "precipitation_mm": _safe_float(current_raw.get("precipMM")),

        "dew_point_c": _safe_float(current_raw.get("DewPointC")),
        "heat_index_c": _safe_float(current_raw.get("HeatIndexC")),
        "wind_chill_c": _safe_float(current_raw.get("WindChillC")),
        "wind_gust_kph": _safe_float(current_raw.get("WindGustKmph")),

        "observation_time": current_raw.get("observation_time"),
    }

    forecast = [
        _parse_day(day)
        for day in data.get("weather", [])[:days]
    ]

    # Location information returned by wttr.in
    nearest_area = data.get("nearest_area", [{}])[0]

    area_name = ""
    country = ""
    region = ""

    try:
        area_name = nearest_area["areaName"][0]["value"]
    except (KeyError, IndexError, TypeError):
        pass

    try:
        country = nearest_area["country"][0]["value"]
    except (KeyError, IndexError, TypeError):
        pass

    try:
        region = nearest_area["region"][0]["value"]
    except (KeyError, IndexError, TypeError):
        pass

    return {
        "location": location,

        "resolved_location": {
            "name": area_name or location,
            "region": region,
            "country": country,
        },

        "current": current,

        "forecast": forecast,

        "fetched_at": datetime.now().isoformat(),
    }


# ============================================================
# BACKWARD-COMPATIBLE SIMPLE WEATHER
# ============================================================

def get_weather_summary(location: str = "Chittagong") -> str:
    """
    Simple human-readable current weather.

    Useful for old code that expects a string.
    """

    weather = get_weather(location, days=1)
    current = weather["current"]

    return (
        f"In {location}, it's "
        f"{current['temperature_c']:.0f} degrees and "
        f"{current['condition']}. "
        f"It feels like {current['feels_like_c']:.0f} degrees, "
        f"with {current['humidity']}% humidity "
        f"and winds at {current['wind_kph']:.0f} kilometers per hour."
    )


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def get_today_forecast(location: str = "Chittagong") -> dict:
    weather = get_weather(location, days=1)

    if not weather["forecast"]:
        return {
            "success": False,
            "error": "No forecast data available.",
        }

    return {
        "success": True,
        "location": weather["location"],
        "current": weather["current"],
        "today": weather["forecast"][0],
    }


def get_week_forecast(location: str = "Chittagong") -> dict:
    weather = get_weather(location, days=7)

    return {
        "success": True,
        "location": weather["location"],
        "forecast": weather["forecast"],
        "fetched_at": weather["fetched_at"],
    }


def find_weather_extremes(
    location: str = "Chittagong",
) -> dict:
    """
    Find hottest, coldest, wettest and windiest forecast days.
    """

    weather = get_weather(location, days=7)
    forecast = weather["forecast"]

    if not forecast:
        return {
            "success": False,
            "error": "No forecast data available.",
        }

    hottest = max(
        forecast,
        key=lambda x: x["max_temperature_c"],
    )

    coldest = min(
        forecast,
        key=lambda x: x["min_temperature_c"],
    )

    wettest = max(
        forecast,
        key=lambda x: max(
            (h["precipitation_mm"] for h in x["hourly"]),
            default=0,
        ),
    )

    windiest = max(
        forecast,
        key=lambda x: max(
            (h["wind_kph"] for h in x["hourly"]),
            default=0,
        ),
    )

    return {
        "success": True,
        "location": location,

        "hottest_day": {
            "date": hottest["date"],
            "max_temperature_c": hottest["max_temperature_c"],
        },

        "coldest_day": {
            "date": coldest["date"],
            "min_temperature_c": coldest["min_temperature_c"],
        },

        "wettest_day": {
            "date": wettest["date"],
            "max_hourly_precipitation_mm": max(
                (
                    h["precipitation_mm"]
                    for h in wettest["hourly"]
                ),
                default=0,
            ),
        },

        "windiest_day": {
            "date": windiest["date"],
            "max_wind_kph": max(
                (
                    h["wind_kph"]
                    for h in windiest["hourly"]
                ),
                default=0,
            ),
        },
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import json

    weather = get_weather()

    print(
        json.dumps(
            weather,
            indent=2,
            ensure_ascii=False,
        )
    )