from datetime import datetime
from zoneinfo import ZoneInfo


ZOE_TIMEZONE = ZoneInfo("Asia/Dhaka")


def get_time_context() -> dict:
    now = datetime.now(ZOE_TIMEZONE)

    hour = now.hour

    if 5 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 22:
        period = "evening"
    else:
        period = "late_night"

    return {
        "datetime": now.isoformat(),
        "date": f"{now.strftime('%B')} {now.day}, {now.year}",
        "day": now.strftime("%A"),
        "time": now.strftime("%I:%M %p"),
        "timezone": "Asia/Dhaka",
        "period": period,
    }