from datetime import datetime


def get_current_time() -> str:
    now = datetime.now()

    return now.strftime(
        "It is %A, %B %-d, %Y, at %-I:%M %p."
    )