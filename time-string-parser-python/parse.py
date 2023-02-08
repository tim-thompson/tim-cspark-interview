import re
from datetime import datetime, timedelta

snap_unit_operations = {
    "s": lambda time: time.replace(microsecond=0),
    "m": lambda time: time.replace(second=0, microsecond=0),
    "h": lambda time: time.replace(minute=0, second=0, microsecond=0),
    "d": lambda time: time.replace(hour=0, minute=0, second=0, microsecond=0),
    "mon": lambda time: time.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
    "y": lambda time: time.replace(
        month=1, day=1, hour=0, minute=0, second=0, microsecond=0
    ),
}


def parse(time_str: str) -> datetime:
    """Parses a time string and returns a datetime object

    Args:
        time_str (str): A string in the format of 'now()[+/-][offset]@[snap_unit]'

    Raises:
        Exception: when the time string is invalid

    Returns:
        datetime: the parsed datetime object
    """
    time_str = time_str.strip()

    if time_str.startswith("now()"):
        now = datetime.utcnow()

        # Split the time string into modifiers and snap units
        time_str_split = time_str[5:].split("@")

        modified_date = _parse_and_apply_offset(now, time_str_split[0])

        if len(time_str_split) > 1: # If there is a snap unit
            modified_date = _parse_and_apply_snap(modified_date, time_str_split[1])

        return modified_date
    else:
        raise Exception(
            "Invalid time function specified, currently only 'now()' is supported"
        )


def _parse_and_apply_offset(
    existing_datetime: datetime, offset_str: str
) -> datetime:
    """Parses an offset string and applies the offset to a datetime object

    Args:
        existing_datetime (datetime): The datetime object to apply the offset to
        offset_str (str): The offset string in the format of '[+-][offset]'

    Raises:
        Exception: when the offset string is invalid

    Returns:
        datetime: the datetime object with the offset applied
    """
    if offset_str:
        offset = offset_str.strip()
        modifications = re.split("([+-])", offset)

        for modification in modifications:
            if modification:
                if modification == "+":
                    operation = (
                        lambda existing_datetime, delta: existing_datetime + delta
                    )
                elif modification == "-":
                    operation = (
                        lambda existing_datetime, delta: existing_datetime - delta
                    )
                else:
                    delta = timedelta()
                    try:
                        if "s" in modification:
                            delta += timedelta(seconds=int(modification[:-1]))
                        elif "mon" in modification:
                            delta += timedelta(days=30 * int(modification[:-3]))
                        elif "m" in modification:
                            delta += timedelta(minutes=int(modification[:-1]))
                        elif "h" in modification:
                            delta += timedelta(hours=int(modification[:-1]))
                        elif "d" in modification:
                            delta += timedelta(days=int(modification[:-1]))
                        elif "y" in modification:
                            delta += timedelta(days=365 * int(modification[:-1]))
                        else:
                            raise Exception("Invalid time offset")
                    except ValueError:
                        raise Exception("Invalid time offset")

                    # Calculate the new datetime using the operation
                    existing_datetime = operation(existing_datetime, delta)

    return existing_datetime


def _parse_and_apply_snap(existing_datetime: datetime, snap_str: str) -> datetime:
    """Parses a snap string and applies the snap to a datetime object

    Args:
        existing_datetime (datetime): The datetime object to apply the snap to
        snap_str (str): The snap string in the format of '[snap_unit]'

    Raises:
        Exception: when the snap string is invalid

    Returns:
        datetime: the datetime object with the snap applied
    """    
    if snap_str:
        snap = snap_str.strip()
        if snap in snap_unit_operations:
            return snap_unit_operations[snap](existing_datetime)
        else:
            raise Exception("Invalid snap time unit")
