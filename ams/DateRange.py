from datetime import datetime

from ams.utils import date_utils


class DateRange:
    def __init__(self, from_date: datetime, to_date: datetime):
        self.from_date = from_date
        self.to_date = to_date

    @classmethod
    def from_date_strings(cls, from_date_str: str, to_date_str: str):
        from_date = date_utils.parse_std_datestring(from_date_str)
        to_date = date_utils.parse_std_datestring(to_date_str)
        return DateRange(from_date=from_date, to_date=to_date)