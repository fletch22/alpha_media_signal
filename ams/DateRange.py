from datetime import datetime

from ams.utils import date_utils


class DateRange:
    from_date = None
    to_date = None

    def __init__(self, from_date: datetime, to_date: datetime):
        self.from_date = from_date
        self.to_date = to_date

    @classmethod
    def from_date_strings(cls, from_date_str: str, to_date_str: str):
        from_date = date_utils.parse_std_datestring(from_date_str)
        to_date = date_utils.parse_std_datestring(to_date_str)
        return DateRange(from_date=from_date, to_date=to_date)

    @property
    def start_date_str(self):
        return date_utils.get_standard_ymd_format(self.from_date)

    @property
    def end_date_str(self):
        return date_utils.get_standard_ymd_format(self.to_date)