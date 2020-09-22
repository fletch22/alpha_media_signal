import random
from datetime import timedelta

from ams.utils import date_utils


def test_dt():
    date_max = "2020-11-01"
    date_min = "2020-11-30"

    timestamp_max = date_utils.parse_std_datestring(date_max)
    timestamp_min = date_utils.parse_std_datestring(date_min)

    total_days = (timestamp_max - timestamp_min).days
    rnd_days = int(random.randint(total_days, 10))

    rand_dt = timestamp_min + timedelta(days=rnd_days)
    rand_date_string = date_utils.get_standard_ymd_format(rand_dt)

    print(rand_date_string)


def test_return():
    success_rt = .53

    min = 40

    rnd_amts = []
    for i in range(100):
        rand_cents = random.randint(0, 100)
        rnd_amts.append(float(rand_cents / 100))

    success_pct = success_rt * 100

    up_or_downs = [True if i < success_pct else False for i in range(100)]
    random.shuffle(up_or_downs)

    tots = 40
    for ndx, ud in enumerate(up_or_downs):
        amt = rnd_amts[ndx]
        print(amt)
        if ud:
            tots += amt
        else:
            tots += -amt


