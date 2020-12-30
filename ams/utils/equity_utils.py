import math
import re
from statistics import mean
from typing import List


def convert_equity_name_to_common(name: str):
    tokens = [
        ' A',
        ' & Co',
        ' Corporation',
        ' Corp',
        ' Com',
        ' Co',
        ' Inc',
        ' Co Ltd',
        ' Ltd',
        ' plc',
        ' LLC',
        ' AG',
        ' SA',
        ' S A',
        ' SpA',
        ' SE',
        ' LP',
        ' (International)',
        ' International',
        ' Usa',
        ' USA',
        ' Acquisition',
        ' Holding Co',
        ' Holdings',
        ' Holding',
        ' Group',
        ' Bancorp',
        ' Bancorporation',
        ' Bancshares',
        ' Interactive Entertainment Technology',
        ' Capital',
        ' Partners',
        ' Industries',
        ' REIT',
        ' Systems',
        ' Therapeutics',
        ' Technologies',
        ' Pharmaceutical',
        ' Financial',
        ' Software',
        ' N V',
        ' Nv',
        ' I',
        ' II',
        ' III',
        ' IV',
        ' &',
    ]

    for t in tokens:
        escaped = re.escape(t)
        pattern = re.compile(f'{escaped}$', re.IGNORECASE)
        name = re.sub(pattern, '', name)

    return name


def calc_variance(lst: List[float]):
    mean_diff = mean(lst)
    sum_sq_diff = 0
    for d in lst:
        sq_diff = (d - mean_diff) ** 2
        sum_sq_diff += sq_diff
    variance = math.sqrt(sum_sq_diff / (len(lst)))
    return variance