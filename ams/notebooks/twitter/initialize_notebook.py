import sys

paths_to_add = ['/home/jovyan/work']

for p in paths_to_add:
    if p not in sys.path:
        sys.path.append(p)
