import time


class Stopwatch:
    start = None

    def __init__(self, start_now=False):
        self.start = time.time()

    def end(self, msg: str = "Time", print_elapsed=True):
        elapsed = round(time.time() - self.start, 4)
        print(f"{msg} elapsed: {elapsed}s")