import time


class Stopwatch:
    start_time = None

    def __init__(self, start_now=True):
        if start_now:
            self.start_time = time.time()

    def start(self):
        if self.start_time is not None:
            raise Exception("Stopwatch already started.")

        self.start_time = time.time()

    def end(self, msg: str = "Time", print_elapsed=True):
        elapsed = round(time.time() - self.start_time, 4)
        print(f"{msg} elapsed: {elapsed}s")