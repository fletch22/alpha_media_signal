import threading
from queue import Queue

from ams.config import logger_factory

logger = logger_factory.create(__name__)


class PrinterThread():
    def __init__(self):
        self.display_queue = Queue()  # synchronizes console output
        self.thread = threading.Thread(
            target=self.display_worker,
            args=(self.display_queue,),
        )

    def display_worker(self, display_queue):
        while True:
            line = display_queue.get()
            if line is None:  # simple termination logic, other sentinels can be used
                break
            if line is not None:
                logger.info(line)

    def start(self):
        self.thread.start()

    def end(self):
        self.display_queue.put(None)
        self.thread.join()

    def print(self, message: object):
        self.display_queue.put(message)
