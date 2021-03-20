from pathlib import Path
from typing import Callable

from ams.config import constants


class Pipe:
    src_dir_path = None
    dest_dir_path = None

    def __init__(self, src_dir_path: Path, dest_dir_path: Path, fn_start: Callable, **kwargs):
        self.src_dir_path = src_dir_path
        self.dest_dir_path = dest_dir_path
        self.kwargs = kwargs
        self.fn_start = fn_start

    def start_pipe(self):
        kwargs = dict(src_dir_path=self.src_dir_path,
                      dest_dir_path=self.dest_dir_path).update(self.kwargs)
        self.fn_start(kwargs)

    def connect_next_pipe(self, dest_dir_path: Path, fn_start: Callable, **kwargs):
        return Pipe(src_dir_path=self.dest_dir_path,
                    dest_dir_path=dest_dir_path,
                    fn_start=fn_start,
                    **kwargs)


if __name__ == '__main__':
    src_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "raw_drop", "main")
    dest_smallified_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "smallify_process", "main")
    dest_flattened_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "flatten_process", "main")
    dest_added_id_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "add_id", "main")
    snow_plow_stage = True
    should_delete_leftovers = True

    from ams.pipes.p_smallify_files import process as smallify_process
    from ams.pipes.p_flatten import process as flatten_process
    from ams.pipes.p_add_id import process as add_id_process

    Pipe(src_dir_path=src_dir_path,
         dest_dir_path=dest_smallified_path,
         snow_plow_stage=True,
         should_delete_leftovers=should_delete_leftovers,
         fn_start=smallify_process.start) \
        .connect_next_pipe(dest_dir_path=dest_flattened_path, fn_start=flatten_process.start, snow_plow_stage=snow_plow_stage, should_delete_leftovers=should_delete_leftovers) \
        .connect_next_pipe(dest_dir_path=dest_added_id_path, fn_start=add_id_process.start, snow_plow_stage=snow_plow_stage, should_delete_leftovers=should_delete_leftovers)