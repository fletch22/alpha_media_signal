import shutil
from pathlib import Path

from ams.config import constants
from ams.services import file_services


class TempDir():
    dir_path = None

    def __init__(self):
        self.dir_path = Path(file_services.create_unique_folder_name(parent_dir=constants.TRANSIENT_DIR_PATH, prefix="TempDir"))

    def __enter__(self):
        return self.dir_path

    def __exit__(self, type, value, traceback):
        if self.dir_path is not None and self.dir_path.exists():
            shutil.rmtree(self.dir_path)
