import os
import shutil
from pathlib import Path
from typing import List, Callable, TypeVar

from ams.config import constants
from ams.services import file_services

STAGING_FOLDER_NAME = "staging"
IN_TRANSITION_ENDING = ".in_transition"
ARCHIVE_FOLDER_NAME = "archive"

pipe_process_callback = TypeVar('pipe_process_callback', bound=Callable[[Path, Path], None])


def get_files_in_transition(dir_path: Path) -> List[Path]:
    return file_services.list_files(dir_path, IN_TRANSITION_ENDING, use_dir_recursion=False)


def revert_to_original_location(source_dir_path: Path, files_in_transition: List[Path]):
    for f in files_in_transition:
        filename = f.name
        dest_path = Path(source_dir_path, filename[:-len(IN_TRANSITION_ENDING)])
        shutil.move(str(f), str(dest_path))


def remove_remaining_files(staging_path: Path):
    files = file_services.list_files(staging_path, use_dir_recursion=False)
    for f in files:
        f.unlink()


def revert_in_transition_files(source_dir_path: Path):
    staging_path = Path(source_dir_path, STAGING_FOLDER_NAME)
    files_in_transition = get_files_in_transition(staging_path)

    revert_to_original_location(source_dir_path, files_in_transition)

    remove_remaining_files(staging_path)


def start(source_path: Path, output_dir_path: Path, process_callback: pipe_process_callback):
    revert_in_transition_files(source_dir_path=source_path)

    process(source_path, output_dir_path, process_callback=process_callback)


def move_files_to_staging(source_path: Path):
    staging_dir_path = Path(source_path, STAGING_FOLDER_NAME)
    os.makedirs(staging_dir_path, exist_ok=True)

    files = file_services.list_files(source_path, ends_with=".txt", use_dir_recursion=False)
    for f in files:
        filename = f.name
        dest_path = Path(staging_dir_path, f"{filename}{IN_TRANSITION_ENDING}")
        shutil.move(str(f), str(dest_path))

    return staging_dir_path


def archive(source_path: Path, staging_dir_path: Path):
    archive_dir_path = Path(source_path, ARCHIVE_FOLDER_NAME)
    os.makedirs(archive_dir_path, exist_ok=True)

    files_in_transition = get_files_in_transition(staging_dir_path)
    archive_output_dir_path = file_services.create_unique_folder_name(archive_dir_path, prefix="archive")
    for f in files_in_transition:
        filename = f.name
        dest_path = Path(archive_output_dir_path, filename[:-len(IN_TRANSITION_ENDING)])
        shutil.move(str(f), str(dest_path))


def process(source_path: Path, output_dir_path: Path, process_callback: Callable[[Path, Path], None]):
    staging_dir_path = move_files_to_staging(source_path)

    process_callback(staging_dir_path, output_dir_path)

    archive(source_path=source_path, staging_dir_path=staging_dir_path)


if __name__ == '__main__':
    output_dir_path = Path(constants.DATA_PATH, "bar")
    source_dir_path = Path(constants.DATA_PATH, "foo")

    def foo(bar: Path, banana: Path):
        pass

    start(source_path=source_dir_path, output_dir_path=output_dir_path, process_callback=foo)
