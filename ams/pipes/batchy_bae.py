import os
import shutil
from pathlib import Path
from typing import List, Callable, TypeVar

from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.services import file_services

logger = logger_factory.create(__name__)

STAGING_FOLDER_NAME = "stage"
IN_TRANSITION_ENDING = ".in_transition"
ARCHIVE_FOLDER_NAME = "archive"

pipe_process_callback = TypeVar('pipe_process_callback', bound=Callable[[Path, Path], None])


def get_files_in_transition(dir_path: Path) -> List[Path]:
    return file_services.list_files(dir_path, IN_TRANSITION_ENDING, use_dir_recursion=True)


def get_folders_in_transition(dir_path: Path) -> List[Path]:
    return file_services.list_child_folders(dir_path, ends_with=IN_TRANSITION_ENDING)


def revert_files_to_original_location(dest_dir_path: Path, files_in_transition: List[Path]):
    for f in files_in_transition:
        filename = f.name
        dest_path = Path(dest_dir_path, filename[:-len(IN_TRANSITION_ENDING)])
        shutil.move(str(f), str(dest_path))


def delete_handle_error(dir_path: Path):
    """
    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmdir(path, onerror=onerror)``
    """
    file_services.remove_folder_read_only(dir_path=dir_path, recursive=True)

    files = file_services.list_files(dir_path, use_dir_recursion=True)
    for f in files:
        f.unlink()

    folders = file_services.list_child_folders(dir_path)
    for fld in folders:
        delete_handle_error(fld)

    try:
        dir_path.rmdir()
    # NOTE: 2021-02-27: chris.flesche: May have already been deleted.
    except FileNotFoundError as fe:
        pass


def remove_remaining_files(target_path: Path, should_delete: bool):
    files = file_services.list_files(target_path, use_dir_recursion=True)
    folders = file_services.list_child_folders(target_path)

    if should_delete:
        for f in files:
            f.unlink()
        for fld in folders:
            delete_handle_error(fld)
    else:
        tmp_trash = file_services.create_unique_folder_name(constants.TWITTER_TRASH_OUTPUT, prefix="batchy_bae")
        for f in files:
            file_path_new = Path(tmp_trash, f.name)
            if f.exists():
                shutil.move(str(f), str(file_path_new))

        for f in folders:
            file_path_new = Path(tmp_trash, f.name)
            if f.exists():
                shutil.move(str(f), str(file_path_new))


def get_stage_dir(src_dir_path: Path):
    return Path(src_dir_path.parent, STAGING_FOLDER_NAME)


def revert_in_transition_files(src_dir_path: Path):
    staging_path = get_stage_dir(src_dir_path=src_dir_path)
    ensure_dir(staging_path)
    files_in_transition = get_files_in_transition(staging_path)

    revert_files_to_original_location(dest_dir_path=src_dir_path, files_in_transition=files_in_transition)

    remove_remaining_files(staging_path, should_delete=True)


def start(source_path: Path, out_dir_path: Path, process_callback: pipe_process_callback, should_delete_leftovers: bool, should_archive: bool = True,
          snow_plow_stage: bool = True):
    if snow_plow_stage:
        target_path = get_stage_dir(src_dir_path=source_path)
        remove_remaining_files(target_path=target_path, should_delete=should_delete_leftovers)
    else:
        # NOTE: 2021-02-07: chris.flesche: Helps with debugging.
        revert_in_transition_files(src_dir_path=source_path)

    if file_services.has_no_files(the_path=source_path):
        return False
    else:
        process(source_path, out_dir_path=out_dir_path, process_callback=process_callback, should_archive=should_archive)
        return True


def move_to_staging(source_path: Path):
    staging_dir_path = Path(source_path.parent, STAGING_FOLDER_NAME)
    os.makedirs(staging_dir_path, exist_ok=True)

    files = file_services.list_files(source_path, use_dir_recursion=True)
    for f in files:
        filename = f.name
        dest_path = Path(staging_dir_path, f"{filename}{IN_TRANSITION_ENDING}")
        ensure_dir(dest_path.parent)
        shutil.move(str(f), str(dest_path))

    return staging_dir_path


def archive(source_path: Path, staging_dir_path: Path):
    archive_dir_path = Path(source_path.parent, ARCHIVE_FOLDER_NAME)
    os.makedirs(archive_dir_path, exist_ok=True)

    files_in_transition = get_files_in_transition(staging_dir_path)
    total_files = len(files_in_transition)
    archive_output_dir_path = file_services.create_unique_folder_name(archive_dir_path, prefix="archive", ensure_exists=False)
    if total_files > 0:
        os.makedirs(archive_output_dir_path, exist_ok=True)
        for f in files_in_transition:
            filename = f.name
            dest_path = Path(archive_output_dir_path, filename[:-len(IN_TRANSITION_ENDING)])
            shutil.move(str(f), str(dest_path))

    if total_files == 0:
        logger.info("WARNING: No files found to archive.")


def unstage(source_path: Path, out_dir_path: Path):
    logger.info("Unstaging...")
    files_in_transition = get_files_in_transition(source_path)
    total_files = len(files_in_transition)
    if total_files > 0:
        for f in files_in_transition:
            filename = f.name
            dest_path = Path(out_dir_path, filename[:-len(IN_TRANSITION_ENDING)])
            shutil.move(str(f), str(dest_path))

    if total_files == 0:
        logger.info("WARNING: No files found to archive.")


def ensure_clean_output_path(out_dir_path: Path, should_delete_remaining: bool):
    if not file_services.is_empty(out_dir_path):
        logger.info("Cleaning output path ...")
        remove_remaining_files(out_dir_path, should_delete=should_delete_remaining)


def process(source_path: Path, out_dir_path: Path, process_callback: Callable[[Path, Path], None], should_archive: bool = True):
    if not source_path.exists():
        raise Exception(f"Source folder does not exist: '{source_path}'.")

    staging_dir_path = move_to_staging(source_path)
    logger.info(f"Moved to staging folder '{staging_dir_path}'.")

    process_callback(staging_dir_path, out_dir_path)

    if should_archive:
        logger.info("Will archive.")
        archive(source_path=source_path, staging_dir_path=staging_dir_path)
    else:
        logger.info("Will unstage.")
        unstage(source_path=staging_dir_path, out_dir_path=source_path)


if __name__ == '__main__':
    source_dir_path = Path(constants.DATA_PATH, "foo_drop", "main")
    output_dir_path = Path(constants.DATA_PATH, "bar")


    def foo(bar: Path, banana: Path):
        pass


    start(source_path=source_dir_path, out_dir_path=output_dir_path, process_callback=foo, should_archive=False, should_delete_leftovers=False)