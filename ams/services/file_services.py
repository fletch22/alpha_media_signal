import ctypes
import os
import shutil
from datetime import datetime
from os import walk as walker
from pathlib import Path
from threading import Thread
from typing import Sequence, Tuple, List
from zipfile import ZipFile

from ams.config import logger_factory, constants

logger = logger_factory.create(__name__)


def is_empty(the_path: Path):
    for (dirpath, dirnames, filenames) in walker(str(the_path)):
        if len(dirnames) > 0 or len(filenames) > 0:
            return False
    return True


def walk(the_path: Path, use_dir_recursion: bool = True):
    file_paths = []
    for (dirpath, dirnames, filenames) in walker(str(the_path)):
        for f in filenames:
            file_paths.append(Path(dirpath, f))
        if not use_dir_recursion:
            break

    return file_paths


def list_files(parent_path: Path, ends_with: str = None, use_dir_recursion: bool = True):
    file_paths = walk(parent_path, use_dir_recursion=use_dir_recursion)
    file_paths = list(filter(lambda x: x.is_file() and (ends_with is None or str(x).endswith(ends_with)), file_paths))
    return file_paths if file_paths is not None else []


def list_child_folders(parent_path: Path, ends_with: str = None) -> List[Path]:
    results = []
    for (_, dirnames, _) in walker(str(parent_path)):
        folder_names = list(filter(lambda x: ends_with is None or str(x).endswith(ends_with), dirnames))
        results += [Path(parent_path, x) for x in folder_names]
    return results


def create_unique_filename(parent_dir: str, prefix: str, extension: str = None) -> Path:
    from ams.utils import date_utils

    date_str = date_utils.format_file_system_friendly_date(datetime.now())
    proposed_core_item_name = f"{prefix}_{date_str}"

    filename = proposed_core_item_name
    if extension is not None:
        filename = f"{proposed_core_item_name}.{extension}"

    proposed_item = Path(parent_dir, filename)
    count = 1
    while proposed_item.exists():
        filename = f"{proposed_core_item_name}-({count})"
        if extension is not None:
            filename = f"{filename}.{extension}"
        proposed_item = Path(parent_dir, filename)
        count += 1
        if count > 100:
            raise Exception("Something went wrong. Too many files with similar names.")

    return proposed_item


def create_unique_folder_name(parent_dir: str, prefix: str, ensure_exists: bool = True) -> str:
    proposed_dir = str(create_unique_filename(parent_dir, prefix, extension=None))

    if ensure_exists:
        os.makedirs(proposed_dir, exist_ok=True)

    return proposed_dir


def zip_dir(dir_to_zip: Path, output_path):
    with ZipFile(output_path, 'x') as myzip:
        files = walk(dir_to_zip)
        for f in files:
            arcname = f'{f.replace(str(dir_to_zip), "")}'
            myzip.write(f, arcname=arcname)
        myzip.close()

    return output_path


def get_windows_drive_volume_label(drive_letter: str):
    from ctypes import windll

    volumeNameBuffer = ctypes.create_unicode_buffer(1024)
    fileSystemNameBuffer = ctypes.create_unicode_buffer(1024)

    windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(f"{drive_letter}:\\"),
        volumeNameBuffer,
        ctypes.sizeof(volumeNameBuffer),
        fileSystemNameBuffer,
        ctypes.sizeof(fileSystemNameBuffer)
    )

    return volumeNameBuffer.value


def get_date_modified(file_path):
    unix_date = os.path.getmtime(file_path)
    return datetime.fromtimestamp(unix_date)


def file_modified_today(file_path):
    return datetime.today().timetuple().tm_yday - get_date_modified(file_path).timetuple().tm_yday == 0


def get_folders_in_dir(path: str):
    folders = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for folder in d:
            folders.append(os.path.join(r, folder))

    return folders


def fast_copy(files_many: Sequence[Tuple[str, str]]):
    for source_path, destination_path in files_many:
        Thread(target=shutil.copy, args=[source_path, destination_path]).start()


def get_eod_ticker_file_path(symbol: str) -> Path:
    return Path(constants.SHAR_SPLIT_EQUITY_EOD_DIR, f"{symbol}.csv")


def get_single_file_from_path(dir_path: Path, filename_ends_with: str):
    assert (dir_path.exists())

    files = list_files(dir_path, filename_ends_with)
    assert (len(files) == 1)

    return files[0]


def unnest_files(parent: Path, target_path: Path, filename_ends_with: str):
    files = list_files(parent_path=parent, ends_with=filename_ends_with, use_dir_recursion=True)

    for f in files:
        file_dest = Path(target_path, f.name)
        if file_dest.exists():
            if filename_ends_with.startswith("."):
                filename_ends_with = filename_ends_with[1:]
            file_dest = create_unique_filename(parent_dir=str(file_dest.parent), prefix="unnested", extension=filename_ends_with)
        shutil.move(str(f), dst=str(file_dest))