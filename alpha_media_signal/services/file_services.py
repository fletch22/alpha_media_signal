import ctypes
import os
import shutil
from datetime import datetime
from os import walk as walker
from pathlib import Path
from threading import Thread
from typing import Sequence, Tuple
from zipfile import ZipFile

from alpha_media_signal.config import logger_factory

logger = logger_factory.create(__name__)


def walk(the_path: Path):
    file_paths = []
    for (dirpath, dirnames, filenames) in walker(str(the_path)):
        for f in filenames:
            file_paths.append(Path(dirpath, f))

    return file_paths


def list_files(parent_path: Path, ends_with: str = None):
    file_paths = walk(parent_path)
    return list(filter(lambda x: x.is_file() and (ends_with is None or str(x).endswith(ends_with)), file_paths))



def create_unique_file_system_name(parent_dir: str, prefix: str, extension: str = None) -> Path:
    from alpha_media_signal.utils import date_utils

    date_str = date_utils.format_file_system_friendly_date(datetime.now())
    proposed_core_item_name = f"{prefix}_{date_str}"

    if extension is not None:
        proposed_core_item_name = f"{proposed_core_item_name}.{extension}"

    proposed_item = Path(parent_dir, proposed_core_item_name)
    count = 1
    while os.path.exists(proposed_item):
        proposed_item = Path(parent_dir, f"{proposed_core_item_name}-({count})")
        count += 1
        if count > 10:
            raise Exception("Something went wrong. Too many files with similar names.")

    return proposed_item


def get_unique_folder(parent_dir: str, prefix: str, ensure_exists: bool = True) -> str:
    proposed_dir = str(create_unique_file_system_name(parent_dir, prefix, extension=None))

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
