import ctypes
import os
import shutil
from ctypes import windll
from datetime import datetime
from os import walk as walker
from pathlib import Path
from threading import Thread
from typing import Sequence, Tuple
from zipfile import ZipFile

from alpha_media_signal.config import logger_factory

logger = logger_factory.create(__name__)


def walk(dir):
    file_paths = []
    for (dirpath, dirnames, filenames) in walker(dir):
        for f in filenames:
            file_paths.append(os.path.join(dirpath, f))

    return file_paths


# def get_filename_info(filepath):
#   from alpha_media_signal.utils import date_utils
#
#   components = os.path.basename(filepath).split("_")
#
#   category = components[0]
#   symbol = components[1]
#   datestr = components[2].split('.')[0]
#
#   return {
#     "filepath": filepath,
#     "category": category,
#     "symbol": symbol,
#     "date": date_utils.parse_std_datestring(datestr)
#   }

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


def create_unique_folder(parent_dir: str, prefix: str) -> str:
    proposed_dir = str(create_unique_file_system_name(parent_dir, prefix, extension=None))

    os.makedirs(proposed_dir, exist_ok=False)
    return proposed_dir


def zip_dir(dir_to_zip, output_path):
    with ZipFile(output_path, 'x') as myzip:
        files = walk(dir_to_zip)
        for f in files:
            arcname = f'{f.replace(dir_to_zip, "")}'
            myzip.write(f, arcname=arcname)
        myzip.close()

    return output_path


def get_windows_drive_volume_label(drive_letter: str):
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


# def get_eod_ticker_file_path(symbol: str):
#     return os.path.join(config.constants.SHAR_SPLIT_EQUITY_EOD_DIR, f"{symbol}.csv")


# def get_fun_ticker_file_path(symbol: str):
#     return os.path.join(config.constants.SHAR_SPLIT_FUNDAMENTALS_DIR, f"{symbol}.csv")


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


# def truncate_older(binary_holdout_dir: str):
#     one_path = os.path.join(binary_holdout_dir, BinaryCategoryType.ONE)
#     zero_path = os.path.join(binary_holdout_dir, BinaryCategoryType.ZERO)
#
#     assert (os.path.exists(one_path))
#     assert (os.path.exists(zero_path))
#
#     one_files = walk(one_path)
#     zero_files = walk(zero_path)
#
#     one_fileinfos = [get_filename_info(filepath=fp) for fp in one_files]
#     zero_fileinfos = [get_filename_info(filepath=fp) for fp in zero_files]
#
#     one_fileinfos.sort(key=lambda f: f["date"])
#     zero_fileinfos.sort(key=lambda f: f["date"])
#
#     youngest_one = one_fileinfos[0]
#     youngest_zero = zero_fileinfos[0]
#
#     older_fileinfos, younger_fileinfos, youngest = (zero_fileinfos, one_fileinfos, youngest_one) if youngest_one['date'] > youngest_zero['date'] else (one_fileinfos, zero_fileinfos, youngest_zero)
#     # oldest = youngest_one if youngest_one['date'] > youngest_zero['date'] else youngest_zero
#     youngest_date = youngest['date']
#
#     truncated_older = [ofi for ofi in older_fileinfos if ofi['date'] >= youngest_date]
#
#     truncated_older.sort(key=lambda f: f["date"])
#
#     return truncated_older + younger_fileinfos


# def download_file(url: str, local_dest_path: str, overwrite_existing: bool = True):
#     url_request = urllib.request.Request(url, headers={})
#     url_connect = urllib.request.urlopen(url_request)
#
#     if os.path.exists(local_dest_path) and not overwrite_existing:
#         basename = os.path.basename(local_dest_path)
#         logger.info(f"Download aborted! File '{basename}' already exists.")
#         return
#
#     buffer_size = 256
#
#     try:
#         # remember to open file in bytes mode
#         with open(local_dest_path, 'wb') as f:
#             while True:
#                 buffer = url_connect.read(buffer_size)
#                 if not buffer: break
#
#                 # an integer value of size of written data
#                 data_wrote = f.write(buffer)
#     except:
#         raise Exception("Encountered a problem attempting to download file.")
#     finally:
#         # you could probably use with-open-as manner
#         url_connect.close()
