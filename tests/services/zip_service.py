import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from zipfile import ZipFile

from ams.services import file_services
from ams.config import logger_factory

logger = logger_factory.create(__name__)


def zip_file(file_path: Path, output_path: Path):
    with zipfile.ZipFile(str(output_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(filename=file_path, arcname=file_path.name)


def extract_file(source_file_path: Path, output_parent_path: Path):
    with ZipFile(str(source_file_path)) as zip_ref:
        zip_ref.extractall(output_parent_path)


def raw_from_zip_generator(temp_dir_path: Path, source_path: Path, already_proc_filenames: List[str]):
    zips = file_services.list_files(parent_path=source_path, ends_with=".zip")
    logger.info(f"Number of zips found: {len(zips)}")
    for z in zips:
        zip_file = zipfile.ZipFile(str(z))
        internal_files = zip_file.filelist
        for infi in internal_files:
            if not infi.is_dir():
                filename = Path(infi.filename).name
                if filename in already_proc_filenames:
                    logger.info(f"File has already been processed. Skipping {filename}")
                    continue
                output_path = Path(temp_dir_path, filename)
                with open(output_path, "wb") as f:  # open the output path for writing
                    f.write(zip_file.read(infi))  # save the contents of the file in it
                yield output_path

