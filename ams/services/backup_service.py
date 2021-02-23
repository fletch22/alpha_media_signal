import os
import zipfile
from pathlib import Path

from ams.config import logger_factory

logger = logger_factory.create(__name__)


def zipdir(dir_path: Path, ziph, omit_folders: list = None):
    root_len = len(str(dir_path))
    for root, dirs, files in os.walk(str(dir_path)):
        for f in files:
            source_path = os.path.join(root, f)
            arcname = f"{root[root_len:]}/{f}"
            skip_file = False
            for o in omit_folders:
                if arcname.startswith(o):
                    skip_file = True

            if not skip_file:
                logger.info(f"{arcname}")
                ziph.write(source_path, arcname)


def backup_folder(backup_source_path: Path, output_path: Path):
    with zipfile.ZipFile(str(output_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(backup_source_path, zipf, omit_folders=["\\venv", "\\.git"])


def backup_file(backup_source_file: Path, output_path: Path):
    with zipfile.ZipFile(str(output_path), 'w') as zipf:
        zipf.write(str(backup_source_file), arcname=backup_source_file.name, compress_type=zipfile.ZIP_DEFLATED)