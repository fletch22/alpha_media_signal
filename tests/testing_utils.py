from pathlib import Path

from ams.services import file_services


def create_random_text_file(parent_dir_path: Path):
    file_path = file_services.create_unique_filename(parent_dir=str(parent_dir_path), prefix="testing_", extension="txt")
    file_services.create_text_file(file_path=file_path, contents="Abracadabra")