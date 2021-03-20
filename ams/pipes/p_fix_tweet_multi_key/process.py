import json
from pathlib import Path

from ams.config import logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services

logger = logger_factory.create(__name__)


def process(source_path: Path, output_dir_path: Path):
    parent_path = source_path
    files = file_services.walk(parent_path, use_dir_recursion=False)

    logger.info(f"Num files: {len(files)}")

    for f in files:
        count = 0
        f_fixed_path = Path(output_dir_path, f"{f.stem}_fixed.txt")
        with open(str(f_fixed_path), 'a+') as aw:
            logger.info(f"Loading {str(f)}")
            with open(str(f), 'r+') as r:
                while True:
                    count += 1
                    line = r.readline()
                    if len(line) == 0:
                        break
                    try:
                        obj = json.loads(line)
                        line_alt = json.dumps(obj)
                        aw.write(line_alt + "\n")
                    except Exception as e:
                        pass

                    if count % 10000 == 0:
                        logger.info(count)
                logger.info("Closing file.")

    logger.info(f"Total records processed: {count}")


def start(source_dir_path: Path, twitter_root_path: Path, should_delete_leftovers: bool):
    output_dir_path = Path(twitter_root_path, "fixed_drop", "main")
    ensure_dir(output_dir_path)

    batchy_bae.ensure_clean_output_path(output_dir_path, should_delete_remaining=should_delete_leftovers)

    batchy_bae.start_drop_processing(source_path=source_dir_path, out_dir_path=output_dir_path,
                                     process_callback=process, should_archive=False, should_delete_leftovers=should_delete_leftovers)

    return output_dir_path
