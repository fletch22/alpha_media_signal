import json
from pathlib import Path

from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services

logger = logger_factory.create(__name__)


def fix_naked_ticker(broke_json):
    token = '{"version": "0.9.2", "f22_ticker": '
    start_pos = len(token)
    next_char = broke_json[start_pos:start_pos + 1]
    if next_char == "\"":
        return broke_json
    end_ticker_pos = broke_json.index(",", start_pos)
    ticker = broke_json[start_pos:end_ticker_pos]

    return f"""{broke_json[:start_pos]}\"{ticker}\"{broke_json[end_ticker_pos:]}"""


def process(source_path: Path, output_dir_path: Path):
    files = file_services.walk(source_path, use_dir_recursion=False)

    print(f"Num files to process: {len(files)}")

    max_records_per_file = 50000
    total_records_processed = 0
    for f in files:
        count = 0
        with open(str(f), 'r+') as r:
            wf = None
            create_new = True
            while True:
                if create_new:
                    file_path = file_services.create_unique_filename(parent_dir=str(output_dir_path), prefix="smallified", extension="txt")
                    wf = open(str(file_path), 'a+')
                    create_new = False
                count += 1
                line = r.readline()
                if len(line) == 0:
                    if wf is not None:
                        print("Closing file.")
                        wf.close()
                    break
                try:
                    line = line.replace('{"version": "0.9.1", "f22_ticker": ticker, ', '{"version": "0.9.1", "f22_ticker": "ticker", ')
                    line = fix_naked_ticker(line)
                    obj = json.loads(line)
                    if "version" in obj.keys():
                        obj = obj["tweet"]
                    line_alt = json.dumps(obj)
                    wf.write(line_alt + "\n")
                    total_records_processed += 1
                except Exception as e:
                    pass
                if count % 1000 == 0:
                    print(count)
                if count >= max_records_per_file:
                    if wf is not None:
                        print("Closing file.")
                        wf.close()
                    count = 0
                    create_new = True

    logger.info(f"Total records processed: {total_records_processed}")


def start(source_dir_path: Path, twitter_root_path: Path, snow_plow_stage: bool):
    output_dir_path = Path(twitter_root_path, "smallified_raw_drop", "main")
    ensure_dir(output_dir_path)

    batchy_bae.ensure_clean_output_path(output_dir_path)

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path,
                     process_callback=process, should_archive=False, snow_plow_stage=snow_plow_stage)

    return source_dir_path, output_dir_path


def start_old():
    source_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "raw_drop", "main")
    output_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "smallified_raw_drop", "main")
    ensure_dir(output_dir_path)

    batchy_bae.ensure_clean_output_path(output_dir_path)

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path,
                     process_callback=process, should_archive=False)

    return output_dir_path