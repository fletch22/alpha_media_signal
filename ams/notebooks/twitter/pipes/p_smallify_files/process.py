import json
from pathlib import Path

from ams.config import constants
from ams.config.constants import ensure_dir
from ams.notebooks.twitter.pipes import batchy_bae
from ams.services import file_services


def process(source_path: Path, output_dir_path: Path):
    files = file_services.walk(source_path, use_dir_recursion=False)

    print(f"Num files to process: {len(files)}")

    max_records_per_file = 50000
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
                    obj = json.loads(line)
                    line_alt = json.dumps(obj)
                    wf.write(line_alt + "\n")
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


def start():
    source_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "raw_drop", "main")
    output_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "smallified_raw_drop", "main")
    ensure_dir(output_dir_path)

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path,
                     process_callback=process, should_archive=False)

    return output_dir_path


if __name__ == '__main__':
    start()