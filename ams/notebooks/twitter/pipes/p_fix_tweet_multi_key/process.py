import json
from pathlib import Path

from ams.config import constants
from ams.notebooks.twitter.pipes import batchy_bae
from ams.services import file_services


def process(source_path: Path, output_dir_path: Path):
    parent_path = source_path
    files = file_services.walk(parent_path, use_dir_recursion=False)

    print(f"Num files: {len(files)}")

    for f in files:
        count = 0
        f_fixed_path = Path(output_dir_path, f"{f.stem}_fixed.txt")
        with open(str(f_fixed_path), 'a+') as aw:
            print(f"Loading {str(f)}")
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

                    if count % 1000 == 0:
                        print(count)
                print("Closing file.")


def start():
    source_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "raw_drop", "staging")
    output_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "fixed_drop")

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path, process_callback=process)

    return output_dir_path


if __name__ == '__main__':
    start()
