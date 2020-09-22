from pathlib import Path

from pyspark.sql import DataFrame

from ams.services import file_services


def persist_dataframe_as_csv(df: DataFrame, output_drop_folder_path: Path, prefix: str, num_output_files: int = -1):
    output_folder_path = file_services.create_unique_folder_name(str(output_drop_folder_path), prefix=prefix, ensure_exists=False)

    if num_output_files > 0:
        df = df.repartition(num_output_files)

    df.write.option("header", True).option("quoteAll", True).csv(str(output_folder_path))

    print(str(output_folder_path))
