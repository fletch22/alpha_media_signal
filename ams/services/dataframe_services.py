from enum import Enum
from pathlib import Path

from pyspark.sql import DataFrame

from ams.config import logger_factory
from ams.services import file_services

logger = logger_factory.create(__name__)

class PersistedDataFrameTypes(Enum):
    CSV = "CSV"
    PARQUET = "PARQUET"
    ORC = "ORC"


def persist_dataframe(df: DataFrame, output_drop_folder_path: Path, prefix: str, num_output_files: int = -1,
                      file_type: PersistedDataFrameTypes = PersistedDataFrameTypes.PARQUET):
    output_folder_path = file_services.create_unique_folder_name(str(output_drop_folder_path), prefix=prefix, ensure_exists=False)

    if num_output_files > 0:
        df = df.repartition(num_output_files)

    if file_type == PersistedDataFrameTypes.PARQUET:
        df.write.save(str(output_folder_path), format="parquet")
    elif file_type == PersistedDataFrameTypes.ORC:
        df.write.save(str(output_folder_path), format="orc")
    elif file_type == PersistedDataFrameTypes.CSV:
        df.write.save(str(output_folder_path), format="csv")

    logger.info(f" Output: {str(output_folder_path)}")

    return output_folder_path