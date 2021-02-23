from ams.config import logger_factory

logger = logger_factory.create(__name__)


def test_append_sys_path():
    import sys
    from pathlib import Path

    additional_paths = [Path(__file__).parent.parent.parent.absolute()]

    for a in additional_paths:
        add_path = str(a)
        logger.info(add_path)
        if a not in sys.path:
            sys.path.append(add_path)