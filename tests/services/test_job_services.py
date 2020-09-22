# from ams.config import constants
# from ams.services import jobs_services


def test_archive_files():
    pass
    # # Arrange
    # parent = Path(constants.TWITTER_OUTPUT)
    #
    # # Act
    # jobs_services.archive_files(parent=parent)

    # Assert
    #
    # A folder has one output writer only.
    # Pipe reads from moves 'closed' output files from output folder to reading folder.
    # Only closed files are read.
    # Closed files are moved to a staging folder.


def test_foo():
    print('foo')


def test_append_sys_path():
    import sys
    from pathlib import Path

    additional_paths = [Path(__file__).parent.parent.parent.absolute()]

    for a in additional_paths:
        add_path = str(a)
        print(add_path)
        if a not in sys.path:
            sys.path.append(add_path)
