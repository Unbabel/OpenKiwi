from pathlib import Path


def file_path(name):
    package_directory = Path(__file__).parent
    return package_directory / name
