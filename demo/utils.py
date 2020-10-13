# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# from flask import flash


# def flash_errors(form, category="warning"):
#     """Flash all errors for a form."""
#     for field, errors in form.errors.items():
#         for error in errors:
#             flash("{0} - {1}".format(getattr(form, field).label.text, error), category)


def download_kiwi(url, directory="assets"):
    """Download file at `url`. Extract if needed"""

    def _check_if_downloading(file):
        print("Checking if download in progress", file=sys.stderr)
        size = file.stat().st_size
        time.sleep(1)
        return size != file.stat().st_size

    # get filename from url
    print("Getting filename", file=sys.stderr)
    filename = url.split("/")[-1]

    filepath = Path(directory) / Path(filename)

    target_directory = filepath.parent / filepath.stem
    # if the file isn't there or doesn't have the correct size, download it
    print("Checking if file already downloaded", file=sys.stderr)

    currently_downloading = _check_if_downloading(filepath)

    if not filepath.exists() or (
        filepath.stat().st_size != 1785933237 and not currently_downloading
    ):
        print("Downloading", file=sys.stderr)
        urllib.request.urlretrieve(url, filename=filepath)
        print("Download has finished.", file=sys.stderr)

    print("Extracting {}".format(filepath), file=sys.stderr)
    _maybe_extract(filepath, target_directory=target_directory)

    print("Done Downloading & extracting", file=sys.stderr)
    return filepath


def _maybe_extract(compressed_path, target_directory=None):
    """checks if files have already been extracted and extracts them if not"""

    extension = compressed_path.suffix

    if target_directory is None:
        target_directory = compressed_path.parent / compressed_path.stem

    if not target_directory.exists():
        if "zip" in extension:
            with zipfile.ZipFile(compressed_path, "r") as zipped:
                zipped.extractall(target_directory)
        else:
            print("File type not supported", file=sys.stderr)

    print("Done extracting", file=sys.stderr)
