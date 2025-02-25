import random
import shutil
import time
from os.path import splitext
from pathlib import Path
from typing import Tuple
import mimetypes
from fastapi import UploadFile


def get_file_mime_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or 'application/octet-stream'


def get_unique_name(filename: str) -> str:
    """
    Return a unique name by template [timestamp]_[random number 0..1000][extension]
    """
    _, ext = splitext_(filename)
    ts = int(time.time())
    rnd = random.randint(0, 1000)
    return str(ts) + '_' + str(rnd) + ext


def splitext_(path: str) -> Tuple[str, str]:
    """
    get extensions with several dots
    """
    if len(path.split('.')) > 2:
        return path.split('.')[0], '.' + '.'.join(path.split('.')[-2:])
    return splitext(path)


def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()


def split_s3_path(s3_path: str) -> [str, str]:
    # "s3://bucket/dir1/dir2/dir3/file.png"
    list_dirs = s3_path.split('s3://')[-1].split('/')
    bucket = list_dirs[0]
    file_path = '/'.join(list_dirs[1:])
    return bucket, file_path
