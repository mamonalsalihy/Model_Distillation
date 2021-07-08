import wget
import os
import zipfile

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.count import config


def download(url, data_dir):
    """
    Note:
        (1) Creates data directory if it doesn't exist
        (2) Downloads Wikitext using wget
        (3) Unzips the contents to the data directory
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("Downloading Wikitext 103 to data directory ... ")

    wget.download(url, out=data_dir)

    file_name = os.path.split(url)[-1]
    file_path = os.path.join(data_dir, file_name)

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)


if __name__ == '__main__':
    raw_character_level = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'
    word_level = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
    download(url=raw_character_level, data_dir=config.DATA_DIR)
    download(url=word_level, data_dir=config.DATA_DIR)
