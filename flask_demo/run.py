import contents
# STL
import sys
from pathlib import Path
import argparse


sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.count import predictor

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer")
    parser.add_argument("archive_dir")
    parser.add_argument("max")
    parser.add_argument("temperature")
    args = parser.parse_args()

    app = contents.create_app(args)
    app.run(debug=True)