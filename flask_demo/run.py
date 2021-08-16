import contents
# STL
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.count import predictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer', metavar="tokenizer", type=str, required=True)
    parser.add_argument('-a', '--archive-dir', metavar="archive_dir", type=str, required=True)
    parser.add_argument('-m', '--max', metavar="max", type=int, required=False, default=100)
    parser.add_argument('-b', '--backwards', metavar='backwards', type=bool, required=False, default=False)
    parser.add_argument("--temperature", metavar='temperature', type=float, required=False, default=1.0)
    args = parser.parse_args()

    app = contents.create_app(args)
    app.run(debug=True)
