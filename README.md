The CoUnT: Compressed Unidirectional Transformers
-------------------------------------------------

- [Overview](#package-overview)
- [Installation](#installation)
- [Training](#training)
- [Examples](#Examples)
- [Documentation](#documentation)
- [Team](#team)

Overview
--------
This repository contains all the source code and experimentation for *The CoUnT*, a language
model/distillation technique for large language models.

Installation
------------
1. Clone the repository and setup the data directory
```bash
git clone https://github.com/mamonalsalihy/Model-Distillation
cd Model-Distillation
mkdir -p ./data/vocab
```
2. Download the raw WikiText-103 data from their website, and unzip into the data directory.
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip -d ./data/
```
3. Start up a virtual environment (optional)
```bash
# with pipenv
pipenv --python 3.9  # or other version

# with virtualenv
python -m venv <name-of-venv>
source ./<name-of-venv>/bin/activate
```
4. Install requirements
```bash
# with pipenv (to be used with pipenv virtual environment)
pipenv sync

# with pip
pip install -r requirements.txt
```
5. Run setup scripts
```bash
# train the tokenizer
python ./scripts/train_tokenizer.py

# build the vocabulary
python ./scripts/build_vocabulary.py
```

Training
--------
We'll update this section with training scripts as we continue.

Examples
--------
We'll update this section with examples once we have a working model.

Documentation
-------------
This section will contain our project proposal and other important information


Team
----
Contributors:

- Partha Parthasarathy [Mentor]
- [Mamon Alsalihy](https://github.com/mamonalsalihy)
- [Austin King](https://github.com/aukking)
- [Nilay Patel](https://github.com/offendo)
