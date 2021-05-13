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
1. Clone the repository
```bash
git clone https://github.com/mamonalsalihy/Model-Distillation
```
2. Download and store WikiText-103 in the data directory.
```bash
python ./scripts/download_wikitext103.py
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
