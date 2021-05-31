The CoUnT: Compressed Unidirectional Transformers
-------------------------------------------------

- [Overview](#package-overview)
- [Installation](#installation)
- [Training](#training)
- [Hyperparameter optimization](#hyperparameter-optimization)
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
git clone https://github.com/mamonalsalihy/Model_Distillation
```
2. Start up a virtual environment (optional)
```bash
# with pipenv
pipenv --python 3.9  # or other version
pipenv shell

# with virtualenv
python -m venv <name-of-venv>
source ./<name-of-venv>/bin/activate
```
3. Install requirements
 ```bash
 # with pipenv (to be used with pipenv virtual environment)
 pipenv sync
 
 # with pip
 pip install -r requirements.txt
 ```
4. Run setup scripts
```bash
# Downloads and stores WikiText-103 to data directory
python ./scripts/download_wikitext103.py

# train the tokenizer
python ./scripts/train_tokenizer.py

# build the vocabulary
python ./scripts/build_vocabulary.py
```

Training
--------
Choose a configuration script in `configs/`. For example, let's use the hypothetical `example-config.jsonnet`

```bash
allennlp train configs/example-config.jsonnet \     # specify which config file to use
               -s experiments/example-experiment \  # specify a directory for the experiment (saved models, logs, metadata, etc.)
               --include-package src.count          # specify any packages to include code from
```

Hyperparameter optimization
---------------------------
Currently under construction


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
