import argparse
import os
from flask import Flask, render_template, request
# STL
import sys
from pathlib import Path
import argparse

# Torch
import torch

# AllenNLP
from allennlp.models import Model
from allennlp.common import Params
from allennlp.data.fields import Field, TextField, TensorField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from tokenizers import Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Local
from src.count.data import *
from src.count import config
from src.count.tokenizer import WikiTextTokenizer
from src.count.decoders.transformer_decoder import TransformerDecoder
from src.count.models.teacher_student import TeacherStudent
from src.count.models.simple_transformer import SimpleTransformerLanguageModel
from src.count.models.dual_directional import DualDirectionalModel
from src.count.predictor import LMInference


def create_app(args, test_config=None):
    print("Creating flask app ... ")
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    # We can create a secrete key using the following
    # import secrets secrets.token_hex(16)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'Model_Distillation.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    inf = init_model(args)

    @app.route('/analytics')
    def analytics():
        print("Rendering analystics page")
        return render_template('analytics.html')

    # Accesses the root URL
    @app.route('/', methods=["POST", "GET"])
    def root():
        if request.method == "POST":
            # baseline models
            # ===============
            if request.form.get('submit_button') == "16-layer-baseline":
                args.archive_dir = '/data/users/nilay/the-count/saved-experiments/16-layer'
            if request.form.get('submit_button') == '10-layer-baseline':
                args.archive_dir = '/data/users/nilay/the-count/saved-experiments/10-layer'
            if request.form.get('submit_button') == '6-layer-baseline':
                args.archive_dir = '/data/users/nilay/the-count/saved-experiments/6-layer'

            # knowledge distillation models
            # =============================
            if request.form.get('submit_button') == '16-to-10':
                args.archive_dir = '/data/users/malsalih/Model_Distillation/experiments/16_to_10'
            if request.form.get('submit_button') == '16-to-6':
                args.archive_dir = '/data/users/malsalih/Model_Distillation/experiments/16_to_6'
            if request.form.get('submit_button') == '16-to-4':
                args.archive_dir = '/data/users/malsalih/Model_Distillation/experiments/16_to_4'

            # bidirectional knowledge distillation models
            # ===========================================
            if request.form.get('submit_button') == '2x6-to-6':
                args.archive_dir = '/data/users/aukking/Model_Distillation/saved-experiments/2x6-6-raw2'

            inf = init_model(args)
            input_query = request.form.get("query")
            if input_query == '':
                return render_template('index.html', query="Please input a query")
            if input_query is None:
                return render_template('index.html', query='Model was switched!')
            count_response = inf.speak(input_query, int(args.max), float(args.temperature))
            return render_template('index.html', query=count_response)
        else:
            return render_template("index.html")

    return app


def init_model(args):
    tokenizer = Tokenizer.from_file(args.tokenizer)
    params = Params.from_file(Path(args.archive_dir) / "config.json")
    model = Model.load(params, serialization_dir=args.archive_dir)
    model = model.student if isinstance(model, TeacherStudent) else model
    inf = LMInference(model, tokenizer, args.backwards)
    return inf


if __name__ == '__main__':
    create_app()
