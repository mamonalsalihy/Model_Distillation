"""
The purpose of this file is to make sure all libraries are imported correctly
"""

# Local
from src.count import config
from src.count.data import WikiTextReader
from src.count.decoders.lstm_decoder import LSTMDecoder
from src.count.decoders.transformer_decoder import TransformerDecoder
from src.count.models.simple_transformer import SimpleTransformerLanguageModel
from src.count.models.teacher_student import TeacherStudent
from src.count.models.base_lstm import SimpleLSTMLanguageModel
from src.count.tokenizer import WikiTextTokenizer
from src.utils.misc_utils import get_model_size

# logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)