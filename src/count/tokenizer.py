import logging
from typing import List

# AllenNLP
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from overrides import overrides

# Huggingface tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.processors import BertProcessing

# Local
import config

logger = logging.getLogger(__name__)


@Tokenizer.register("wikitext-tokenizer")
class WikiTextTokenizer(Tokenizer):
    """An AllenNLP wrapper around a Huggingface tokenizer.
    Registered as a `Tokenizer` with name "wikitext-tokenizer".

    Parameters
    ----------
    model_name : `str`
        The name of the pretrained wordpiece tokenizer to use.

    add_special_tokens : `bool`, optional, (default=`True`)
        If set to `True`, the sequences will be encoded with the special tokens relative to their
        model. In our case, that is the CLS and SEP tokens.

    """  # noqa: E501

    def __init__(
        self,
        tokenizer_path: str,
        add_special_tokens: bool = True,
    ) -> None:
        self.tokenizer = HFTokenizer.from_file(tokenizer_path)
        self._add_special_tokens = add_special_tokens

        # If we're adding special tokens, add the BERT post-processor
        if self._add_special_tokens:
            sep_idx = self.tokenizer.token_to_id(config.SEP)
            cls_idx = self.tokenizer.token_to_id(config.CLS)
            self.tokenizer.post_processor = BertProcessing(
                sep=(config.SEP, sep_idx), cls=(config.CLS, cls_idx)
            )

        self._tokenizer_lowercases = "a" in self.tokenizer.encode("A").tokens

    @overrides
    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[Token]:
        """
        This method only handles a single sentence (or sequence) of text.
        """
        encoded_tokens = self.tokenizer.encode(
            sequence=text,
            add_special_tokens=add_special_tokens,
        )
        # token_ids contains a final list with ids for both regular and special tokens
        token_ids, token_type_ids, special_tokens_mask, token_offsets = (
            encoded_tokens.ids,
            encoded_tokens.type_ids,
            encoded_tokens.special_tokens_mask,
            encoded_tokens.offsets,
        )

        tokens = []
        for token_id, token_type_id, special_token_mask, offsets in zip(
            token_ids, token_type_ids, special_tokens_mask, token_offsets
        ):
            # In `special_tokens_mask`, 1s indicate special tokens and 0s indicate regular tokens.
            # NOTE: in transformers v3.4.0 (and probably older versions) the docstring
            # for `encode_plus` was incorrect as it had the 0s and 1s reversed.
            # https://github.com/huggingface/transformers/pull/7949 fixed this.
            if not self._add_special_tokens and special_token_mask == 1:
                continue

            if offsets is None or offsets[0] >= offsets[1]:
                start = None
                end = None
            else:
                start, end = offsets

            tokens.append(
                Token(
                    text=self.tokenizer.id_to_token(token_id),
                    text_id=token_id,
                    type_id=token_type_id,
                    idx=start,
                    idx_end=end,
                )
            )

        return tokens

    def tokenize_multiple(self, texts: List[str], add_special_tokens: bool = True) -> List[Token]:
        tokenized_sents = [
            self.tokenize(text=t, add_special_tokens=add_special_tokens) for t in texts
        ]
        all_tokens = [token for sent in tokenized_sents for token in sent]
        # if we aren't adding special tokens, we're already done; just return the joined lists
        if not add_special_tokens:
            return all_tokens

        # otherwise, we gotta filter out the cls tokens that we don't want
        # save the cls token
        cls_token = all_tokens[0]
        assert cls_token.text == config.CLS

        return [cls_token] + list(filter(lambda x: x.text != config.CLS, all_tokens))

    def num_special_tokens_for_sequence(self) -> int:
        return self.tokenizer.num_special_tokens_to_add(is_pair=False)

    def num_special_tokens_for_pair(self) -> int:
        return self.tokpenizer.num_special_tokens_to_add(is_pair=True)
