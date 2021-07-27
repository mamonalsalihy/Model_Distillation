# STL
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Torch transformer
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary
from allennlp.data import TensorDict

# Models
from allennlp.models import Model

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local

logger = logging.getLogger(__name__)


@Model.register("teacher-student-language-model", exist_ok=True)
class TeacherStudent(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        student: Model,
        teacher: Model,
        hard_label_weight: float = 0.0,
        temperature: float = 3,
        teacher_state_dict: Optional[str] = None,
    ) -> None:
        super().__init__(vocab)

        self.teacher = teacher
        self.student = student

        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size()
        self.temp = temperature
        self.hard_label_weight = hard_label_weight

        self.kldiv = nn.KLDivLoss(reduction="batchmean")

        if teacher_state_dict is not None:
            state_dict = torch.load(teacher_state_dict)
            self.teacher.load_state_dict(state_dict)
        self.teacher.eval()

        logger.info("Number of parameters (student only): %s", self.count_parameters())

    def kl_loss(self, student_logits, teacher_logits):
        # Divide by temperature
        s = student_logits / self.temp
        t = teacher_logits / self.temp

        # Get log probs for predictions & probs for labels
        log_probs = torch.log_softmax(s, dim=-1).view(-1, self.vocab_size)
        soft_labels = torch.softmax(t, dim=-1).view(-1, self.vocab_size)

        # => T^2 * KL(x, y)
        return (self.temp ** 2) * self.kldiv(log_probs, soft_labels)

    def forward(
        self,
        tokens: TensorDict,
        ratio: float,
    ) -> Dict[str, torch.Tensor]:

        student_output = self.student(tokens, ratio)
        student_logits = student_output["logits"]

        # Calculate Loss and Perplexity
        # =============================
        ce_loss = student_output["loss"]
        self.student.perplexity(ce_loss)
        self.student.word_perplexity(ce_loss * ratio)

        if self.training:
            with torch.no_grad():
                teacher_output = self.teacher(tokens, ratio)
                teacher_logits = teacher_output["logits"]
                self.teacher.perplexity(teacher_output['loss'])
            # Calculate KL divergence loss
            kl_loss = self.kl_loss(student_logits, teacher_logits)
            loss = (1 - self.hard_label_weight) * kl_loss + self.hard_label_weight * ce_loss
        else:
            loss = ce_loss

        return {
            "logits": student_logits,
            "loss": loss,
            "student_loss": ce_loss,
        }

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Takes logits from `forward` and computes the corresponding label

        Arguments
        ---------
        output_dict : Dict[str, torch.Tensor]
            Dictionary returned by `forward`. Must contain a key with `logits`.
        Returns
        -------
        Dict[str, torch.Tensor]:
            Same as input dictionary, but with another key `label` indicating the predicted label
        """
        return self.student.make_output_human_readable(output_dict)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "perplexity": self.student.perplexity.get_metric(reset),
            "word_perplexity": self.student.word_perplexity.get_metric(reset),
            "teacher": self.teacher.perplexity.get_metric(reset),
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        teacher_parameters = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        total -= teacher_parameters
        millions = total // 1_000_000
        thousands = (total - millions * 1_000_000) // 1_000
        string = str(millions) + "." + str(thousands) + "M"
        return string

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()
