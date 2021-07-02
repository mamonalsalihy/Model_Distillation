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
        teacher_state_dict: Optional[str] = None,
    ) -> None:
        super().__init__(vocab)

        self.teacher = teacher
        self.student = student

        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size()

        self.kldiv = nn.KLDivLoss(reduction="mean")

        if teacher_state_dict is not None:
            state_dict = torch.load(teacher_state_dict)
            self.teacher.load_state_dict(state_dict)
        self.teacher.eval()

        logger.info("Number of parameters (student only): %s", self.count_parameters())

    def forward(
        self,
        tokens: TensorDict,
    ) -> Dict[str, torch.Tensor]:

        student_output = self.student(tokens)
        student_logits = student_output["logits"]
        student_log_probs = torch.log_softmax(student_output["logits"], dim=-1)
        student_loss = student_output["loss"]

        preds = student_log_probs.view(-1, self.vocab_size)

        if self.training:
            with torch.no_grad():
                teacher_output = self.teacher(tokens)
                teacher_logits = teacher_output["logits"]
                teacher_probs = torch.softmax(teacher_logits, dim=-1)

                soft_labels = teacher_probs.view(-1, self.vocab_size)
                teacher_loss = teacher_output["loss"]

            loss = self.kldiv(preds, soft_labels)
            # logger.info("Teacher Loss: %s", teacher_loss.item())
            # logger.info("KL Div Loss %s", loss.item())
        else:
            loss = student_loss
            # logger.info("Student (CE) Loss: %s", student_loss.item())

        return {
            "logits": student_logits,
            "loss": loss,
            "log_probs": student_log_probs,
            "student_loss": student_loss,
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
        return {"perplexity": self.student.metric.get_metric(reset)}

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
