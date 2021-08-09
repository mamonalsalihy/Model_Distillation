import torch
from allennlp.training.metrics import Metric
from sklearn.metrics import matthews_corrcoef


@Metric.register("mcc", exist_ok=True)
class MCC(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, gold_labels, mask=None):
        predictions = torch.argmax(predictions, dim=-1)
        coef = matthews_corrcoef(gold_labels.numpy(), predictions.numpy())
        return torch.tensor(coef, device=predictions.device, dtype=torch.float)
