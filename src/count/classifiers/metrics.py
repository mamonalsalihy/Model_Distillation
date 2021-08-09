from typing import Optional
import numpy as np
import torch
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric



@Metric.register("mcc", exist_ok=True)
class MCC(Metric):
    def __init__(self) -> None:
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predictions`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predictions`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        if gold_labels.size() != predictions.size():
            raise ValueError(
                f"gold_labels must have shape == predictions.size() but "
                f"found tensor of shape: {gold_labels.size()}"
            )
        if mask is not None and mask.size() != predictions.size():
            raise ValueError(
                f"mask must have shape == predictions.size() but "
                f"found tensor of shape: {mask.size()}"
            )

        batch_size = predictions.size(0)

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

            # We want to skip predictions that are completely masked;
            # so we'll keep predictions that aren't.
            keep = mask.view(batch_size, -1).max(dim=1)[0]
        else:
            keep = torch.ones(batch_size, device=predictions.device).bool()

        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)

        # At this point, predictions is (batch_size, rest_of_dims_combined),
        # so .eq -> .prod will be 1 if every element of the instance prediction is correct
        # and 0 if at least one element of the instance prediction is wrong.
        # Because of how we're handling masking, masked positions are automatically "correct".
        positive_preds = (predictions == 1)
        positive_trues = (gold_labels == 1)

        negative_preds = (predictions == 0)
        negative_trues = (gold_labels == 0)

        # calculate the statistics split
        tp = ((positive_preds & positive_trues) * keep).sum()
        tn = ((negative_preds & negative_trues) * keep).sum()
        fp = ((positive_preds & negative_trues) * keep).sum()
        fn = ((negative_preds & positive_trues) * keep).sum()
        total_count = keep.sum()

        self.tp += dist_reduce_sum(tp).item()
        self.tn += dist_reduce_sum(tn).item()
        self.fp += dist_reduce_sum(fp).item()
        self.fn += dist_reduce_sum(fn).item()
        self.total_count += dist_reduce_sum(total_count).item()

    def get_metric(self, reset: bool):
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = np.sqrt(((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)))
        if denominator > 1e-12:
            matthews_corr_coef = numerator / denominator
        else:
            matthews_corr_coef = 0.0
        if reset:
            self.reset()

        return matthews_corr_coef

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0.0
        self.total_count = 0.0
