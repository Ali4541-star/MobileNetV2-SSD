import torchmetrics as tm
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import roc_curve
from torchmetrics.classification import BinaryROC

class EER(tm.Metric):

    def __init__(self, pos_label=1):
        super(EER, self).__init__()
        self.pos_label = pos_label

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        
        preds = preds[:, self.pos_label].to("cpu")
        targets = torch.argmax(targets, dim=1).to("cpu")
        fpr, tpr, _ = roc_curve(
            targets.detach().numpy(),
            preds.detach().numpy(),
            pos_label=self.pos_label)

        eer = 1 - tpr[np.argmin(np.abs(tpr - (1 - fpr)))]

        self.thresh = torch.tensor(eer)

    def compute(self) -> torch.Tensor:
        return self.thresh


class EER2(tm.Metric):

    def __init__(self, pos_label=1):
        super(EER, self).__init__()
        self.pos_label = pos_label
        self.roc = BinaryROC()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        
        preds = preds[:, self.pos_label].to("cpu")
        targets = torch.argmax(targets, dim=1).to("cpu")
        fpr, tpr, _ = self.roc(preds, targets, pos_label=self.pos_label, drop_intermediate=True)

        eer = 1 - tpr[np.argmin(np.abs(tpr - (1 - fpr)))]
        print(f"EER is {eer}")

        self.thresh = torch.tensor(eer)

    def compute(self) -> torch.Tensor:
        return self.thresh
