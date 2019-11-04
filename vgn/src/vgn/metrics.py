from ignite.metrics import Metric
import torch


class Accuracy(Metric):
    def reset(self):
        self.num_correct = 0
        self.num_examples = 0

    def update(self, out):
        pred, target, mask = out["pred_quality"], out["target_quality"], out["mask"]
        correct = torch.eq(torch.round(pred), target) * mask

        self.num_correct += torch.sum(correct).item()
        self.num_examples += torch.sum(mask).item()

    def compute(self):
        return self.num_correct / self.num_examples
