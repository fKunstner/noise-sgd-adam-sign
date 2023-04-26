import torch
import torch.nn.functional as F

from .. import dataset, model, optim
from .problem import Problem


class ImageProb(Problem):
    def __init__(self, exp_dict):
        super().__init__(exp_dict)

        self.train_dataloader, self.valid_dataloader = dataset.init(
            self.dataset_name,
            self.batch_size,
            self.device,
            drop_last=self.drop_last,
            shuffle=(
                False if self.fake_full_batch_mode else exp_dict.get("shuffle", True)
            ),
            fake_full_batch_mode=self.fake_full_batch_mode,
        )

        if "model_args" not in exp_dict and exp_dict["dataset"] == "mnist":
            exp_dict["model_args"] = {}
            exp_dict["model_args"]["in_channels"] = 1

        self.model = model.init(
            exp_dict["model"],
            model_args=exp_dict["model_args"] if "model_args" in exp_dict else None,
        )

        self.model.to(self.device)

        self.optim = optim.init(
            exp_dict["opt"],
            self.model,
        )

    def calculate_loss(self, data):
        labels = data[0][1:][0].to(self.device).float()
        X = data[0][0]
        X = X.to(self.device)
        y = self.model(X.float())
        return self.loss_func(y, labels)

    @torch.no_grad()
    def eval_loop(self, is_validation=False):
        dataloader = self.valid_dataloader if is_validation else self.train_dataloader

        correct = torch.zeros(1).to(self.device)
        epoch_loss = 0
        images_counter = 0
        accumulation_counter = 0
        iteration_counter = 0

        self.model.eval()
        self.model.to(self.device)

        for (X, labels) in dataloader:
            X = X.to(self.device)
            labels = labels.to(self.device).float()

            y = self.model(X)
            predicted = F.softmax(y, dim=1)
            _, predicted_labels = torch.max(predicted, 1)

            images_counter += labels.size(0)
            correct += (predicted_labels == labels).sum()

            loss = self.loss_func(y, labels)
            if self.grad_accumulate:
                loss = loss / self.accumulate_steps
            epoch_loss += loss.item()
            iteration_counter += 1
            if (
                not self.grad_accumulate
                or iteration_counter % self.accumulate_steps == 0
            ):
                accumulation_counter += 1
            if self.dummy_run:
                accumulation_counter = 1
                break

        results = {}
        accuracy = correct.item() / images_counter

        if is_validation:
            results["valid_accuracy"] = accuracy
        else:
            results["train_accuracy"] = accuracy
            results["training_loss"] = epoch_loss / max(accumulation_counter, 1)

        return results
