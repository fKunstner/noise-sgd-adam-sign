from .problem import Problem
from .. import dataset, model, optim
import torch


class SimpleProb(Problem):
    def __init__(self, exp_dict):
        super().__init__(exp_dict)

        self.train_dataloader, self.valid_dataloader = dataset.init(
            self.dataset_name,
            self.batch_size,
            self.device,
            drop_last=self.drop_last,
            shuffle=exp_dict.get("shuffle", True),
        )

        features_dim = next(iter(self.train_dataloader))[0].shape[1]
        self.model = model.init(
            exp_dict["model"],
            features_dim=features_dim,
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
    def eval_loss(self, is_validation=False):
        dataloader = self.valid_dataloader if is_validation else self.train_dataloader
        self.model.eval()
        self.model.to(self.device)

        epoch_loss = 0.0
        iteration_counter = 0
        accumulation_counter = 0

        for (X, labels) in dataloader:
            labels = labels.to(self.device).float()
            y = self.model(X.float())
            loss = self.loss_func(y, labels)
            if self.grad_accumulate:
                loss = loss / self.accumulate_steps
            iteration_counter += 1
            if (
                not self.grad_accumulate
                or iteration_counter % self.accumulate_steps == 0
            ):
                accumulation_counter += 1

            epoch_loss += loss.item()

            if self.fake_full_batch_mode and accumulation_counter == 1:
                break
            if self.dummy_run:
                accumulation_counter = 1
                break

        epoch_loss = epoch_loss / max(accumulation_counter, 1)
        results = {}
        if is_validation:
            results["valid_mse"] = epoch_loss
        else:
            results["train_mse"] = epoch_loss
            results["training_loss"] = epoch_loss

        return results
