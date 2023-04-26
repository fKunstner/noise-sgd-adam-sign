import torch
from torch.nn.utils import parameters_to_vector as p2v
from abc import ABCMeta, abstractmethod
from explib import config
from ..util import get_grads, enable_running_stats, disable_running_stats
import os
import numpy as np
from pathlib import Path
import csv
from ..dataset import *


class Problem(metaclass=ABCMeta):
    def __init__(self, exp_dict):
        self.model_name = exp_dict["model"]
        self.batch_size = exp_dict["batch_size"]
        self.seed = exp_dict["seed"]
        self.fake_full_batch_mode = (
            "fake_full_batch_mode" in exp_dict and exp_dict["fake_full_batch_mode"]
        )
        self.drop_last = "drop_last" in exp_dict and exp_dict["drop_last"]
        self.device = exp_dict["device"]
        self.dataset_name = exp_dict["dataset"]
        self.optim_name = exp_dict["opt"]["name"]
        self.init_noise_norm = (
            "init_noise_norm" in exp_dict and exp_dict["init_noise_norm"]
        )
        self.save_path = os.path.join(
            config.get_workspace(), exp_dict["dataset"], exp_dict["exp_uuid"]
        )
        self.trained_norms = exp_dict["trained_norms"]
        self.save_norm_samples = (
            "save_norm_samples" in exp_dict and exp_dict["save_norm_samples"]
        )
        self.dummy_run = exp_dict["dummy_run"]

        if "loss_func" in exp_dict:
            self.loss_func = self.get_loss_function(exp_dict["loss_func"])

        # Gradient accumulation for noise norm calculation
        if "accumulate_steps" in exp_dict:
            self.accumulate_steps = exp_dict["accumulate_steps"]
            self.grad_accumulate = True
        else:
            self.accumulate_steps = 1
            self.grad_accumulate = False
        self.exp_uuid = exp_dict["exp_uuid"]

    @abstractmethod
    def calculate_loss(self, data):
        pass

    @abstractmethod
    def eval_loop(self, is_validation=False):
        pass

    def train_loop(self):
        """Train for one epoch"""
        self.model.train()
        self.model.to(self.device)
        self.optim.zero_grad()

        epoch_loss = 0.0
        iteration_counter = 0
        accumulation_counter = 0

        fvals, gnorms_1, gnorms_2 = [], [], []

        for (step, *data) in enumerate(self.train_dataloader):
            loss = self.calculate_loss(data)
            if self.grad_accumulate:
                loss = loss / self.accumulate_steps
            loss.backward()
            iteration_counter += 1
            if (
                not self.grad_accumulate
                or iteration_counter % self.accumulate_steps == 0
            ):
                fvals.append(loss.item())
                gnorms_1.append(grad_norm_squared(self.optim, p=1).item())
                gnorms_2.append(grad_norm_squared(self.optim, p=2).item())

                self.optim.step()

                self.optim.zero_grad()
                accumulation_counter += 1

            epoch_loss += loss.item()

            if self.fake_full_batch_mode and accumulation_counter == 1:
                break
            if self.dummy_run:
                accumulation_counter = 1
                break

        epoch_loss = epoch_loss / accumulation_counter
        return epoch_loss, fvals, gnorms_1, gnorms_2

    def calc_norms(self, norm_epoch, mean_grad=None):
        """
        Calculate noise norms. If mean_grad is None, will calculate
        the gradient mean first. If not None, will calculate the norms and save them
        """
        self.model.train()
        self.model.to(self.device)
        self.optim.zero_grad()
        iteration_counter = 0
        accumulation_counter = 0

        calc_total_grad = mean_grad is None

        self.model.apply(disable_running_stats)
        if calc_total_grad:
            logs_path = os.path.join(self.save_path, "noise")
            Path(logs_path).mkdir(parents=True, exist_ok=True)
            grads = None
        else:
            # calc norms
            noise_norms = []
        for (step, *data) in enumerate(self.train_dataloader):
            loss = self.calculate_loss(data)
            if self.grad_accumulate:
                loss = loss / self.accumulate_steps
            loss.backward()
            iteration_counter += 1
            if (
                not self.grad_accumulate
                or iteration_counter % self.accumulate_steps == 0
            ):
                if calc_total_grad:
                    grad = get_grads(self.model).cpu()
                    grads = grad if grads is None else grads + grad
                else:
                    # calc norms
                    grad = get_grads(self.model).cpu()
                    noise_norm = (grad - mean_grad).norm().item() ** 2
                    noise_norms.append(noise_norm)
                self.optim.zero_grad()
                accumulation_counter += 1

            if self.fake_full_batch_mode and accumulation_counter == 1:
                break
            if self.dummy_run:
                break
        if calc_total_grad:
            torch.save(
                grads,
                self.save_path
                + "/noise/grad_{}_{}".format(accumulation_counter, norm_epoch),
            )
            self.calc_norms(
                norm_epoch=norm_epoch, mean_grad=grads / accumulation_counter
            )
            self.model.apply(enable_running_stats)
            return
        else:
            # calc norms
            final_noise_norms = np.asarray(noise_norms)
            np.save(
                self.save_path
                + "/noise/norm_{}_{}_{}_{}_{}_{}".format(
                    self.model_name,
                    self.dataset_name,
                    self.batch_size * self.accumulate_steps,
                    self.seed,
                    self.optim_name,
                    norm_epoch,
                ),
                final_noise_norms,
            )
            if self.save_norm_samples:
                if self.dataset_name in [PTB, WIKITEXT2, SQUAD]:
                    self.get_outliers_helper(final_noise_norms)

    def logLoss(self, predicted, actual):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(predicted, actual.long())

    def get_loss_function(self, function_name):
        if function_name == "logloss":
            criterion = self.logLoss
        elif function_name == "mse":
            criterion = torch.nn.MSELoss()
        else:
            raise Exception("unsupported loss function: " + function_name)

        return criterion


@torch.no_grad()
def grad_norm_squared(optim, p=2):
    v = p2v(
        [
            p.grad
            for group in optim.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
    )
    return v.norm(p=p) ** 2
