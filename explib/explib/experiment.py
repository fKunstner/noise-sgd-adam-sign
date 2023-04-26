import time
import torch
import math
import os
import random
import datetime
from pathlib import Path
import numpy as np
from explib import config
from explib.expmaker.experiment_defs import make_wuuid, exp_dict_to_str

from . import logging, problem


class Experiment:
    def __init__(
        self,
        exp_dict,
        slug,
        exp_uuid,
        disable_wandb,
        gpu,
        dummy_run=False,
    ):
        """Create an experiment"""
        self.seed = exp_dict["seed"]
        self.apply_seed()
        cpu_only = exp_dict.get("cpu_only", False)
        self.device = gpu if torch.cuda.is_available() and not cpu_only else "cpu"

        self.data_logger = logging.init_logging_for_exp(
            slug,
            exp_uuid,
            exp_dict,
            disable_wandb,
            additional_config={
                "device": self.device,
                "uuid": exp_uuid,
                "wuuid": make_wuuid(exp_dict),
                "exp_dict_str": exp_dict_to_str(exp_dict),
            },
        )
        logging.info(f"Creating experiment. Received experiment dictionnary {exp_dict}")

        self.max_epoch = exp_dict["max_epoch"]
        self.fake_full_batch_mode = exp_dict.get("fake_full_batch_mode", False)
        self.drop_last = exp_dict.get("drop_last", False)
        self.trained_norms = exp_dict.get("trained_norms", False)
        self.init_noise_norm = exp_dict.get("init_noise_norm", False)

        exp_dict["device"] = self.device
        exp_dict["trained_norms"] = self.trained_norms
        exp_dict["exp_uuid"] = exp_uuid
        exp_dict["dummy_run"] = dummy_run

        self.problem = problem.init(exp_dict)

        self.save_path = os.path.join(
            config.get_workspace(), exp_dict["dataset"], exp_uuid
        )
        self.model_dir = os.path.join(self.save_path, "model")
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.model_file = os.path.join(self.model_dir, "model.pt")
        if not os.path.isfile(self.model_file):
            self.model_file = os.path.join(
                self.model_dir, "model_{}.pt".format(self.max_epoch)
            )

    def apply_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def run(self):
        """Run the experiment"""
        start_time = time.time()
        # TODO: Allow to continue training from a nonzero epoch?
        logging.info("Starting experiment run")

        starting_epoch = 0

        if self.init_noise_norm or self.trained_norms:
            logging.info("Initial run to compute noise norms")
            r_start = time.time()
            self.problem.calc_norms(norm_epoch=0)
            r_end = time.time()
            logging.info(f"Norm computation time: {r_end - r_start}")

        logging.info("Initial evaluation")
        r_start = time.time()
        initial_train_metrics = self.problem.eval_loop(is_validation=False)
        print(initial_train_metrics)
        initial_valid_metrics = self.problem.eval_loop(is_validation=True)
        self.data_logger(initial_valid_metrics, commit=False)
        self.data_logger(initial_train_metrics)
        r_end = time.time()
        logging.info(f"Initial evaluation time: {r_end - r_start}")

        epochs_to_compute_noise_norm = [
            1,
            int(self.max_epoch * 0.1),
            int(self.max_epoch * 0.25),
            int(self.max_epoch * 0.5),
            int(self.max_epoch * 0.75),
        ]

        for epoch in range(starting_epoch, self.max_epoch):
            logging.info(f"Epoch {epoch}/{self.max_epoch}")

            if self.trained_norms and epoch in epochs_to_compute_noise_norm:
                logging.info(f"Computing noise norms at epoch {epoch}")
                self.problem.calculate_noise_norm(epoch=epoch)

            # run training loop
            epoch_begin_time = time.time()
            train_loss, func_vals, gnorms_1, gnorms_2 = self.problem.train_loop()
            epoch_end_time = time.time()
            epoch_training_time = epoch_end_time - epoch_begin_time

            if math.isnan(train_loss) or math.isinf(train_loss):
                if math.isnan(train_loss):
                    self.data_logger({"training_error": "nan"})
                else:
                    self.data_logger({"training_error": "inf"})
                break

            # run eval loop
            logging.info(f"Running evaluation")
            train_metrics = self.problem.eval_loop(is_validation=False)
            self.data_logger(train_metrics, commit=False)
            valid_metrics = self.problem.eval_loop(is_validation=True)
            self.data_logger(valid_metrics, commit=False)

            self.data_logger(
                {
                    "epoch": epoch,
                    "average_training_loss": train_loss,
                    "function_values": func_vals,
                    "norm_squared_gradients": gnorms_2,
                    "norm_squared_gradients_l1": gnorms_1,
                    "epoch_training_time": epoch_training_time,
                }
            )

        # save model
        if not os.path.isfile(self.model_file):
            with open(self.model_file, "wb") as f:
                torch.save(self.problem.model.state_dict(), f)
            if self.trained_norms:
                self.problem.calculate_noise_norm(epoch=self.max_epoch)

        end_time = time.time()
        self.data_logger(
            {"exp_runtime": str(datetime.timedelta(seconds=end_time - start_time))}
        )
