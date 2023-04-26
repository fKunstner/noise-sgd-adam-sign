import csv

import torch
from accelerate import Accelerator
from datasets import load_metric

from .. import dataset, model, optim
from .problem import Problem


class BertSquadProb(Problem):
    def __init__(self, exp_dict):
        super().__init__(exp_dict)

        (
            self.train_dataloader,
            self.train_dataloader_for_eval,
            self.valid_dataloader,
            self.valid_dataset,
            self.valid_examples,
            self.train_dataset,
            self.train_examples,
            self.tokenizer,
        ) = dataset.init(
            exp_dict["dataset"],
            self.batch_size,
            self.device,
            extra_params={**exp_dict["model_args"], "model_name": self.model_name},
            drop_last=self.drop_last,
            fake_full_batch_mode=self.fake_full_batch_mode,
            shuffle=False if self.save_norm_samples else exp_dict.get("shuffle", True),
            outliers_filename=exp_dict.get("outliers_filename", None),
        )

        self.model = model.init(
            exp_dict["model"],
            model_args=exp_dict["model_args"],
        )

        self.model.to(self.device)

        self.optim = optim.init(
            exp_dict["opt"],
            self.model,
        )

        self.accelerator = Accelerator()
        (
            self.model,
            self.optim,
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.optim, self.train_dataloader, self.valid_dataloader
        )
        self.train_dataloader_for_eval = self.accelerator.prepare(
            self.train_dataloader_for_eval
        )

        self.metric = load_metric("squad")

    def calculate_loss(self, data):
        return self.model(**data[0]).loss

    @torch.no_grad()
    def eval_loop(self, is_validation=False):

        if self.dummy_run:
            results = {}
            if not is_validation:
                results["training_loss"] = float("nan")
                results["train_exact_f1"] = float("nan")
                results["train_exact_match"] = float("nan")
            else:
                results["valid_exact_match"] = float("nan")
                results["valid_exact_f1"] = float("nan")
            return results

        if is_validation:
            dataloader = self.valid_dataloader
            dataset = self.valid_dataset
            examples = self.valid_examples
        else:
            dataloader = self.train_dataloader_for_eval
            dataset = self.train_dataset
            examples = self.train_examples

        # TODO: merge the loss and metrics calculations here into one loop

        # loss = model.bert_base_pretrained.eval_loss(
        # self, self.model, self.train_dataloader
        # )
        metrics, loss = model.bert_base_pretrained.evaluate(
            self,
            self.model,
            dataloader,
            self.accelerator,
            dataset,
            examples,
            self.metric,
        )
        results = {}

        if not is_validation:
            results["training_loss"] = loss
            results["train_exact_f1"] = metrics["f1"]
            results["train_exact_match"] = metrics["exact_match"]
        else:
            results["valid_loss"] = loss
            results["valid_exact_match"] = metrics["exact_match"]
            results["valid_exact_f1"] = metrics["f1"]

        return results

    def get_outliers_helper(self, final_noise_norms):
        with open(
            self.save_path + "/noise/outliers_{}.csv".format(self.exp_uuid),
            "w",
        ) as fw:
            writer = csv.writer(fw, delimiter=",")
            writer.writerow(["index", "norm", "question", "context"])
            rows = []
            for (step, *data) in enumerate(self.train_dataloader):
                noise = final_noise_norms[step]
                input_ids = data[0]["input_ids"].tolist()
                questions, contexts = self.norm_helper(input_ids)
                row = [step, noise, questions, contexts]
                rows.append(row)
            rows = sorted(rows, key=lambda x: x[1], reverse=True)
            writer.writerows(rows)

    def norm_helper(self, input_ids):
        decoded = self.tokenizer.batch_decode(input_ids)
        questions, contexts = [], []
        for x in decoded:
            x = x.split("[SEP]")
            questions.append(x[0])
            contexts.append(x[1])
        return questions, contexts
