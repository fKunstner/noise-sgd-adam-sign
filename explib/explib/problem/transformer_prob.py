import csv
import math

import torch

from .. import dataset, model, optim
from .problem import Problem


class TransformerProb(Problem):
    def __init__(self, exp_dict):
        super().__init__(exp_dict)

        init_outputs = dataset.init(
            self.dataset_name,
            self.batch_size,
            self.device,
            extra_params=exp_dict.get("model_args", None),
            drop_last=self.drop_last,
            shuffle=exp_dict.get("shuffle", False),
            outliers_filename=exp_dict.get("outliers_filename", None),
        )

        if len(init_outputs) == 3:
            (
                self.train_dataloader,
                self.valid_dataloader,
                transformer_len,
            ) = init_outputs
        elif len(init_outputs) == 4:
            (
                self.train_dataloader,
                self.valid_dataloader,
                transformer_len,
                self.corpus,
            ) = init_outputs
        else:
            raise ValueError(
                "Don't know how to process this number of dataset.init output values"
            )

        self.model = model.init(
            exp_dict["model"],
            model_args=exp_dict["model_args"],
            transformer_len=transformer_len,
        )

        self.model.to(self.device)

        self.optim = optim.init(
            exp_dict["opt"],
            self.model,
        )

    def calculate_loss(self, data):
        labels_seq_len = data[0][1:]
        X = data[0][0]
        X = X.to(self.device)
        labels, seq_len = labels_seq_len[0], labels_seq_len[1]
        return self.loss_helper(X, labels, seq_len)

    def transformer_xl_loss(self, data, target):
        mems = tuple()

        ret = self.model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        return loss.float().mean().type_as(loss)

    def transformer_encoder_loss(self, data, target, seq_len):
        src_mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)
        output = self.model(data, src_mask)
        output_flat = output.view(-1, self.model.ntoken)
        return self.loss_func(output_flat, target.view(-1))

    @torch.no_grad()
    def eval_loop(self, is_validation=False):
        dataloader = self.valid_dataloader if is_validation else self.train_dataloader
        self.model.eval()
        self.model.to(self.device)
        self.optim.zero_grad()

        epoch_loss = 0.0
        ppl_loss = 0.0
        total_len = 0
        iteration_counter = 0
        accumulation_counter = 0

        for (X, labels, seq_len) in dataloader:
            loss = self.loss_helper(X, labels, seq_len)
            ppl_loss += seq_len * loss
            total_len += seq_len
            if self.grad_accumulate:
                loss = loss / self.accumulate_steps
            epoch_loss += loss
            iteration_counter += 1
            if (
                not self.grad_accumulate
                or iteration_counter % self.accumulate_steps == 0
            ):
                accumulation_counter += 1
            if (
                self.fake_full_batch_mode
                and accumulation_counter == 1
                and not is_validation
            ):
                break
            if self.dummy_run:
                accumulation_counter = 1
                break

        results = {}
        ppl_loss = ppl_loss / total_len
        try:
            ppl = math.exp(ppl_loss)
        except OverflowError:
            ppl = float("inf")

        if is_validation:
            results["valid_ppl"] = ppl
        else:
            results["train_ppl"] = ppl
            results["training_loss"] = epoch_loss / max(accumulation_counter, 1)

        return results

    def loss_helper(self, X, labels, seq_len):
        if self.model_name in [model.TRANSFORMER_XL, model.TRANSFORMER_XL_DET]:
            loss = self.transformer_xl_loss(X, labels)
        elif self.model_name in [
            model.TRANSFORMER_ENCODER,
            model.TRANSFORMER_ENCODER_DET,
        ]:
            loss = self.transformer_encoder_loss(X, labels, seq_len)
        else:
            raise Exception("Transformer not supported!")
        return loss

    def get_outliers_helper(self, final_noise_norms):
        with open(
            self.save_path + "/noise/outliers_{}.csv".format(self.exp_uuid),
            "w",
            encoding="utf-8",
        ) as fw:
            writer = csv.writer(fw, delimiter=",")
            writer.writerow(["index", "norm", "text"])
            rows = []
            for (step, *data) in enumerate(self.train_dataloader):
                noise = final_noise_norms[step]
                X = data[0][0]
                X = X.to(self.device)
                sentences = self.corpus.vocab.convert_to_sent_from_tensor(X)
                row = [step, noise, sentences]
                rows.append(row)
            rows = sorted(rows, key=lambda x: x[1], reverse=True)
            writer.writerows(rows)
