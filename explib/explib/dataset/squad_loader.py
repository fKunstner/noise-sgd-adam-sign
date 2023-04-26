import tokenize
import datasets
import os
from datasets import load_dataset
from accelerate import Accelerator
from explib import config
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from torch.utils.data.dataloader import DataLoader

import numpy as np

BERT_BASE_PRETRAINED = "bert-base-uncased"
DISTILBERT = "distilbert-base-uncased"


def squad_loader(
    dataset_name,
    batch_size,
    tgt_len,
    doc_stride,
    model_name,
    drop_last=False,
    fake_full_batch_mode=False,
    shuffle=True,
    outliers_filename=None,
):

    split = "train"
    if fake_full_batch_mode:
        seed = np.random.get_state()[1][0] % 13
        start = seed * 6144
        end = (seed + 1) * 6144
        split = "train[{}:{}]".format(start, end)

    if dataset_name == "squad":
        raw_datasets = load_dataset(
            "squad",
            cache_dir=os.path.join(config.get_workspace(), "datasets"),
            split=split,
        )
    else:
        raw_datasets = load_dataset(
            "adversarial_qa",
            "dbert",
            cache_dir=os.path.join(config.get_workspace(), "datasets"),
            split=split,
        )
    column_names = raw_datasets.column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    if model_name == "bert_base_pretrained":
        model_name = BERT_BASE_PRETRAINED
    else:
        model_name = DISTILBERT

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = tokenizer.model_max_length

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # if "train" not in raw_datasets:
    #     raise ValueError("--do_train requires a train dataset")
    train_examples = raw_datasets

    # Create train feature from dataset
    train_dataset = train_examples.map(
        prepare_train_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if dataset_name == "squad":
        eval_examples = load_dataset(
            "squad",
            cache_dir=os.path.join(config.get_workspace(), "datasets"),
            split="validation",
        )
    else:
        eval_examples = load_dataset(
            "adversarial_qa",
            "dbert",
            cache_dir=os.path.join(config.get_workspace(), "datasets"),
            split="validation",
        )

    # Validation Feature Creation
    eval_dataset = eval_examples.map(
        prepare_train_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )

    eval_dataset_valid = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )

    train_dataset_for_eval = train_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset for eval",
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    if outliers_filename is not None:
        outlier_indices = np.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "outliers",
                outliers_filename,
            )
        )
        outlier_indices = np.ndarray.tolist(outlier_indices)
        indices = [
            i for i in range(len(train_dataset)) if str(i) not in outlier_indices
        ]
        train_dataset = train_dataset.select(indices)
        train_dataset_for_eval = train_dataset_for_eval.select(indices)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        collate_fn=data_collator,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    # eval_dataset_for_model = eval_dataset_prepared.remove_columns(
    # ["example_id", "offset_mapping"]
    # )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=batch_size
    )

    # train_dataset_eval_for_model = train_dataset_for_eval.remove_columns(
    #     ["example_id", "offset_mapping"]
    # )

    train_dataloader_for_eval = DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    return (
        train_dataloader,
        train_dataloader_for_eval,
        eval_dataloader,
        eval_dataset_valid,
        eval_examples,
        train_dataset_for_eval,
        train_examples,
        tokenizer,
    )
