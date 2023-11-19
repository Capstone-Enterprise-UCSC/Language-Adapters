import argparse
import json
import logging
import math
import os
import random

import datasets
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
from datasets import Dataset, DatasetDict, concatenate_datasets
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    MODEL_MAPPING,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import transformers.adapters.composition as ac


# this is from adapter-hub to check the versions etc
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# end of adapter-hub code

# Parsing input arguments
def parse_args():

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="The root path of the dataset",
    ),

    parser.add_argument(
        "--predict_with_generate",
        type=bool,
        default=True,
        help="",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help=(
            "Whether to pad all samples to model maximum sentence "
            "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    # parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
    # parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
    
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )

    parser.add_argument(
        "--src_lang",
        type=str,
        default=None,
        help="Language on the encoder side of the adapter",
        choices=["ha","en","ig","sw","yo"],
    )

    parser.add_argument(
        "--tgt_lang",
        type=str,
        default=None,
        help="Language on the decoder side of the adapter",
        choices=["ha","en","ig","sw","yo"],
    )
    
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    return args

# Data
def get_tokenized_dataset(data_path, src_lang, tgt_lang, tokenizer, args):

    dataset_name = f'{data_path}/{src_lang}-{tgt_lang}'

    train_dataset = Dataset.from_pandas(pd.read_csv(f'{dataset_name}/cleaned_train.csv'))
    validation_dataset = Dataset.from_pandas(pd.read_csv(f'{dataset_name}/cleaned_dev.csv'))

    new_feature_train_dataset_src_lang = [src_lang] * len(train_dataset)
    train_dataset = train_dataset.add_column("src_lang", new_feature_train_dataset_src_lang)
    new_feature_validation_dataset_src_lang = [src_lang] * len(validation_dataset)
    validation_dataset = validation_dataset.add_column("src_lang", new_feature_validation_dataset_src_lang)

    new_feature_train_dataset_tgt_lang = [tgt_lang] * len(train_dataset)
    train_dataset = train_dataset.add_column("tgt_lang", new_feature_train_dataset_tgt_lang)
    new_feature_validation_dataset_tgt_lang = [tgt_lang] * len(validation_dataset)
    validation_dataset = validation_dataset.add_column("tgt_lang", new_feature_validation_dataset_tgt_lang)

    dataset = DatasetDict({'train': train_dataset, 'validation': validation_dataset})

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = [example for example in examples[src_lang]]
        targets = [example for example in examples[tgt_lang]]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True, padding=padding)

        labels = tokenizer(text_target=targets, max_length=args.max_source_length, truncation=True, padding=padding)

        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=[src_lang, tgt_lang])

    return tokenized_dataset


def get_processed_dataset(data_path, src_langs, tgt_langs, tokenizer, args):
    all_tokenized_datasets = []
    for _src_lang in src_langs:
        for _tgt_lang in tgt_langs:
            tokenized_dataset = get_tokenized_dataset(
                data_path=data_path,
                src_lang=_src_lang,
                tgt_lang=_tgt_lang,
                tokenizer=tokenizer,
                args=args
            )
            all_tokenized_datasets.append(tokenized_dataset)
    
    all_tokenized_train_dataset = [_tokenized_dataset['train'] for _tokenized_dataset in all_tokenized_datasets]
    all_tokenized_validation_dataset = [_tokenized_dataset['validation'] for _tokenized_dataset in all_tokenized_datasets]

    processed_dataset = DatasetDict({'train': concatenate_datasets(all_tokenized_train_dataset), 'validation': concatenate_datasets(all_tokenized_validation_dataset)})

    return processed_dataset

# Custom Data Collator for Language Mixing
class LanguagePairBatchCollator:
    def __init__(self, dataset, batch_size, all_language_pairs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.all_language_pairs = all_language_pairs

    def __call__(self, batch):
        batched_data = {}

        if not batch:
            return {}
        
        num_samples = len(self.dataset)
        indices = list(range(num_samples))
        random.shuffle(indices)

        shuffled_data = [self.dataset[i] for i in indices]

        for _src_lang, _tgt_lang in self.all_language_pairs:
            filtered_data = [example for example in shuffled_data if example['src_lang'] == _src_lang and example['tgt_lang'] == _tgt_lang]
            filtered_data_len = len(filtered_data)
            batched_data[(_src_lang, _tgt_lang)] = [filtered_data[i:i+self.batch_size] for i in range(0, filtered_data_len, self.batch_size) if i+self.batch_size < filtered_data_len]

        flattened_batched_data = [_value for _, _values in batched_data.items() for _value in _values]
        
        return flattened_batched_data

def main():
    
    args = parse_args()

    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
    accelerator.wait_for_everyone()

    # Load model
    tokenizer = M2M100Tokenizer.from_pretrained(f"facebook/{args.model_name_or_path}")
    model = M2M100ForConditionalGeneration.from_pretrained(f"facebook/{args.model_name_or_path}")

    # Configure lang adapters
    enc_config = "pfeiffer[output_adapter=False,monolingual_enc_adapter=True]"
    dec_config = "pfeiffer[output_adapter=False,monolingual_dec_adapter=True]"

    # Add lang adapters
    model.add_adapter("enc_ha", config=enc_config)
    model.add_adapter("dec_en", config=dec_config)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    processed_datasets = get_processed_dataset(
        data_path=args.data_path,
        src_langs=['en', 'ig'],
        tgt_langs=['ha'],
        tokenizer=tokenizer,
        args=args
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_batch_collator = LanguagePairBatchCollator(train_dataset, batch_size=args.per_device_train_batch_size, all_language_pairs=[('en', 'ha'), ('ig', 'ha')])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=10000, collate_fn=train_batch_collator)
    eval_batch_collator = LanguagePairBatchCollator(eval_dataset, batch_size=args.per_device_eval_batch_size, all_language_pairs=[('en', 'ha'), ('ig', 'ha')])
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=eval_batch_collator, batch_size=10000)

    len_train_dataloader = len(list(train_dataloader)[0])

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len_train_dataloader / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len_train_dataloader / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("translation_no_trainer", experiment_config)

    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        train_dataloader_list = list(train_dataloader)[0]
        len_train_dataloader = len(train_dataloader_list)
        for step, batch in enumerate(train_dataloader_list):
            _input_ids = torch.stack([torch.tensor(_x['input_ids']) for _x in batch], dim=0).cuda()
            _attention_mask = torch.stack([torch.tensor(_x['attention_mask']) for _x in batch], dim=0).cuda()
            _labels = torch.stack([torch.tensor(_x['labels']) for _x in batch], dim=0).cuda()
            outputs = model(**{"input_ids": _input_ids, "attention_mask": _attention_mask, "labels": _labels})
            loss = outputs.loss
    
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len_train_dataloader - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()

        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
        }
        samples_seen = 0
        eval_dataloader_list = list(eval_dataloader)[0]
        for step, batch in enumerate(eval_dataloader_list):
            with torch.no_grad():
                _input_ids = torch.stack([torch.tensor(_x['input_ids']) for _x in batch], dim=0).cuda()
                _attention_mask = torch.stack([torch.tensor(_x['attention_mask']) for _x in batch], dim=0).cuda()
                _labels = torch.stack([torch.tensor(_x['labels']) for _x in batch], dim=0).cuda()
                _batch = {"input_ids": _input_ids, "attention_mask": _attention_mask, "labels": _labels}
                generated_tokens = accelerator.unwrap_model(model).generate(
                    _batch["input_ids"],
                    attention_mask=_batch["attention_mask"],
                    forced_bos_token_id=tokenizer.get_lang_id(batch[0]["tgt_lang"]),
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = _batch["labels"]
                if not args.pad_to_max_length:
                    labels = accelerator.pad_across_processes(_batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                if accelerator.num_processes > 1:
                    if step == len(eval_dataloader) - 1:
                        decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        eval_metric = metric.compute()
        logger.info({"bleu": eval_metric["score"]})

        if args.with_tracking:
            accelerator.log(
                {
                    "bleu": eval_metric["score"],
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_bleu": eval_metric["score"]}, f)


if __name__ == "__main__":
    # example of how to invoke trainer
    # python3 -m train --data_path '/home/ss64293/projects/271b/data' --pad_to_max_length True --model_name_or_path 'm2m100_418M' --output_dir 'mixed_training' --seed 42 --with_tracking
    main()
