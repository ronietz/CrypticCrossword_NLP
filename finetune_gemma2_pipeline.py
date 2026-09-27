import os
import argparse
import torch
import torch.distributed.tensor

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "google/gemma-2-2b"

TRAIN_PATH = "data/processed/cryptonite_train.jsonl"
VAL_PATH = "data/processed/cryptonite_val.jsonl"

ANSWER_OUT = "outputs/gemma2_answer"
WORDPLAY_OUT = "outputs/gemma2_wordplay"

SEED = 42


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--task",
    choices=["answer", "wordplay"],
    required=True,
)

parser.add_argument(
    "--resume",
    action="store_true",
)

args = parser.parse_args()


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# LOAD UNITED DATASET
# ============================================================

raw = load_dataset(
    "json",
    data_files={
        "train": TRAIN_PATH,
        "validation": VAL_PATH,
    },
)


def has_source(example, source):
    meta = example.get("meta") or {}
    sources = meta.get("sources") or []
    return source in sources


if args.task == "answer":

    train_ds = raw["train"].filter(
        lambda x: has_source(x, "cryptonite"),
        desc="Selecting Cryptonite train",
    )

    full_val_ds = raw["validation"].filter(
        lambda x: has_source(x, "cryptonite"),
        desc="Selecting Cryptonite validation",
    )

    # We keep the full validation untouched for final accuracy/pass@N.
    # During training we only use 2,000 examples for eval_loss,
    # otherwise evaluation becomes unnecessarily expensive.
    shuffled_val = full_val_ds.shuffle(seed=SEED)

    eval_ds = shuffled_val.select(
        range(min(2000, len(shuffled_val)))
    )

    output_dir = ANSWER_OUT

    max_prompt_length = 96
    max_target_length = 24

    print(
        f"Cryptonite train: {len(train_ds)}",
        flush=True,
    )
    print(
        f"Cryptonite full validation: {len(full_val_ds)}",
        flush=True,
    )
    print(
        f"Training-time validation subset: {len(eval_ds)}",
        flush=True,
    )


else:

    train_ds = raw["train"].filter(
        lambda x:
            has_source(x, "wordplay")
            and bool(str(x.get("definition", "")).strip())
            and bool(str(x.get("wordplay", "")).strip()),
        desc="Selecting Wordplay train",
    )

    eval_ds = raw["validation"].filter(
        lambda x:
            has_source(x, "wordplay")
            and bool(str(x.get("definition", "")).strip())
            and bool(str(x.get("wordplay", "")).strip()),
        desc="Selecting Wordplay validation",
    )

    output_dir = WORDPLAY_OUT

    max_prompt_length = 128
    max_target_length = 160

    print(
        f"Wordplay train: {len(train_ds)}",
        flush=True,
    )
    print(
        f"Wordplay validation: {len(eval_ds)}",
        flush=True,
    )


print(
    f"\nTASK = {args.task}",
    flush=True,
)


# ============================================================
# PROMPTS
# ============================================================

def make_prompt_target(example):

    clue = str(example["clue"]).strip()
    enumeration = str(example["enumeration"]).strip()
    answer = str(example["answer"]).strip()

    if args.task == "answer":

        # IMPORTANT:
        # We deliberately DO NOT use "letters".
        # In this processed dataset "letters" contains the answer.
        prompt = (
            "solve the cryptic clue.\n"
            f"clue: {clue}\n"
            f"enumeration: {enumeration}\n"
            "answer:"
        )

        target = answer

    else:

        definition = str(
            example["definition"]
        ).strip()

        wordplay = str(
            example["wordplay"]
        ).strip()

        prompt = (
            "explain the cryptic clue.\n"
            f"clue: {clue}\n"
            f"enumeration: {enumeration}\n"
            f"answer: {answer}\n"
            "reasoning:"
        )

        target = (
            f"definition: {definition} ; "
            f"wordplay: {wordplay}"
        )

    return prompt, target


# ============================================================
# TOKENIZATION
#
# Loss is computed ONLY on the desired output.
# Prompt tokens receive label -100.
# ============================================================

def encode(prompt, target):

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    )

    target_tokens = tokenizer(
        " " + target,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_length,
    )

    prompt_ids = prompt_tokens["input_ids"]
    target_ids = target_tokens["input_ids"]

    if tokenizer.eos_token_id is not None:
        target_ids = (
            target_ids
            + [tokenizer.eos_token_id]
        )

    input_ids = (
        prompt_ids
        + target_ids
    )

    labels = (
        [-100] * len(prompt_ids)
        + target_ids
    )

    attention_mask = (
        [1] * len(input_ids)
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def preprocess(batch):

    output = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    n = len(batch["clue"])

    for i in range(n):

        example = {
            key: batch[key][i]
            for key in batch
        }

        prompt, target = make_prompt_target(
            example
        )

        encoded = encode(
            prompt,
            target,
        )

        for key in output:
            output[key].append(
                encoded[key]
            )

    return output


train_tok = train_ds.map(
    preprocess,
    batched=True,
    batch_size=1000,
    remove_columns=train_ds.column_names,
    desc=f"Tokenizing {args.task} train",
)

eval_tok = eval_ds.map(
    preprocess,
    batched=True,
    batch_size=1000,
    remove_columns=eval_ds.column_names,
    desc=f"Tokenizing {args.task} validation",
)


# ============================================================
# GEMMA + LoRA
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# TRAINING
# ============================================================

if args.task == "answer":

    # 431K examples.
    # One full epoch is already a substantial fine-tuning run.
    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=1,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,

        gradient_accumulation_steps=4,

        learning_rate=2e-4,
        weight_decay=0.01,

        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        fp16=True,

        logging_steps=50,

        evaluation_strategy="steps",
        eval_steps=2000,

        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,

        report_to="none",

        dataloader_num_workers=4,

        seed=SEED,
        data_seed=SEED,

        remove_unused_columns=False,
    )

    callbacks = []


else:

    # Small reasoning dataset.
    # Stop automatically when validation loss stops improving.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=8,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,

        gradient_accumulation_steps=16,

        learning_rate=2e-4,
        weight_decay=0.01,

        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        fp16=True,

        logging_steps=25,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=2,

        report_to="none",

        dataloader_num_workers=4,

        seed=SEED,
        data_seed=SEED,

        remove_unused_columns=False,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]


collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
    return_tensors="pt",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=collator,
    callbacks=callbacks,
)


# ============================================================
# OPTIONAL RESUME
# ============================================================

resume_checkpoint = None

if args.resume:

    resume_checkpoint = get_last_checkpoint(
        output_dir
    )

    print(
        "Resume checkpoint:",
        resume_checkpoint,
        flush=True,
    )


trainer.train(
    resume_from_checkpoint=resume_checkpoint
)


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

final_dir = os.path.join(
    output_dir,
    "final_model",
)

os.makedirs(
    final_dir,
    exist_ok=True,
)

trainer.model.save_pretrained(
    final_dir
)

tokenizer.save_pretrained(
    final_dir
)

print(
    f"\nFINAL MODEL SAVED TO: {final_dir}",
    flush=True,
)

print("DONE.", flush=True)
