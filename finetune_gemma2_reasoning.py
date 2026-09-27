import os
import json
import random
import re

import numpy as np
import torch
import torch.distributed.tensor

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


# ============================================================
# CONFIG
# ============================================================

DEBUG_MODE = False

MODEL_NAME = "google/gemma-2-2b"

DATA_PATH = "data/star_reasoning_train.jsonl"

OUTPUT_DIR = (
    "outputs/gemma2_reasoning_debug"
    if DEBUG_MODE
    else "outputs/gemma2_reasoning"
)

SEED = 42
MAX_LENGTH = 256

print(f"DEBUG_MODE: {DEBUG_MODE}")
print(f"Base model: {MODEL_NAME}")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")


# ============================================================
# SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# MODEL
# ============================================================

dtype = (
    torch.bfloat16
    if torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    else torch.float16
)

print("Using dtype:", dtype)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
)

model.config.use_cache = False

# Needed when combining gradient checkpointing + LoRA
model.enable_input_require_grads()


# ============================================================
# LoRA
# ============================================================

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
# LOAD EXACT SAME REASONING DATA
# ============================================================

records = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:

        if not line.strip():
            continue

        obj = json.loads(line)

        records.append({
            "source": obj["source"].strip(),
            "target": obj["target"].strip(),
            "answer": obj["answer"].strip(),
        })

print(f"Loaded {len(records)} reasoning examples")


dataset = Dataset.from_list(records)

split = dataset.train_test_split(
    test_size=0.10,
    seed=SEED,
)

train_raw = split["train"]
eval_raw = split["test"]

print(
    f"Train: {len(train_raw)} / "
    f"Validation: {len(eval_raw)}"
)


if DEBUG_MODE:
    train_raw = train_raw.select(
        range(min(50, len(train_raw)))
    )

    eval_raw = eval_raw.select(
        range(min(50, len(eval_raw)))
    )


# ============================================================
# CAUSAL-LM TOKENIZATION
#
# IMPORTANT:
# Gemma is NOT encoder-decoder like T5.
#
# The sequence is:
#
#   PROMPT + TARGET
#
# But labels corresponding to PROMPT are -100,
# so the model gets loss ONLY on:
#
#   definition + wordplay + answer
# ============================================================

def tokenize_example(example):

    source = example["source"]
    target = example["target"]

    prompt = (
        f"{source}\n"
        f"response: "
    )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    target_ids = target_ids + [tokenizer.eos_token_id]

    # Preserve the target if truncation is ever necessary.
    max_prompt_len = MAX_LENGTH - len(target_ids)

    if max_prompt_len < 1:
        target_ids = target_ids[: MAX_LENGTH - 1]
        max_prompt_len = 1

    prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + target_ids

    labels = (
        [-100] * len(prompt_ids)
        + target_ids
    )

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataset = train_raw.map(
    tokenize_example,
    remove_columns=train_raw.column_names,
)

eval_dataset = eval_raw.map(
    tokenize_example,
    remove_columns=eval_raw.column_names,
)


# ============================================================
# SANITY CHECK
# ============================================================

example = train_raw[0]

print("\n" + "=" * 80)
print("TRAINING EXAMPLE")
print("=" * 80)

print("\nSOURCE:")
print(example["source"])

print("\nTARGET:")
print(example["target"])

print("=" * 80 + "\n")


# ============================================================
# COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding=True,
)


# ============================================================
# TRAINING
#
# LoRA, not full fine-tuning.
#
# We use validation loss to select the best checkpoint.
# Final answer accuracy will later be measured with exactly
# the same official Cryptonite evaluation used for Flan-T5.
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    learning_rate=2e-4,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,

    per_device_eval_batch_size=1,

    num_train_epochs=1 if DEBUG_MODE else 10,
    max_steps=5 if DEBUG_MODE else -1,

    gradient_checkpointing=True,

    eval_strategy=(
        "steps"
        if DEBUG_MODE
        else "epoch"
    ),

    eval_steps=5 if DEBUG_MODE else None,

    save_strategy=(
        "steps"
        if DEBUG_MODE
        else "epoch"
    ),

    save_steps=5 if DEBUG_MODE else 500,

    save_total_limit=3,

    load_best_model_at_end=not DEBUG_MODE,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=10,

    fp16=(
        not torch.cuda.is_bf16_supported()
        if torch.cuda.is_available()
        else False
    ),

    bf16=(
        torch.cuda.is_bf16_supported()
        if torch.cuda.is_available()
        else False
    ),

    report_to="none",

    seed=SEED,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=(
        None
        if DEBUG_MODE
        else [
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ]
    ),
)


# ============================================================
# TRAIN
# ============================================================

print(
    "Starting Gemma2 reasoning LoRA fine-tuning..."
)

trainer.train()


# ============================================================
# SAVE BEST LoRA ADAPTER
# ============================================================

FINAL_DIR = os.path.join(
    OUTPUT_DIR,
    "final_model"
)

trainer.model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

with open(
    os.path.join(FINAL_DIR, "base_model.txt"),
    "w",
    encoding="utf-8",
) as f:
    f.write(MODEL_NAME + "\n")


print(
    "\nTraining complete."
    f"\nBest Gemma2 reasoning adapter saved to:"
    f"\n{FINAL_DIR}"
)


# ============================================================
# QUICK GENERATION SANITY CHECK
# ============================================================

trainer.model.eval()

print("\n" + "=" * 80)
print("GENERATION SANITY CHECK")
print("=" * 80)

for i in range(min(5, len(eval_raw))):

    ex = eval_raw[i]

    prompt = (
        f"{ex['source']}\n"
        f"response: "
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(trainer.model.device)

    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Remove the original prompt tokens
    generated_ids = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    prediction = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("\n---")
    print("SOURCE:")
    print(ex["source"])

    print("\nGOLD:")
    print(ex["target"])

    print("\nMODEL:")
    print(prediction)

print("\nDone.")
