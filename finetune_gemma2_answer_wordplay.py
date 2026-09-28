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

OUTPUT_DIR = "outputs/gemma2_answer_wordplay"

SEED = 42

MAX_PROMPT_LENGTH = 96
MAX_TARGET_LENGTH = 24


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from the latest checkpoint in OUTPUT_DIR if one exists.",
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
# LOAD PROCESSED DATASET AND KEEP WORDPLAY ONLY
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


# Use the same usable Wordplay subset as the reasoning-generator training:
# source=wordplay and non-empty human definition/wordplay annotations.
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

print(
    f"Wordplay train: {len(train_ds)}",
    flush=True,
)
print(
    f"Wordplay validation: {len(eval_ds)}",
    flush=True,
)


# ============================================================
# ANSWER-ONLY PROMPT
# ============================================================

def make_prompt_target(example):
    clue = str(example["clue"]).strip()
    enumeration = str(example["enumeration"]).strip()
    answer = str(example["answer"]).strip()

    # IMPORTANT:
    # Do not use the "letters" field from the processed dataset:
    # it contains the gold answer and would leak the target.
    #
    # Although these rows have human definition/wordplay annotations,
    # they are deliberately NOT included in the input. This experiment
    # trains an answer-only solver on the Wordplay subset.
    prompt = (
        "solve the cryptic clue.\n"
        f"clue: {clue}\n"
        f"enumeration: {enumeration}\n"
        "answer:"
    )

    target = answer

    return prompt, target


# ============================================================
# TOKENIZATION
#
# Loss is computed only on answer tokens.
# Prompt tokens receive label -100.
# ============================================================

def encode(prompt, target):
    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
    )

    target_tokens = tokenizer(
        " " + target,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
    )

    prompt_ids = prompt_tokens["input_ids"]
    target_ids = target_tokens["input_ids"]

    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

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

        prompt, target = make_prompt_target(example)
        encoded = encode(prompt, target)

        for key in output:
            output[key].append(encoded[key])

    return output


train_tok = train_ds.map(
    preprocess,
    batched=True,
    batch_size=1000,
    remove_columns=train_ds.column_names,
    desc="Tokenizing Wordplay answer train",
)

eval_tok = eval_ds.map(
    preprocess,
    batched=True,
    batch_size=1000,
    remove_columns=eval_ds.column_names,
    desc="Tokenizing Wordplay answer validation",
)


# ============================================================
# GEMMA-2-2B + LoRA
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# Small dataset: use gradient checkpointing as in the original
# Wordplay training branch.
model.gradient_checkpointing_enable()
model.enable_input_require_grads()


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

# The Wordplay subset is small (~5.4K examples), so use the
# small-dataset regime from the original Wordplay branch:
# train for up to 8 epochs and stop when validation loss stops improving.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

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
        OUTPUT_DIR
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
# SAVE FINAL ANSWER-ONLY ADAPTER
# ============================================================

final_dir = os.path.join(
    OUTPUT_DIR,
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
