import os
import subprocess

import pandas as pd
import torch
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_folder
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.utils import is_torch_bf16_gpu_available
from trl import SFTConfig, SFTTrainer

load_dotenv(".env")

file_path = "submission_deberta.csv"

if not os.path.exists(file_path):
    print("Deberta output not fround. Starting Deberta Pipeline.")
    subprocess.run(["uv", "run", "train_deberta.py"])

deberta_output = file_path

df = pd.read_csv("train.csv")
q = pd.read_csv(deberta_output)

df["expert_1"] = (q["rule_violation"] * 100).round(2).astype(str) + "%"
df.to_csv("stack_train.csv")

# Main configuration parameters
WANDB = True  # Enable/disable Weights & Biases logging
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"  # Pre-trained model to fine-tune
IS_DEBUG = True  # Debug mode with small dataset
N_FOLDS = 5  # Number of cross-validation folds
EPOCH = 1  # Training epochs
LR = 1e-4  # Learning rate
TRAIN_BS = 1  # 8  # Training batch size
GRAD_ACC_NUM = 8  # 1  # Gradient accumulation steps
EVAL_BS = 8  # Evaluation batch size
FOLD = 0  # Current fold to train
SEED = 42  # Random seed for reproducibility

# Derive experiment name and paths
EXP_ID = "jigsaw-lora-finetune-stack"
if IS_DEBUG:
    EXP_ID += "_debug"
EXP_NAME = EXP_ID + f"_fold{FOLD}"
COMPETITION_NAME = "jigsaw-kaggle"
OUTPUT_DIR = "./ "  # f"/kaggle/output/{EXP_NAME}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_OUTPUT_PATH = f"{OUTPUT_DIR}/trained_model"


def main():
    # Load the dataset
    df = pd.read_csv("stack_train.csv")
    if IS_DEBUG:
        # Use a small subset for debugging
        df = df.sample(50, random_state=SEED).reset_index(drop=True)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"  # Important for causal language models

    # Define system prompt for the classification task
    SYS_PROMPT = """
You are an Expert Reddit Comment Moderator working in a group. You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No.
"""

    prompts = []
    for i, row in df.iterrows():
        text = f"""
r/{row.subreddit}
Rule: {row.rule}

1) {row.positive_example_1}
Violation: Yes

2) {row.negative_example_1}
Violation: No

3) {row.negative_example_2}
Violation: No

4) {row.positive_example_2}
Violation: Yes

Comment: {row.body}

Other Expert opinion:
Expert violation confidence : {row.expert_1}
"""

        # Format as a chat conversation using the model's template
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text},
        ]

        prompt = (
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            + "Answer:"
        )
        prompts.append(prompt)

    # Add the formatted prompts to the dataframe
    df["prompt"] = prompts
    df["completion"] = df["rule_violation"].apply(
        lambda x: "Yes" if x == 1 else "No"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # device_map="auto",  # Automatically distribute model across available GPUs
    )

    # Configure LoRA parameters
    lora_config = LoraConfig(
        r=16,  # Rank of the update matrices
        lora_alpha=16,  # Alpha parameter for LoRA scaling
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        task_type=TaskType.CAUSAL_LM,
        bias="none",  # Don't train bias terms
        # Target the attention and MLP modules of the transformer
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

    # Initialize Weights & Biases for experiment tracking

    wandb_key = os.getenv("WANDB_API_KEY")

    if WANDB:
        wandb.login(key=wandb_key)
        wandb.init(project=COMPETITION_NAME, name=EXP_NAME + "-Qwen2.5_32b")
        REPORT_TO = "wandb"
    else:
        REPORT_TO = "none"

    # Split data into train and validation sets
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        if fold == FOLD:
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_val = df.iloc[val_idx].reset_index(drop=True)
            break

    # Save the split data
    df_train.to_pickle(f"{OUTPUT_DIR}/train.pkl")
    df_val.to_pickle(f"{OUTPUT_DIR}/val.pkl")

    # Set up training arguments
    training_args = SFTConfig(
        output_dir=MODEL_OUTPUT_PATH,
        logging_steps=1,  # Log metrics every 10 steps
        logging_strategy="steps",
        eval_strategy="steps",  # No evaluation during training
        eval_steps=1,
        save_strategy="steps",
        save_steps=0.1,  # Save checkpoint after 10% of training steps
        save_total_limit=10,  # Keep only the 10 most recent checkpoints
        num_train_epochs=EPOCH,
        optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
        lr_scheduler_type="linear",
        warmup_ratio=0.1,  # Warm up learning rate over 10% of steps
        learning_rate=LR,
        weight_decay=0.01,
        # Use BF16 if available, otherwise FP16
        bf16=is_torch_bf16_gpu_available(),
        fp16=not is_torch_bf16_gpu_available(),
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACC_NUM,
        gradient_checkpointing=True,  # Save memory with gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=False,
        report_to=[REPORT_TO],  # to log to stdout,
        seed=42,
        remove_unused_columns=False,  # Keep all columns in the dataset
    )

    df_train_dataset = Dataset.from_pandas(df_train)
    df_val_dataset = Dataset.from_pandas(df_val)

    # Initialize trainer
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=df_train_dataset,
        eval_dataset=df_val_dataset,
        peft_config=lora_config,
    )

    # Start training
    trainer_output = trainer.train()

    # Save the final model
    trainer.save_model(MODEL_OUTPUT_PATH)
    api = HfApi()
    api.create_repo(
        repo_id="Weedoo/jigsaw-kaggle-Qwen2.5-32b-stack-deberta",
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    upload_folder(
        folder_path=MODEL_OUTPUT_PATH,
        repo_id="Weedoo/jigsaw-kaggle-Qwen2.5-32b-stack-deberta",
        repo_type="model",
    )


if __name__ == "__main__":
    main()
