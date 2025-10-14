import os
import re

import numpy as np
import pandas as pd  # type: ignore
import torch
from dotenv import load_dotenv

load_dotenv(".env")


def url_to_semantics(text):
    if not isinstance(text, str):
        return ""

    urls = re.findall(r"https?://[^\s/$.?#].[^\s]*", text)
    if not urls:
        return ""

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()

        domain_match = re.search(
            r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower
        )
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split(".")
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3:
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        path = re.sub(
            r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower
        )
        path_parts = [
            p for p in re.split(r"[/_.-]+", path) if p and p.isalnum()
        ]

        for part in path_parts:
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if (
                part_clean
                and part_clean not in seen_semantics
                and len(part_clean) > 3
            ):
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    if not all_semantics:
        return ""
    return f"\nURL Keywords: {' '.join(all_semantics)}"


def get_dataframe_to_train():
    """
    Process test data to create additional training samples
    """
    test_dataset = pd.read_csv(
        "train.csv"
    )  # remove body and use examples from the dataset

    flatten = []

    ## test data is not labelled therefore use the example to create additional data for training
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[
                [
                    "rule",
                    "subreddit",
                    "positive_example_1",
                    "positive_example_2",
                    "negative_example_1",
                    "negative_example_2",
                ]
            ].copy()
            if violation_type == "positive":
                # body uses the current positive_example
                body_col = f"positive_example_{i}"
                other_positive_col = (
                    f"positive_example_{3 - i}"  # another positive
                )
                sub_dataset["body"] = sub_dataset[body_col]
                sub_dataset["positive_example"] = sub_dataset[
                    other_positive_col
                ]
                # negative_example randomly selected
                sub_dataset["negative_example"] = np.where(
                    np.random.rand(len(sub_dataset)) < 0.5,
                    sub_dataset["negative_example_1"],
                    sub_dataset["negative_example_2"],
                )
                sub_dataset["rule_violation"] = 1

            else:  # violation_type == "negative"
                body_col = f"negative_example_{i}"
                other_negative_col = f"negative_example_{3 - i}"
                sub_dataset["body"] = sub_dataset[body_col]
                sub_dataset["negative_example"] = sub_dataset[
                    other_negative_col
                ]
                sub_dataset["positive_example"] = np.where(
                    np.random.rand(len(sub_dataset)) < 0.5,
                    sub_dataset["positive_example_1"],
                    sub_dataset["positive_example_2"],
                )
                sub_dataset["rule_violation"] = 0

            # Delete the original candidate column
            sub_dataset.drop(
                columns=[
                    "positive_example_1",
                    "positive_example_2",
                    "negative_example_1",
                    "negative_example_2",
                ],
                inplace=True,
            )

            flatten.append(sub_dataset)

    # merge all DataFrame
    dataframe = pd.concat(flatten, axis=0)

    dataframe = dataframe.drop_duplicates(ignore_index=True)
    dataframe = dataframe.sample(frac=1, random_state=42).reset_index(
        drop=True  # shuffle
    )

    return dataframe


def build_dataframe_deberta(dataframe=None):
    def build_prompt(row):
        rule = row["rule"]
        body = row["body"]
        url_features = url_to_semantics(body)

        return f"{rule}[SEP]{body}{url_features}"

    if dataframe is None:  # training
        dataframe = get_dataframe_to_train()

    dataframe = dataframe.copy()
    dataframe["input_text"] = dataframe.apply(build_prompt, axis=1)

    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: 1,
                0: 0,
            }
        )

    return dataframe


class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class JigsawTrainer:
    def __init__(self, model_path, save_path):
        self.model_path = model_path
        self.save_path = save_path

    def run(self):
        raise NotImplementedError


class DebertaBase(JigsawTrainer):
    def train_with_data(self, data):
        """
        Run Trainer on data
        """
        from transformers import (
            DataCollatorWithPadding,
            DebertaV2ForSequenceClassification,
            DebertaV2Tokenizer,
            Trainer,
            TrainingArguments,
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
        collator = DataCollatorWithPadding(tokenizer)

        data.drop_duplicates(
            subset=["body", "rule"], keep="first", inplace=True
        )

        train_encodings = tokenizer(
            data["input_text"].tolist(),
            truncation=True,
            max_length=512,
        )

        train_labels = data["rule_violation"].tolist()
        train_dataset = RedditDataset(train_encodings, train_labels)

        model = DebertaV2ForSequenceClassification.from_pretrained(
            self.model_path, num_labels=2
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            warmup_ratio=0.1,
            weight_decay=0.01,
            report_to="none",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        trainer.train()
        # trainer.save_model(self.save_path)
        # tokenizer.save_pretrained(self.save_path)

        test_df = pd.read_csv("train.csv")
        test_data = build_dataframe_deberta(
            test_df
        )  # use body to create input_text
        test_encodings = tokenizer(
            test_data["input_text"].tolist(),
            truncation=True,
            max_length=256,
        )

        test_dataset = RedditDataset(test_encodings)

        predictions = trainer.predict(test_dataset)
        full_probs = torch.nn.functional.softmax(
            torch.tensor(predictions.predictions), dim=1
        )

        probs = full_probs[:, 1].numpy()

        submission_df = pd.DataFrame(
            {
                "row_id": data["row_id"],
                "rule_violation": probs,
            }
        )
        submission_df.to_csv(self.save_path, index=False)

    def run(self):
        """
        Run Trainer on data determined by Config.data_path
        """
        dataframe = build_dataframe_deberta()
        self.train_with_data(dataframe)


def main():
    MODEL_NAME = "microsoft/deberta-v3-base"
    SAVE_PATH = "submission_deberta.csv"

    trainer = DebertaBase(MODEL_NAME, SAVE_PATH)

    trainer.run()


if __name__ == "__main__":
    main()
