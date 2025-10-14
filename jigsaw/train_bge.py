import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import re
import warnings
from urllib.parse import urlparse

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
)
from sentence_transformers.losses import TripletLoss
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

load_dotenv(".env")

warnings.filterwarnings("ignore")


def cleaner(text):
    """Replace URLs with format: <url>: (domain/important-path)"""
    if not text:
        return text

    # Regex pattern to match URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

    def replace_url(match):
        url = match.group(0)
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix if present
            if domain.startswith("www."):
                domain = domain[4:]

            # Extract meaningful path parts (first 1-2 segments)
            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                # Take first 1-2 meaningful path segments
                important_path = "/".join(path_parts[:2])
                return f"<url>: ({domain}/{important_path})"
            else:
                return f"<url>: ({domain})"
        except Exception:
            return "<url>: (unknown)"

    return re.sub(url_pattern, replace_url, str(text))


def load_test_data():
    """Load test data."""
    print("Loading test data...")
    test_df = pd.read_csv("train.csv")
    print(f"Loaded {len(test_df)} test examples")
    print(f"Unique rules: {test_df['rule'].nunique()}")
    return test_df


def collect_all_texts(test_df):
    """Collect all unique texts from test set."""
    print("\nCollecting all texts for embedding...")

    all_texts = set()

    # Add all bodies
    for body in test_df["body"]:
        if pd.notna(body):
            all_texts.add(cleaner(str(body)))

    # Add all positive and negative examples
    example_cols = [
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
    ]

    for col in example_cols:
        for example in test_df[col]:
            if pd.notna(example):
                all_texts.add(cleaner(str(example)))

    all_texts = list(all_texts)
    print(f"Collected {len(all_texts)} unique texts")
    return all_texts


def generate_embeddings(texts, model, batch_size=64):
    """Generate BGE embeddings for all texts."""
    print(f"Generating embeddings for {len(texts)} texts...")

    embeddings = model.encode(
        sentences=texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    return embeddings


def create_test_triplet_dataset(
    test_df, augmentation_factor=2, random_seed=42, subsample_fraction=1.0
):
    """Create triplet dataset from test data: anchor=rule, positive=positive_example, negative=negative_example."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    anchors = []
    positives = []
    negatives = []

    print("Creating rule-aligned triplets from test data...")

    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Processing test rows"
    ):
        rule = cleaner(str(row["rule"]))

        pos_examples = []  # Will contain compliant comments (rule-aligned)
        neg_examples = []  # Will contain violating comments (rule-misaligned)

        for neg_col in [
            "negative_example_1",
            "negative_example_2",
        ]:  # Compliant → triplet positive
            if pd.notna(row[neg_col]):
                pos_examples.append(cleaner(str(row[neg_col])))

        for pos_col in [
            "positive_example_1",
            "positive_example_2",
        ]:  # Violating → triplet negative
            if pd.notna(row[pos_col]):
                neg_examples.append(cleaner(str(row[pos_col])))

        for pos_ex in pos_examples:
            for neg_ex in neg_examples:
                anchors.append(rule)
                positives.append(pos_ex)
                negatives.append(neg_ex)

    if augmentation_factor > 0:
        print(f"Adding {augmentation_factor}x augmentation...")

        rule_positives = {}
        rule_negatives = {}

        for rule in test_df["rule"].unique():
            rule_df = test_df[test_df["rule"] == rule]

            pos_pool = []
            neg_pool = []

            for _, row in rule_df.iterrows():
                for neg_col in [
                    "negative_example_1",
                    "negative_example_2",
                ]:  # Compliant → triplet positive
                    if pd.notna(row[neg_col]):
                        pos_pool.append(cleaner(str(row[neg_col])))
                for pos_col in [
                    "positive_example_1",
                    "positive_example_2",
                ]:  # Violating → triplet negative
                    if pd.notna(row[pos_col]):
                        neg_pool.append(cleaner(str(row[pos_col])))

            rule_positives[rule] = list(set(pos_pool))
            rule_negatives[rule] = list(set(neg_pool))

        for rule in test_df["rule"].unique():
            clean_rule = cleaner(str(rule))
            pos_pool = rule_positives[rule]
            neg_pool = rule_negatives[rule]

            n_samples = min(
                augmentation_factor * len(pos_pool),
                len(pos_pool) * len(neg_pool),
            )

            for _ in range(n_samples):
                if pos_pool and neg_pool:
                    anchors.append(clean_rule)
                    positives.append(random.choice(pos_pool))
                    negatives.append(random.choice(neg_pool))

    combined = list(zip(anchors, positives, negatives))
    random.shuffle(combined)

    # Apply subsampling if requested
    original_count = len(combined)
    if subsample_fraction < 1.0:
        n_samples = int(len(combined) * subsample_fraction)
        combined = combined[:n_samples]
        print(
            f"Subsampled {original_count} -> {len(combined)} triplets ({subsample_fraction * 100:.1f}%)"
        )

    anchors, positives, negatives = (
        zip(*combined) if combined else ([], [], [])
    )

    print(f"Created {len(anchors)} triplets from test data")

    dataset = Dataset.from_dict(
        {
            "anchor": list(anchors),
            "positive": list(positives),
            "negative": list(negatives),
        }
    )

    return dataset


def fine_tune_model(
    model,
    train_dataset,
    epochs=3,
    batch_size=32,
    learning_rate=2e-5,
    margin=0.25,
    output_dir="./models/test-finetuned-bge",
):
    """Fine-tune the sentence transformer model using triplet loss on test data."""

    print(f"Fine-tuning model on {len(train_dataset)} triplets...")

    loss = TripletLoss(model=model, triplet_margin=margin)

    # Calculate max_steps for small datasets
    dataset_size = len(train_dataset)
    steps_per_epoch = max(1, dataset_size // batch_size)
    max_steps = steps_per_epoch * epochs

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=0,
        learning_rate=learning_rate,
        logging_steps=max(1, max_steps // 4),
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        max_grad_norm=1.0,
        dataloader_drop_last=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()

    final_model_path = f"{output_dir}/final"
    print(f"Saving fine-tuned model to {final_model_path}...")
    model.save_pretrained(final_model_path)

    return model, final_model_path


def load_or_create_finetuned_model(test_df):
    """Load fine-tuned model if exists, otherwise create and fine-tune it."""

    fine_tuned_path = "./models/test-finetuned-bge/final"

    if os.path.exists(fine_tuned_path):
        print(f"Loading existing fine-tuned model from {fine_tuned_path}...")
        try:
            word_embedding_model = models.Transformer(
                fine_tuned_path, max_seq_length=128, do_lower_case=True
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode="mean",
            )
            model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model]
            )
            print("Loaded fine-tuned model with explicit pooling")
        except Exception:
            model = SentenceTransformer(fine_tuned_path)
            print("Loaded fine-tuned model with default configuration")
        model.half()
        return model

    print("Fine-tuned model not found. Creating new one...")

    print("Loading base BGE embedding model...")
    # Try Kaggle path first, fallback to HuggingFace
    try:
        model_path = "BAAI/bge-base-en-v1.5"
        word_embedding_model = models.Transformer(
            model_path, max_seq_length=128, do_lower_case=True
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        base_model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model]
        )
        print("Loaded base model from Kaggle path with explicit pooling")
    except Exception:
        model_path = ""  # BAAI/bge-small-en-v1.5
        word_embedding_model = models.Transformer(
            model_path, max_seq_length=128, do_lower_case=True
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        base_model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model]
        )
        print("Loaded base model from local path with explicit pooling")

    triplet_dataset = create_test_triplet_dataset(
        test_df, augmentation_factor=2, subsample_fraction=1.0
    )

    fine_tuned_model, model_path = fine_tune_model(
        model=base_model,
        train_dataset=triplet_dataset,
        epochs=1,
        batch_size=32,
        learning_rate=2e-5,
        margin=0.25,
    )

    print(f"Fine-tuning completed. Model saved to: {model_path}")
    fine_tuned_model.half()
    return fine_tuned_model


def generate_rule_embeddings(test_df, model):
    """Generate embeddings for each unique rule."""
    print("Generating rule embeddings...")

    unique_rules = test_df["rule"].unique()
    rule_embeddings = {}

    for rule in unique_rules:
        clean_rule = cleaner(str(rule))
        rule_emb = model.encode(
            clean_rule, convert_to_tensor=False, normalize_embeddings=True
        )
        rule_embeddings[rule] = rule_emb

    print(f"Generated embeddings for {len(rule_embeddings)} rules")
    return rule_embeddings


def create_rule_centroids(test_df, text_to_embedding, rule_embeddings):
    """Create single centroid (mean) for positive and negative examples for each rule."""
    print("\nCreating rule centroids (single mean centroid per type)...")

    rule_centroids = {}

    for rule in test_df["rule"].unique():
        rule_data = test_df[test_df["rule"] == rule]

        # Collect positive examples
        pos_embeddings = []
        for _, row in rule_data.iterrows():
            for col in ["positive_example_1", "positive_example_2"]:
                if pd.notna(row[col]):
                    clean_text = cleaner(str(row[col]))
                    if clean_text in text_to_embedding:
                        pos_embeddings.append(text_to_embedding[clean_text])

        # Collect negative examples
        neg_embeddings = []
        for _, row in rule_data.iterrows():
            for col in ["negative_example_1", "negative_example_2"]:
                if pd.notna(row[col]):
                    clean_text = cleaner(str(row[col]))
                    if clean_text in text_to_embedding:
                        neg_embeddings.append(text_to_embedding[clean_text])

        if pos_embeddings and neg_embeddings:
            pos_embeddings = np.array(pos_embeddings)
            neg_embeddings = np.array(neg_embeddings)

            # Compute mean centroids
            pos_centroid = pos_embeddings.mean(axis=0)
            neg_centroid = neg_embeddings.mean(axis=0)

            # Normalize centroids
            pos_centroid = pos_centroid / np.linalg.norm(pos_centroid)
            neg_centroid = neg_centroid / np.linalg.norm(neg_centroid)

            rule_centroids[rule] = {
                "positive": pos_centroid,
                "negative": neg_centroid,
                "pos_count": len(pos_embeddings),
                "neg_count": len(neg_embeddings),
                "rule_embedding": rule_embeddings[rule],
            }

            print(
                f"  Rule: {rule[:50]}... - Pos: {len(pos_embeddings)}, Neg: {len(neg_embeddings)}"
            )

    print(f"Created centroids for {len(rule_centroids)} rules")
    return rule_centroids


def predict_test_set(test_df, text_to_embedding, rule_centroids):
    """Predict test set using Euclidean distance between body and pos/neg centroids."""
    print("\nMaking predictions on test set with Euclidean distance...")

    row_ids = []
    predictions = []

    for rule in test_df["rule"].unique():
        print(f"  Processing rule: {rule[:50]}...")
        rule_data = test_df[test_df["rule"] == rule]

        if rule not in rule_centroids:
            continue

        pos_centroid = rule_centroids[rule]["positive"]
        neg_centroid = rule_centroids[rule]["negative"]

        # Collect all valid embeddings and row_ids for this rule
        valid_embeddings = []
        valid_row_ids = []

        for _, row in rule_data.iterrows():
            body = cleaner(str(row["body"]))
            row_id = row["row_id"]

            if body in text_to_embedding:
                valid_embeddings.append(text_to_embedding[body])
                valid_row_ids.append(row_id)

        if not valid_embeddings:
            continue

        # Convert to numpy array
        query_embeddings = np.array(valid_embeddings)

        # Compute Euclidean distances
        pos_distances = np.linalg.norm(query_embeddings - pos_centroid, axis=1)
        neg_distances = np.linalg.norm(query_embeddings - neg_centroid, axis=1)

        # Score: closer to positive (lower distance) = higher violation score
        rule_predictions = neg_distances - pos_distances

        row_ids.extend(valid_row_ids)
        predictions.extend(rule_predictions)

    print(f"Made predictions for {len(predictions)} test examples")
    return row_ids, np.array(predictions)


def main():
    """Main inference pipeline."""
    print("=" * 70)
    print("SIMPLE SIMILARITY CLASSIFIER - INFERENCE")
    print("=" * 70)

    # Step 1: Load test data
    test_df = load_test_data()

    # Step 2: Load or create fine-tuned model
    print("\n" + "=" * 50)
    print("MODEL PREPARATION PHASE")
    print("=" * 50)
    model = load_or_create_finetuned_model(test_df)

    # Step 3: Collect all texts
    all_texts = collect_all_texts(test_df)

    # Step 4: Generate embeddings with fine-tuned model
    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION PHASE")
    print("=" * 50)
    all_embeddings = generate_embeddings(all_texts, model)

    # Step 5: Create text to embedding mapping
    text_to_embedding = {
        text: emb for text, emb in zip(all_texts, all_embeddings)
    }

    # Step 6: Generate rule embeddings
    rule_embeddings = generate_rule_embeddings(test_df, model)

    # Step 7: Create rule centroids from test examples
    rule_centroids = create_rule_centroids(
        test_df, text_to_embedding, rule_embeddings
    )

    # Step 8: Predict test set
    print("\n" + "=" * 50)
    print("PREDICTION PHASE")
    print("=" * 50)
    row_ids, predictions = predict_test_set(
        test_df, text_to_embedding, rule_centroids
    )

    # Step 9: Create submission with rule-conditioned scores
    submission_df = pd.DataFrame(
        {"row_id": row_ids, "rule_violation": predictions}
    )

    submission_df["rule_violation"] = (
        submission_df["rule_violation"] - submission_df["rule_violation"].min()
    ) / (
        submission_df["rule_violation"].max()
        - submission_df["rule_violation"].min()
    )

    submission_df.to_csv("submission_bge.csv", index=False)
    print(
        f"\nSaved predictions for {len(submission_df)} test examples to submission.csv"
    )

    print(f"\n{'=' * 70}")
    print("FINE-TUNED EUCLIDEAN DISTANCE INFERENCE COMPLETED")
    print("Model: Fine-tuned BGE on test data triplets")
    print("Method: Single centroid with Euclidean distance")
    print("Predicted on {len(test_df)} test examples")
    print(
        f"Prediction stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}"
    )
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
