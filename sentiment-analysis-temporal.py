import gc
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split

# Set constants
FILE_PATH = "Movies_and_TV.jsonl"
BATCH_SIZE = 16
NUM_EPOCHS = 1

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load evaluation metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def load_data(file_path):
    """
    Load data from a JSONL file.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSON lines"):
            d = json.loads(line)
            data.append(
                {"text": d["text"], "rating": d["rating"], "timestamp": d["timestamp"]}
            )
    df = pd.DataFrame(data)
    return df


def assign_label(rating):
    """
    Map rating to binary labels:
      - 1.0, 2.0 => 0 (negative)
      - 4.0, 5.0 => 1 (positive)
    Discard if rating == 3.0 or unexpected.
    """
    if rating in [1.0, 2.0]:
        return 0
    elif rating in [4.0, 5.0]:
        return 1
    else:
        return None


def preprocess_data(df):
    """
    Preprocess the DataFrame:
    - Assign binary labels
    - Convert timestamps to datetime
    - Extract year and month
    - Keep relevant columns only
    """
    df["label"] = df["rating"].apply(assign_label)
    df = df.dropna(subset=["label"])

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["reviewDate"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.dropna(subset=["reviewDate"])

    df["reviewYear"] = df["reviewDate"].dt.year
    df["reviewMonth"] = df["reviewDate"].dt.month
    df = df[["text", "label", "reviewYear", "reviewMonth"]].dropna()

    df["label"] = df["label"].astype(int)
    return df


def balance_classes(df_split):
    """
    Balance the dataset to have equal counts of positive and negative labels.
    """
    count_neg = len(df_split[df_split["label"] == 0])
    count_pos = len(df_split[df_split["label"] == 1])
    min_count = min(count_neg, count_pos)
    if min_count == 0:
        raise ValueError("One of the classes has zero samples.")

    df_neg = df_split[df_split["label"] == 0].sample(min_count, random_state=42)
    df_pos = df_split[df_split["label"] == 1].sample(min_count, random_state=42)

    balanced_df = (
        pd.concat([df_neg, df_pos])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return balanced_df


def split_and_balance(df_subset, test_size=0.2, random_state=42):
    """
    Split into train/test sets and balance each.
    """
    train_df, test_df = train_test_split(
        df_subset,
        test_size=test_size,
        random_state=random_state,
        stratify=df_subset["label"],
    )
    train_bal = balance_classes(train_df)
    print("Train set label distribution after balancing:")
    print(train_bal["label"].value_counts())
    test_bal = balance_classes(test_df)
    print("Test set label distribution after balancing:")
    print(test_bal["label"].value_counts())
    return train_bal, test_bal


def resample_set(df, num_samples_per_label):
    """
    Resample the dataset to a fixed number of samples per label.
    """
    return (
        df.groupby("label")
        .apply(lambda x: x.sample(num_samples_per_label, random_state=42))
        .reset_index(drop=True)
        .sample(frac=1, random_state=42)
    )


def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1_val = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]
    return {"accuracy": acc, "f1": f1_val}


def train_model_on_split(df_train, df_val, output_dir, epochs=2, batch_size=16):
    """
    Train a DistilBERT model and evaluate on validation data.
    """
    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.config.problem_type = "single_label_classification"

    train_dataset = train_dataset.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final Evaluation on {output_dir}: {metrics}")

    return trainer


def evaluate_on_test_set(trainer, df_test):
    """
    Evaluate the trained model on the test set.
    """
    dataset_test = Dataset.from_pandas(df_test)
    tokenizer = trainer.tokenizer

    dataset_test = dataset_test.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    dataset_test = dataset_test.rename_column("label", "labels")
    dataset_test.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    trainer.model.to(device)
    metrics = trainer.evaluate(dataset_test)
    return metrics


def free_vram():
    """
    Free VRAM by clearing cache.
    """
    torch.cuda.empty_cache()
    gc.collect()


def run_experiment(experiment_name, period_trains, period_tests):
    """
    Run training and cross-period evaluation for a given experiment.
    """
    print(f"\n=================== {experiment_name} ===================")
    for train_period, (train_data, val_data) in period_trains.items():
        print(f"\n** Training on {train_period} **")
        output_dir = f"distilbert_{experiment_name}_{train_period}"
        trainer = train_model_on_split(
            train_data,
            val_data,
            output_dir=output_dir,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
        )

        for test_period, test_data in period_tests.items():
            print(f"Evaluating on {test_period}...")
            metrics = evaluate_on_test_set(trainer, test_data)
            print(f"Model trained on {train_period} tested on {test_period}: {metrics}")

        del trainer
        free_vram()


def prepare_experiment_1(df):
    """
    Prepare data splits for Experiment 1:
    Periods: 2004-2007, 2008-2011, 2012-2015, 2016-2019, 2020-2023.
    """
    periods = {
        "2004_2007": df[(df["reviewYear"] >= 2004) & (df["reviewYear"] <= 2007)],
        "2008_2011": df[(df["reviewYear"] >= 2008) & (df["reviewYear"] <= 2011)],
        "2012_2015": df[(df["reviewYear"] >= 2012) & (df["reviewYear"] <= 2015)],
        "2016_2019": df[(df["reviewYear"] >= 2016) & (df["reviewYear"] <= 2019)],
        "2020_2023": df[(df["reviewYear"] >= 2020) & (df["reviewYear"] <= 2023)],
    }

    print("\nShapes for Experiment 1:")
    for period, data in periods.items():
        print(f"{period}: {data.shape}")

    period_trains = {}
    period_tests = {}
    for period, data in periods.items():
        print(f"\nProcessing {period}...")
        train_bal, test_bal = split_and_balance(data)
        period_trains[period] = (train_bal, test_bal)
        period_tests[period] = test_bal

    train_sizes = [len(train) for train, _ in period_trains.values()]
    test_sizes = [len(test) for test in period_tests.values()]
    n_train = min(train_sizes)
    n_test = min(test_sizes)

    print(f"\nSampling each train set to size {n_train} (per experiment period)")
    print(f"Sampling each test set to size {n_test} (per experiment period)")

    for period in period_trains:
        train_bal, test_bal = period_trains[period]
        num_samples_train = n_train // 2
        num_samples_test = n_test // 2

        period_trains[period] = (
            resample_set(train_bal, num_samples_train),
            resample_set(test_bal, num_samples_test),
        )
        period_tests[period] = resample_set(test_bal, num_samples_test)

    return period_trains, period_tests


def prepare_experiment_2(df):
    """
    Prepare data splits for Experiment 2:
    Years: 1999, 2021, 2022, 2023.
    """
    periods = {
        "1999": df[df["reviewYear"] == 1999],
        "2021": df[df["reviewYear"] == 2021],
        "2022": df[df["reviewYear"] == 2022],
        "2023": df[df["reviewYear"] == 2023],
    }

    print("\nShapes for Experiment 2:")
    for period, data in periods.items():
        print(f"{period}: {data.shape}")

    period_trains = {}
    period_tests = {}
    for period, data in periods.items():
        if data.empty:
            print(f"Warning: {period} has no data. Skipping.")
            continue
        print(f"\nProcessing {period}...")
        train_bal, test_bal = split_and_balance(data)
        period_trains[period] = (train_bal, test_bal)
        period_tests[period] = test_bal

    train_sizes = [len(train) for train, _ in period_trains.values()]
    test_sizes = [len(test) for test in period_tests.values()]
    if not train_sizes or not test_sizes:
        raise ValueError("No data available for Experiment 2 after splitting.")
    n_train = min(train_sizes)
    n_test = min(test_sizes)

    print(f"\nSampling each train set to size {n_train} (per experiment period)")
    print(f"Sampling each test set to size {n_test} (per experiment period)")

    for period in period_trains:
        train_bal, test_bal = period_trains[period]
        num_samples_train = n_train // 2
        num_samples_test = n_test // 2

        period_trains[period] = (
            resample_set(train_bal, num_samples_train),
            resample_set(test_bal, num_samples_test),
        )
        period_tests[period] = resample_set(test_bal, num_samples_test)

    return period_trains, period_tests


def prepare_experiment_3(df):
    """
    Prepare data splits for Experiment 3:
    Months in 2022: Jan to Dec.
    """
    month_names = {
        1: "Jan_2022",
        2: "Feb_2022",
        3: "Mar_2022",
        4: "Apr_2022",
        5: "May_2022",
        6: "Jun_2022",
        7: "Jul_2022",
        8: "Aug_2022",
        9: "Sep_2022",
        10: "Oct_2022",
        11: "Nov_2022",
        12: "Dec_2022",
    }

    df_months = {
        month_names[m]: df[(df["reviewYear"] == 2022) & (df["reviewMonth"] == m)]
        for m in range(1, 13)
    }

    print("\nShapes for Experiment 3 (2022 monthly):")
    for m_name, dset in df_months.items():
        print(f"{m_name}: {dset.shape}")

    period_trains = {}
    period_tests = {}
    for m_name, dset in df_months.items():
        if dset.empty:
            print(f"Warning: {m_name} is empty. Skipping.")
            continue
        print(f"\nProcessing {m_name}...")
        train_bal, test_bal = split_and_balance(dset)
        period_trains[m_name] = (train_bal, test_bal)
        period_tests[m_name] = test_bal

    train_sizes = [len(train) for train, _ in period_trains.values()]
    test_sizes = [len(test) for test in period_tests.values()]
    if not train_sizes or not test_sizes:
        raise ValueError("No data available for Experiment 3 after splitting.")
    n_train = min(train_sizes)
    n_test = min(test_sizes)

    print(f"\nSampling each train set to size {n_train} (per experiment period)")
    print(f"Sampling each test set to size {n_test} (per experiment period)")

    for period in period_trains:
        train_bal, test_bal = period_trains[period]
        num_samples_train = n_train // 2
        num_samples_test = n_test // 2

        period_trains[period] = (
            resample_set(train_bal, num_samples_train),
            resample_set(test_bal, num_samples_test),
        )
        period_tests[period] = resample_set(test_bal, num_samples_test)

    return period_trains, period_tests


def main():
    df_raw = load_data(FILE_PATH)
    df = preprocess_data(df_raw)

    label_counts_per_year = (
        df.groupby(["reviewYear", "label"]).size().unstack(fill_value=0)
    )
    print("\nData counts per label per year:")
    print(label_counts_per_year)

    # Experiment 1
    period_trains_1, period_tests_1 = prepare_experiment_1(df)
    run_experiment("Experiment1", period_trains_1, period_tests_1)

    # Experiment 2
    period_trains_2, period_tests_2 = prepare_experiment_2(df)
    run_experiment("Experiment2", period_trains_2, period_tests_2)

    # Experiment 3
    period_trains_3, period_tests_3 = prepare_experiment_3(df)
    run_experiment("Experiment3", period_trains_3, period_tests_3)


if __name__ == "__main__":
    main()
