from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
# from sklearn.metrics import accuracy_score
import evaluate
from datasets import Dataset
import random
import logging
import sys
import argparse
import os
import torch
from io import BytesIO
import pandas as pd
from itertools import chain
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--max_samples", type=int)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    


    # load datasets
    # obj = s3_client.get_object(
    #     Bucket=BUCKET,
    #     Key=f'hackweek-2023/train/train.csv'
    # )
    # df_train_from_s3 = pd.read_csv(BytesIO(obj['Body'].read()))

    # obj = s3_client.get_object(
    #     Bucket=BUCKET,
    #     Key=f'hackweek-2023/test/test.csv'
    # )
    # df_test_from_s3 = pd.read_csv(BytesIO(obj['Body'].read()))
    
    train_df = pd.read_csv(f'{args.training_dir}/train.csv')
    test_df = pd.read_csv(f'{args.test_dir}/test.csv')

    train_dataset = Dataset.from_dict({"content": train_df.values.tolist()})
    test_dataset = Dataset.from_dict({"content": test_df.values.tolist()})
    
    if args.max_samples is not None:
        max_samples = min(len(train_dataset), args.max_samples)
        train_dataset = train_dataset.select(range(max_samples))
        max_samples = min(len(test_dataset), args.max_samples)
        test_dataset = test_dataset.select(range(max_samples))
    
    # Load pretrained model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        add_prefix_space=True,
        low_cpu_mem_usage=True
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = list(train_dataset.features)
    text_column_name = "content"
    
    max_seq_length = tokenizer.model_max_length
    
    # We tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], is_split_into_words=True, return_special_tokens_mask=True)
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on every text in train dataset",
    )
    
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on every text in test dataset",
    )
    
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    train_dataset = train_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )
    
    test_dataset = test_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)
    
    
    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of = None,
    )
    
    # Define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    # Training
    train_result = trainer.train()
    trainer.save_model(args.model_dir) # Saves the model to s3
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluate model
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(test_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(metrics.items()):
            writer.write(f"{key} = {value}\n")

    kwargs = {"finetuned_from": args.model_name, "tasks": "fill-mask"}
    trainer.create_model_card(**kwargs)