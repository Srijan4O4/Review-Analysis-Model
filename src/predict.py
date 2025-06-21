#!/usr/bin/env python
"""
Inference script for your trained DistilBERT review classifier.

Example usage:
    python predict.py \
        --model_dir path/to/model_dir \
        --text "I love this product, works great!" \
        --rating 5 \
        --max_length 256
"""
import os
import argparse
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description="Load trained DistilBERT classifier and predict.")
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory containing the saved tokenizer and model (from save_pretrained)."
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="The review text to classify."
    )
    parser.add_argument(
        "--rating", type=int, default=None,
        help="Optional numeric rating to append (e.g., 5)."
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="Max token length (should match training)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Load tokenizer and model from the output directory
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Prepare the review text (append rating if provided)
    review = args.text
    if args.rating is not None:
        review = f"{review} Rating: {args.rating}"

    # Tokenize the input
    enc = tokenizer(
        review,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        pred_id = int(torch.argmax(logits, dim=-1).item())

    # Map prediction to label
    label_map = {
        0: "Human-written",
        1: "AI-generated"
    }
    label = label_map.get(pred_id, "Unknown")

    print(f"[Result] {label}")
    print(f"         Text: {review}")

if __name__ == "__main__":
    main()
