"""
Step 2: run the German BERT sentiment model ourselves.

- every headline is classified (positive / neutral / negative + probabilities)
- a random sample of full articles is classified too, to check that our run
  agrees with the labels that came with the scraped dataset

The model (about 440 MB) is downloaded from Hugging Face the first time.
On a MacBook this takes a few minutes; it uses the Apple GPU (mps) if available.

Run from the project folder:  python scripts/02_run_sentiment.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_NAME, PROCESSED, REPORTS

TEXT_SAMPLE_SIZE = 600
BATCH_SIZE = 64
SEED = 42


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return -1  # cpu


def classify(clf, texts, batch_size, max_length):
    """Returns a dataframe with the label and the probability of each class."""
    out = clf(list(texts), batch_size=batch_size, truncation=True, max_length=max_length, top_k=None)
    rows = []
    for scores in out:
        probs = {s["label"]: s["score"] for s in scores}
        label = max(probs, key=probs.get)
        rows.append({
            "label": label,
            "confidence": probs[label],
            "p_positive": probs.get("positive", 0.0),
            "p_neutral": probs.get("neutral", 0.0),
            "p_negative": probs.get("negative", 0.0),
        })
    df = pd.DataFrame(rows)
    # one number per article: +1 fully positive, -1 fully negative
    df["polarity"] = df["p_positive"] - df["p_negative"]
    return df


def main():
    articles = pd.read_csv(PROCESSED / "articles_clean.csv",
                           usecols=["id", "title", "text", "war_related",
                                    "text_sentiment_orig", "title_sentiment_orig"])
    device = pick_device()
    print(f"Loading {MODEL_NAME} on {device} ...")
    clf = pipeline("text-classification", model=MODEL_NAME, device=device)

    # headlines are short, so 64 tokens is plenty
    t0 = time.time()
    print(f"Classifying {len(articles):,} headlines ...")
    titles = classify(clf, articles["title"], BATCH_SIZE, 64)
    titles.insert(0, "id", articles["id"].to_numpy())
    title_secs = time.time() - t0
    titles.to_csv(PROCESSED / "sentiment_titles.csv", index=False)
    print(f"  done in {title_secs / 60:.1f} min")

    # full texts are slow, so only a sample of war articles; BERT reads the first 512 tokens
    sample = articles[articles["war_related"]].sample(TEXT_SAMPLE_SIZE, random_state=SEED)
    t0 = time.time()
    print(f"Classifying {len(sample)} full articles (first 512 tokens) ...")
    texts = classify(clf, sample["text"], 8, 512)
    texts.insert(0, "id", sample["id"].to_numpy())
    text_secs = time.time() - t0
    texts.to_csv(PROCESSED / "sentiment_text_sample.csv", index=False)
    print(f"  done in {text_secs / 60:.1f} min")

    title_agree = (titles["label"].to_numpy() == articles["title_sentiment_orig"].to_numpy()).mean()
    text_agree = (texts["label"].to_numpy() == sample["text_sentiment_orig"].to_numpy()).mean()
    run = {
        "model": MODEL_NAME,
        "device": str(device),
        "headlines": len(titles),
        "headline_minutes": round(title_secs / 60, 2),
        "headlines_per_second": round(len(titles) / title_secs, 1),
        "text_sample": len(texts),
        "text_minutes": round(text_secs / 60, 2),
        "agreement_with_dataset_titles": round(float(title_agree), 4),
        "agreement_with_dataset_texts": round(float(text_agree), 4),
    }
    (REPORTS / "sentiment_run.json").write_text(json.dumps(run, indent=2))
    print(f"Agreement with the labels in the original dataset: "
          f"headlines {title_agree:.1%}, article texts {text_agree:.1%}")


if __name__ == "__main__":
    main()
