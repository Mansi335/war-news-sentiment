"""
Step 2: run the German BERT sentiment model ourselves.

- every headline is classified (positive / neutral / negative + probabilities)
- a random sample of full articles is classified too, to check that our run
  agrees with the labels that came with the scraped dataset

The model (about 440 MB) is downloaded from Hugging Face the first time.
On a MacBook this takes a few minutes; it uses the Apple GPU (mps) if available.

Run from the project folder:  python scripts/02_run_sentiment.py
"""
# Built-in Python tools:
# json = save results as a .json file, sys = change where Python looks for code,
# time = measure how long things take, Path = work with file and folder paths
import json
import sys
import time
from pathlib import Path

# Extra libraries:
# pandas = work with tables, torch = runs the AI model,
# pipeline = an easy way to load a ready-made AI model from Hugging Face
import pandas as pd
import torch
from transformers import pipeline

# Tell Python to also look in the main project folder, so it can find "src"
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Get the model name and folder locations from our settings file (src/config.py)
from src.config import MODEL_NAME, PROCESSED, REPORTS

# How many full articles we check (checking all of them would take too long)
TEXT_SAMPLE_SIZE = 600
# How many headlines the model reads at the same time (bigger = faster, but uses more memory)
BATCH_SIZE = 64
# A fixed number so the "random" sample is the same every time we run the script
SEED = 42


def pick_device():
    """Choose the fastest hardware this computer has for running the model."""
    # "mps" = the graphics chip on newer Macs (fast)
    if torch.backends.mps.is_available():
        return "mps"
    # 0 = an NVIDIA graphics card (fast), if the computer has one
    if torch.cuda.is_available():
        return 0
    # -1 = the normal processor (slowest, but works everywhere)
    return -1  # cpu


def classify(clf, texts, batch_size, max_length):
    """Returns a dataframe with the label and the probability of each class."""
    # Give all texts to the model.
    # truncation=True cuts texts that are too long, max_length is that limit (in tokens, roughly words),
    # top_k=None means "give me the score for every label, not just the winner"
    out = clf(list(texts), batch_size=batch_size, truncation=True, max_length=max_length, top_k=None)
    # We will build one row of results per text
    rows = []
    for scores in out:
        # Turn the model's answer into a simple dictionary,
        # e.g. {"positive": 0.1, "neutral": 0.2, "negative": 0.7}
        probs = {s["label"]: s["score"] for s in scores}
        # The final answer is the label with the highest score
        label = max(probs, key=probs.get)
        # Save the answer, how sure the model is, and all three scores
        rows.append({
            "label": label,
            "confidence": probs[label],
            "p_positive": probs.get("positive", 0.0),
            "p_neutral": probs.get("neutral", 0.0),
            "p_negative": probs.get("negative", 0.0),
        })
    # Turn the list of rows into a table
    df = pd.DataFrame(rows)
    # one number per article: +1 fully positive, -1 fully negative
    df["polarity"] = df["p_positive"] - df["p_negative"]
    return df


def main():
    # Read the cleaned articles from step 1 (only the columns we need, to save memory)
    articles = pd.read_csv(PROCESSED / "articles_clean.csv",
                           usecols=["id", "title", "text", "war_related",
                                    "text_sentiment_orig", "title_sentiment_orig"])
    # Pick the hardware and load the German sentiment model
    # (the first time, it is downloaded from the internet)
    device = pick_device()
    print(f"Loading {MODEL_NAME} on {device} ...")
    clf = pipeline("text-classification", model=MODEL_NAME, device=device)

    # --- Part 1: classify every headline ---
    # headlines are short, so 64 tokens is plenty
    # Start a timer
    t0 = time.time()
    print(f"Classifying {len(articles):,} headlines ...")
    titles = classify(clf, articles["title"], BATCH_SIZE, 64)
    # Add the article id as the first column, so we know which result belongs to which article
    titles.insert(0, "id", articles["id"].to_numpy())
    # Stop the timer
    title_secs = time.time() - t0
    # Save the headline results to a file
    titles.to_csv(PROCESSED / "sentiment_titles.csv", index=False)
    print(f"  done in {title_secs / 60:.1f} min")

    # --- Part 2: classify a sample of full articles ---
    # full texts are slow, so only a sample of war articles; BERT reads the first 512 tokens
    # Pick 600 random war articles
    sample = articles[articles["war_related"]].sample(TEXT_SAMPLE_SIZE, random_state=SEED)
    t0 = time.time()
    print(f"Classifying {len(sample)} full articles (first 512 tokens) ...")
    # Long texts need more memory, so only 8 at a time
    texts = classify(clf, sample["text"], 8, 512)
    # Add the article id, stop the timer and save the results
    texts.insert(0, "id", sample["id"].to_numpy())
    text_secs = time.time() - t0
    texts.to_csv(PROCESSED / "sentiment_text_sample.csv", index=False)
    print(f"  done in {text_secs / 60:.1f} min")

    # --- Part 3: compare our results with the labels that came with the dataset ---
    # Compare row by row: True where both labels are the same.
    # .mean() of True/False values = the share that matched (e.g. 0.85 = 85%)
    title_agree = (titles["label"].to_numpy() == articles["title_sentiment_orig"].to_numpy()).mean()
    text_agree = (texts["label"].to_numpy() == sample["text_sentiment_orig"].to_numpy()).mean()
    # Put the details of this run (speed, counts, agreement) together
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
    # Save the run details to reports/sentiment_run.json and print the result
    (REPORTS / "sentiment_run.json").write_text(json.dumps(run, indent=2))
    print(f"Agreement with the labels in the original dataset: "
          f"headlines {title_agree:.1%}, article texts {text_agree:.1%}")


# Only run main() when this file is started directly (not when it is imported by another file)
if __name__ == "__main__":
    main()
