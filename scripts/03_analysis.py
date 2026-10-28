"""
Step 3: analyse the sentiment results and make the charts for the report.
Also writes the smaller file the dashboard reads.

Run from the project folder:  python scripts/03_analysis.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (EVENTS, FIGURES, PROCESSED, REPORTS, SENTIMENT_COLORS,
                        SOURCE_COLORS, WAR_KEYWORDS)
from src.text import top_words

LABELS = ["negative", "neutral", "positive"]
sns.set_theme(style="whitegrid", font_scale=0.95)


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=150)
    plt.close()
    print("  saved", name)


def load():
    arts = pd.read_csv(PROCESSED / "articles_clean.csv")
    arts["date"] = pd.to_datetime(arts["date"], errors="coerce", format="mixed")
    titles = pd.read_csv(PROCESSED / "sentiment_titles.csv")
    df = arts.merge(titles.add_prefix("title_").rename(columns={"title_id": "id"}), on="id")
    df = df.rename(columns={"title_label": "title_sentiment"})
    df["text_sentiment"] = df["text_sentiment_orig"]
    return df


def share_table(df, by):
    t = pd.crosstab(df[by], df["title_sentiment"], normalize="index").reindex(columns=LABELS, fill_value=0)
    return (t * 100).round(1)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load()
    war = df[df["war_related"]]
    dated = df[df["date"].notna()]
    war_dated = war[war["date"].notna()]
    results = {}
    sources = list(SOURCE_COLORS)

    # 1. how much of each outlet's output was about the war, month by month
    share = dated.groupby(["month", "source"])["war_related"].mean().unstack() * 100
    plt.figure(figsize=(10, 4.8))
    for s in share.columns:
        plt.plot(share.index, share[s], marker="o", ms=3, label=s, color=SOURCE_COLORS[s])
    plt.ylabel("% of articles about the war")
    plt.title("Share of coverage about the war, by month")
    plt.xticks(rotation=45)
    plt.legend()
    save("01_war_coverage_share.png")
    results["war_share_first_last_month"] = {
        s: [round(share[s].dropna().iloc[0], 1), round(share[s].dropna().iloc[-1], 1)] for s in share.columns
    }

    # 2. headline sentiment split per outlet (war articles)
    split = share_table(war, "source").reindex(sources)
    split.plot(kind="barh", stacked=True, figsize=(9, 4),
               color=[SENTIMENT_COLORS[label] for label in LABELS])
    plt.xlabel("% of war headlines")
    plt.ylabel("")
    plt.title("Headline sentiment per outlet (war articles)")
    plt.legend(loc="lower right", ncol=3)
    save("02_sentiment_by_outlet.png")
    results["war_headline_sentiment_pct"] = split.to_dict(orient="index")

    # is the difference between outlets real or just chance?
    counts = pd.crosstab(war["source"], war["title_sentiment"])
    chi2, p, dof, _ = chi2_contingency(counts)
    n = counts.to_numpy().sum()
    cramers_v = (chi2 / (n * (min(counts.shape) - 1))) ** 0.5
    results["chi_square_outlets"] = {"chi2": round(chi2, 1), "dof": int(dof), "p_value": float(p),
                                     "cramers_v": round(cramers_v, 3)}

    # 3. negativity over time
    neg = war_dated.assign(neg=war_dated["title_sentiment"].eq("negative") * 100) \
        .groupby(["month", "source"])["neg"].mean().unstack()
    plt.figure(figsize=(10, 4.8))
    for s in neg.columns:
        plt.plot(neg.index, neg[s], marker="o", ms=3, label=s, color=SOURCE_COLORS[s])
    months = list(neg.index)
    for day, name in EVENTS:
        m = day[:7]
        if m in months:
            plt.axvline(months.index(m), color="#888", ls=":", lw=1)
            plt.text(months.index(m) + 0.1, plt.ylim()[1] * 0.97, name, fontsize=7, va="top", rotation=90)
    plt.ylabel("% negative headlines")
    plt.title("Negative war headlines per month")
    plt.xticks(rotation=45)
    plt.legend()
    save("03_negativity_over_time.png")
    results["monthly_negative_pct"] = neg.round(1).to_dict(orient="index")

    # 4. war vs other news
    comp = df.assign(group=df["war_related"].map({True: "War", False: "Other news"}))
    neg_by = comp.assign(neg=comp["title_sentiment"].eq("negative") * 100) \
        .groupby(["source", "group"])["neg"].mean().unstack().reindex(sources)
    neg_by[["Other news", "War"]].plot(kind="bar", figsize=(8, 4.5), color=["#b8c4ce", "#c0392b"])
    plt.ylabel("% negative headlines")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.title("Negative headlines: war articles vs other news")
    save("04_war_vs_other.png")
    results["negative_pct_war_vs_other"] = neg_by.round(1).to_dict(orient="index")

    # 5. do headline and article body agree?
    cross = pd.crosstab(war["title_sentiment"], war["text_sentiment"], normalize="index") \
        .reindex(index=LABELS, columns=LABELS, fill_value=0) * 100
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cross, annot=True, fmt=".0f", cmap="Blues", cbar=False)
    plt.xlabel("Article text sentiment")
    plt.ylabel("Headline sentiment")
    plt.title("Headline vs article text (% of row)")
    save("05_headline_vs_text.png")
    results["headline_text_agreement"] = round(float((war["title_sentiment"] == war["text_sentiment"]).mean()), 4)
    results["headline_vs_text_pct"] = cross.round(1).to_dict(orient="index")

    # 6. how sure is the model?
    plt.figure(figsize=(8, 4))
    for label in LABELS:
        sns.kdeplot(war.loc[war["title_sentiment"] == label, "title_confidence"], label=label,
                    color=SENTIMENT_COLORS[label], fill=True, alpha=0.2, clip=(0, 1))
    plt.xlabel("Model confidence")
    plt.title("Confidence of the model, by predicted label (war headlines)")
    plt.legend()
    save("06_confidence.png")
    results["mean_confidence"] = war.groupby("title_sentiment")["title_confidence"].mean().round(3).to_dict()
    results["low_confidence_share"] = round(float((war["title_confidence"] < 0.6).mean()), 4)

    # 7. most common words in negative war headlines
    stop = set(WAR_KEYWORDS)
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    words = {}
    for ax, s in zip(axes, sources):
        titles = war[(war["source"] == s) & (war["title_sentiment"] == "negative")]["title"]
        top = top_words(titles, 12, extra_stop=stop)
        words[s] = top
        ax.barh([w for w, _ in top][::-1], [c for _, c in top][::-1], color=SOURCE_COLORS[s])
        ax.set_title(s, fontsize=10)
    fig.suptitle("Most frequent words in negative war headlines (excluding Ukraine/Russia terms)")
    save("07_negative_words.png")
    results["negative_headline_words"] = words

    # 8. validation sample: our model run vs labels shipped with the dataset
    sample_path = PROCESSED / "sentiment_text_sample.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path).merge(df[["id", "text_sentiment_orig"]], on="id")
        cm = pd.crosstab(sample["text_sentiment_orig"], sample["label"]) \
            .reindex(index=LABELS, columns=LABELS, fill_value=0)
        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
        plt.xlabel("Our run")
        plt.ylabel("Label in original dataset")
        plt.title("Article text: our BERT run vs dataset labels")
        save("08_validation.png")
        results["validation_text_agreement"] = round(float((sample["label"] == sample["text_sentiment_orig"]).mean()), 4)

    results["counts"] = {
        "articles": len(df), "war_articles": len(war), "war_articles_dated": len(war_dated),
        "war_by_source": war["source"].value_counts().to_dict(),
        "headline_sentiment_all_war": war["title_sentiment"].value_counts().to_dict(),
    }
    (REPORTS / "analysis_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    # smaller file for the dashboard (no full texts)
    dash = df[["id", "url", "source", "date", "date_estimated", "month", "title", "war_related",
               "title_sentiment", "title_confidence", "title_polarity", "text_sentiment"]].copy()
    dash["snippet"] = df["text"].str.slice(0, 280)
    dash.to_csv(PROCESSED / "dashboard_data.csv.gz", index=False, compression="gzip")
    print("Saved reports/analysis_results.json and data/processed/dashboard_data.csv.gz")


if __name__ == "__main__":
    main()
