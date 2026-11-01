"""
Step 3: analyse the sentiment results and make the charts for the report.
Also writes the smaller file the dashboard reads.

Run from the project folder:  python scripts/03_analysis.py
"""
# Built-in Python tools:
# json = save results as a .json file, sys = change where Python looks for code,
# Path = work with file and folder paths
import json
import re
import sys
from collections import Counter
from pathlib import Path

# matplotlib draws the charts. "Agg" means: save charts as image files,
# don't try to open a window on the screen
import matplotlib
matplotlib.use("Agg")
# plt = drawing charts, pandas = work with tables, seaborn = nicer charts,
# chi2_contingency = a statistics test (explained further down)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

# Tell Python to also look in the main project folder, so it can find "src"
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Get settings (folders, colours, keywords, events) and the word-counting helper
from src.config import (EVENTS, FIGURES, PROCESSED, REPORTS, SENTIMENT_COLORS,
                        SOURCE_COLORS, WAR_KEYWORDS)
from src.text import top_words

# The three sentiment labels, always in this order in the charts
LABELS = ["negative", "neutral", "positive"]
# Use a clean white chart style with a light grid
sns.set_theme(style="whitegrid", font_scale=0.95)


def save(name):
    """Save the chart we just drew as a picture in reports/figures."""
    # Fix spacing so labels don't overlap
    plt.tight_layout()
    # Save as an image (dpi = sharpness)
    plt.savefig(FIGURES / name, dpi=150)
    # Close the chart so the next one starts empty
    plt.close()
    print("  saved", name)


def load():
    """Read the articles and the model results, and join them into one table."""
    # Read the cleaned articles from step 1
    arts = pd.read_csv(PROCESSED / "articles_clean.csv")
    # Turn the date text into real dates (empty or broken dates become "NaT" = no date)
    arts["date"] = pd.to_datetime(arts["date"], errors="coerce", format="mixed")
    # Read the headline results from step 2
    titles = pd.read_csv(PROCESSED / "sentiment_titles.csv")
    # Put "title_" in front of each result column (label -> title_label, ...),
    # then join both tables using the article id
    df = arts.merge(titles.add_prefix("title_").rename(columns={"title_id": "id"}), on="id")
    # Give the label column an easier name
    df = df.rename(columns={"title_label": "title_sentiment"})
    # For the full text, use the label that came with the dataset (it exists for every article)
    df["text_sentiment"] = df["text_sentiment_orig"]
    return df


def share_table(df, by):
    """Make a table of % negative / neutral / positive headlines for each group in "by"."""
    # crosstab counts how often each combination appears;
    # normalize="index" turns the counts into shares per row (each row adds up to 1)
    t = pd.crosstab(df[by], df["title_sentiment"], normalize="index").reindex(columns=LABELS, fill_value=0)
    # Turn shares into percentages with one decimal
    return (t * 100).round(1)


def main():
    # Make sure the folder for the charts exists
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Load all data
    df = load()
    # Smaller tables we use a lot:
    # war = only war articles, dated = only articles with a date, war_dated = both
    war = df[df["war_related"]]
    dated = df[df["date"].notna()]
    war_dated = war[war["date"].notna()]
    # All numbers we find are collected here and saved at the end
    results = {}
    # List of the four outlet names
    sources = list(SOURCE_COLORS)

    # 1. how much of each outlet's output was about the war, month by month
    # Group by month and outlet. The mean of True/False = share of war articles.
    # unstack() turns outlets into columns, * 100 makes it a percentage
    share = dated.groupby(["month", "source"])["war_related"].mean().unstack() * 100
    plt.figure(figsize=(10, 4.8))
    # Draw one line per outlet, in that outlet's colour
    for s in share.columns:
        plt.plot(share.index, share[s], marker="o", ms=3, label=s, color=SOURCE_COLORS[s])
    # Labels, title and legend, then save
    plt.ylabel("% of articles about the war")
    plt.title("Share of coverage about the war, by month")
    plt.xticks(rotation=45)
    plt.legend()
    save("01_war_coverage_share.png")
    # Remember the value in the first and the last month for each outlet
    results["war_share_first_last_month"] = {
        s: [round(share[s].dropna().iloc[0], 1), round(share[s].dropna().iloc[-1], 1)] for s in share.columns
    }

    # 2. headline sentiment split per outlet (war articles)
    # % negative / neutral / positive for each outlet
    split = share_table(war, "source").reindex(sources)
    # Draw stacked bars: one bar per outlet, split into the three sentiment colours
    split.plot(kind="barh", stacked=True, figsize=(9, 4),
               color=[SENTIMENT_COLORS[label] for label in LABELS])
    plt.xlabel("% of war headlines")
    plt.ylabel("")
    plt.title("Headline sentiment per outlet (war articles)")
    plt.legend(loc="lower right", ncol=3)
    save("02_sentiment_by_outlet.png")
    results["war_headline_sentiment_pct"] = split.to_dict(orient="index")

    # is the difference between outlets real or just chance?
    # Count headlines per outlet and label (normal counts, not percentages)
    counts = pd.crosstab(war["source"], war["title_sentiment"])
    # Chi-square test: a very small p-value means the differences are not just luck
    chi2, p, dof, _ = chi2_contingency(counts)
    # Cramér's V shows how BIG the difference is: 0 = no difference, 1 = huge difference
    n = counts.to_numpy().sum()
    cramers_v = (chi2 / (n * (min(counts.shape) - 1))) ** 0.5
    results["chi_square_outlets"] = {"chi2": round(chi2, 1), "dof": int(dof), "p_value": float(p),
                                     "cramers_v": round(cramers_v, 3)}

    # 3. negativity over time
    # New column "neg": 100 if the headline is negative, 0 if not.
    # The mean per month and outlet = % negative headlines
    neg = war_dated.assign(neg=war_dated["title_sentiment"].eq("negative") * 100) \
        .groupby(["month", "source"])["neg"].mean().unstack()
    plt.figure(figsize=(10, 4.8))
    # One line per outlet
    for s in neg.columns:
        plt.plot(neg.index, neg[s], marker="o", ms=3, label=s, color=SOURCE_COLORS[s])
    # Add a dotted vertical line with a label for each important event
    months = list(neg.index)
    for day, name in EVENTS:
        # "2022-04-03" -> "2022-04" (just the month)
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
    # New column "group": "War" for war articles, "Other news" for everything else
    comp = df.assign(group=df["war_related"].map({True: "War", False: "Other news"}))
    # % negative headlines for each outlet and group
    neg_by = comp.assign(neg=comp["title_sentiment"].eq("negative") * 100) \
        .groupby(["source", "group"])["neg"].mean().unstack().reindex(sources)
    # Two bars next to each other per outlet: grey = other news, red = war
    neg_by[["Other news", "War"]].plot(kind="bar", figsize=(8, 4.5), color=["#b8c4ce", "#c0392b"])
    plt.ylabel("% negative headlines")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.title("Negative headlines: war articles vs other news")
    save("04_war_vs_other.png")
    results["negative_pct_war_vs_other"] = neg_by.round(1).to_dict(orient="index")

    # 5. do headline and article body agree?
    # Table: rows = headline label, columns = text label, values = % of the row
    # e.g. "of all negative headlines, how many have a negative text?"
    cross = pd.crosstab(war["title_sentiment"], war["text_sentiment"], normalize="index") \
        .reindex(index=LABELS, columns=LABELS, fill_value=0) * 100
    # Show the table as a coloured grid (darker = higher number)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cross, annot=True, fmt=".0f", cmap="Blues", cbar=False)
    plt.xlabel("Article text sentiment")
    plt.ylabel("Headline sentiment")
    plt.title("Headline vs article text (% of row)")
    save("05_headline_vs_text.png")
    # Share of articles where headline and text have the same label
    results["headline_text_agreement"] = round(float((war["title_sentiment"] == war["text_sentiment"]).mean()), 4)
    results["headline_vs_text_pct"] = cross.round(1).to_dict(orient="index")

    # 6. how sure is the model?
    plt.figure(figsize=(8, 4))
    # For each label, draw a smooth curve showing how the confidence scores are spread out
    # (a peak near 1.0 = the model is usually very sure)
    for label in LABELS:
        sns.kdeplot(war.loc[war["title_sentiment"] == label, "title_confidence"], label=label,
                    color=SENTIMENT_COLORS[label], fill=True, alpha=0.2, clip=(0, 1))
    plt.xlabel("Model confidence")
    plt.title("Confidence of the model, by predicted label (war headlines)")
    plt.legend()
    save("06_confidence.png")
    # Average confidence per label
    results["mean_confidence"] = war.groupby("title_sentiment")["title_confidence"].mean().round(3).to_dict()
    # Share of headlines where the model was unsure (confidence below 60%)
    results["low_confidence_share"] = round(float((war["title_confidence"] < 0.6).mean()), 4)

    # 7. most common words in negative war headlines
    # Skip words like "ukraine" or "putin" - they are in almost every war headline
    stop = set(WAR_KEYWORDS)
    # Four small charts side by side, one per outlet
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    words = {}
    for ax, s in zip(axes, sources):
        # All negative war headlines of this outlet
        titles = war[(war["source"] == s) & (war["title_sentiment"] == "negative")]["title"]
        # The 12 most common words in them
        top = top_words(titles, 12, extra_stop=stop)
        words[s] = top
        # Draw a bar per word ([::-1] flips the order so the most common word is on top)
        ax.barh([w for w, _ in top][::-1], [c for _, c in top][::-1], color=SOURCE_COLORS[s])
        ax.set_title(s, fontsize=10)
    fig.suptitle("Most frequent words in negative war headlines (excluding Ukraine/Russia terms)")
    save("07_negative_words.png")
    results["negative_headline_words"] = words

    # 8. validation sample: our model run vs labels shipped with the dataset
    sample_path = PROCESSED / "sentiment_text_sample.csv"
    # Only do this if step 2 created the sample file
    if sample_path.exists():
        # Join our results with the dataset's labels using the article id
        sample = pd.read_csv(sample_path).merge(df[["id", "text_sentiment_orig"]], on="id")
        # Count how often each pair of labels appears (rows = dataset, columns = ours)
        cm = pd.crosstab(sample["text_sentiment_orig"], sample["label"]) \
            .reindex(index=LABELS, columns=LABELS, fill_value=0)
        # Show it as a coloured grid; the diagonal = both agree
        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
        plt.xlabel("Our run")
        plt.ylabel("Label in original dataset")
        plt.title("Article text: our BERT run vs dataset labels")
        save("08_validation.png")
        # Share of articles where both labels are the same
        results["validation_text_agreement"] = round(float((sample["label"] == sample["text_sentiment_orig"]).mean()), 4)

    # Some basic counts for the report
    results["counts"] = {
        "articles": len(df), "war_articles": len(war), "war_articles_dated": len(war_dated),
        "war_by_source": war["source"].value_counts().to_dict(),
        "headline_sentiment_all_war": war["title_sentiment"].value_counts().to_dict(),
    }
    # Save all results to reports/analysis_results.json
    (REPORTS / "analysis_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    # smaller file for the dashboard (no full texts)
    # Keep only the columns the dashboard needs
    dash = df[["id", "url", "source", "date", "date_estimated", "month", "title", "war_related",
               "title_sentiment", "title_confidence", "title_polarity", "text_sentiment"]].copy()
    # Keep just the first 280 characters of each text as a short preview
    dash["snippet"] = df["text"].str.slice(0, 280)
    # Save it compressed (.gz) so the file is small
    dash.to_csv(PROCESSED / "dashboard_data.csv.gz", index=False, compression="gzip")
    print("Saved reports/analysis_results.json and data/processed/dashboard_data.csv.gz")


# Only run main() when this file is started directly (not when it is imported by another file)
if __name__ == "__main__":
    main()
