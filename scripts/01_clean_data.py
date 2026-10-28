"""
Step 1: clean the scraped articles.

- keep the four main outlets
- remove duplicate articles, near-empty pages and articles written in English
- fix missing Süddeutsche publish dates using the article id in the URL
- keep the first year of the war (24.02.2022 - 24.02.2023)
- flag which articles are about the war

Run from the project folder:  python scripts/01_clean_data.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import END, PROCESSED, RAW, RAW_URL, REPORTS, SOURCES, START, WAR_KEYWORDS

MIN_TEXT_CHARS = 200
EPOCH = pd.Timestamp("1970-01-01")
ENGLISH_WORDS = r"\b(?:the|and|of|to|is|are|with|that|this|was)\b"
GERMAN_WORDS = r"\b(?:der|die|und|das|ist|nicht|mit|den|ein|eine)\b"


def get_source(url):
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
    return SOURCES.get(host)


def article_id(url):
    """Süddeutsche urls end in -1.5541744, Bild urls end in -79268562.bild.html"""
    m = re.search(r"-1\.(\d+)(?:\?|$)", url) or re.search(r"-(\d{6,})\.bild\.html", url)
    return float(m.group(1)) if m else np.nan


def recover_dates(df, source, report):
    """Article ids grow over time, so an undated article gets the date of its
    nearest neighbours by id. Checked on held-out articles that do have a date."""
    part = df[df["source"] == source]
    known = part.dropna(subset=["aid", "date"])
    missing = part[part["date"].isna() & part["aid"].notna()]
    if known.empty or missing.empty:
        return df

    X = known[["aid"]].to_numpy()
    y = ((known["date"] - EPOCH) / pd.Timedelta(days=1)).to_numpy()  # days since 1970

    # validation: hide 10% of the dated articles and predict them
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(known))
    test, train = idx[: len(idx) // 10], idx[len(idx) // 10:]
    knn = KNeighborsRegressor(n_neighbors=5).fit(X[train], y[train])
    err_hours = np.abs(knn.predict(X[test]) - y[test]) * 24

    knn.fit(X, y)
    pred = EPOCH + pd.to_timedelta(knn.predict(missing[["aid"]].to_numpy()), unit="D")
    df.loc[missing.index, "date"] = pred.round("min").astype(df["date"].dtype)
    df.loc[missing.index, "date_estimated"] = True

    report[source] = {
        "recovered": int(len(missing)),
        "validation_articles": int(len(test)),
        "median_error_hours": round(float(np.median(err_hours)), 3),
        "p90_error_hours": round(float(np.percentile(err_hours, 90)), 2),
        "share_within_1_day": round(float((err_hours <= 24).mean()), 4),
    }
    print(f"  {source}: recovered {len(missing)} dates, median error "
          f"{np.median(err_hours) * 60:.1f} min, {(err_hours <= 24).mean():.1%} within a day")
    return df


def main():
    if not RAW.exists():
        sys.exit(f"Raw file not found. Download it first:\n  curl -L -o {RAW} '{RAW_URL}'")

    df = pd.read_csv(RAW)
    summary = {"raw_rows": len(df)}
    print(f"Loaded {len(df):,} articles")

    df["source"] = df["URL"].apply(get_source)
    summary["other_sites_removed"] = int(df["source"].isna().sum())
    df = df[df["source"].notna()].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["Text"])
    summary["duplicate_texts_removed"] = before - len(df)

    df["Title"] = df["Title"].fillna("").str.strip()
    df["Text"] = df["Text"].fillna("").str.strip()
    short = df["Text"].str.len() < MIN_TEXT_CHARS
    summary["short_texts_removed"] = int(short.sum())  # mostly video pages
    df = df[~short & (df["Title"] != "")].copy()

    # Spiegel and Bild also publish some articles in English (e.g. spiegel.de/international).
    # The model only understands German, so those are dropped.
    sample = df["Title"] + " " + df["Text"].str.slice(0, 600)
    english = sample.str.count(ENGLISH_WORDS) > sample.str.count(GERMAN_WORDS)
    summary["english_articles_removed"] = int(english.sum())
    df = df[~english].copy()
    print(f"After removing other sites, duplicates, empty pages and English articles: {len(df):,}")

    df["date"] = pd.to_datetime(df["PublishDate"], errors="coerce")
    df["date_estimated"] = False
    summary["missing_dates_by_source"] = df[df["date"].isna()]["source"].value_counts().to_dict()

    print("Recovering missing dates from article ids:")
    df["aid"] = df["URL"].apply(article_id)
    date_report = {}
    # Tried Bild too, but its ids are not in date order (median error almost 2 days),
    # so only Süddeutsche is recovered. The 137 undated Bild articles stay undated.
    df = recover_dates(df, "Süddeutsche Zeitung", date_report)
    summary["date_recovery"] = date_report

    # Spiegel urls have no usable id, so those articles keep no date.
    # They are still used where time does not matter (e.g. overall sentiment per outlet).
    undated = df["date"].isna()
    in_period = df["date"].between(pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=1))
    summary["outside_period_removed"] = int((~undated & ~in_period).sum())
    df = df[undated | in_period].copy()

    pattern = "|".join(WAR_KEYWORDS)
    title_hit = df["Title"].str.lower().str.contains(pattern, regex=True)
    text_hits = df["Text"].str.lower().str.count(pattern)
    # war article = keyword in the headline, or at least 3 mentions in the text
    df["war_related"] = title_hit | (text_hits >= 3)
    df["war_mentions"] = text_hits

    df["month"] = df["date"].dt.to_period("M").astype(str).replace("NaT", "")
    df["title_words"] = df["Title"].str.split().str.len()
    df["text_words"] = df["Text"].str.split().str.len()

    df = df.rename(columns={
        "URL": "url", "Title": "title", "Text": "text",
        "TextSentiment": "text_sentiment_orig", "TextSentimentScore": "text_score_orig",
        "TitleSentiment": "title_sentiment_orig", "TitleSentimentScore": "title_score_orig",
    })
    df["id"] = np.arange(len(df))
    cols = ["id", "url", "source", "date", "date_estimated", "month", "title", "text",
            "title_words", "text_words", "war_related", "war_mentions",
            "text_sentiment_orig", "text_score_orig", "title_sentiment_orig", "title_score_orig"]
    df = df[cols]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "articles_clean.csv", index=False)

    summary["clean_rows"] = len(df)
    summary["war_related"] = int(df["war_related"].sum())
    summary["by_source"] = df["source"].value_counts().to_dict()
    summary["war_by_source"] = df[df["war_related"]]["source"].value_counts().to_dict()
    summary["dated_rows"] = int(df["date"].notna().sum())
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cleaning_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Saved {len(df):,} articles ({df['war_related'].sum():,} about the war) "
          f"to data/processed/articles_clean.csv")


if __name__ == "__main__":
    main()
