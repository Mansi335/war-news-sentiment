# German Media and the War in Ukraine: a Sentiment Analysis Dashboard

How did German newspapers write about the first year of Russia's full-scale invasion of Ukraine
(24 Feb 2022 – 24 Feb 2023)? This project cleans about 144,000 scraped articles from
Süddeutsche Zeitung, Bild, Der Spiegel and Die Welt, classifies every headline with a German BERT
sentiment model, and shows the results in an interactive dashboard.

Based on the dataset from [LinaYorda/sentiment-analysis-dashboard](https://github.com/LinaYorda/sentiment-analysis-dashboard),
which was scraped with the [newspaper-scraper](https://pypi.org/project/newspaper-scraper/) library.

## Quick start (Mac)

Double-click `start.command`, or in Terminal:

```bash
cd war-news-sentiment
./start.command
```

The first run creates a virtual environment, installs the libraries, downloads the data (400 MB)
and the model (440 MB), runs the pipeline and opens the dashboard at http://localhost:8501.
After that it starts in a few seconds.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_clean_data.py` | Keeps the 4 outlets, removes duplicates, empty pages and English articles, recovers missing Süddeutsche dates, keeps the study period, flags war articles |
| 2 | `scripts/02_run_sentiment.py` | Runs `oliverguhr/german-sentiment-bert` on every headline and on a 600-article sample of full texts |
| 3 | `scripts/03_analysis.py` | Statistics, chi-square test, charts for the report (`reports/figures`) and the dashboard data |
| – | `app.py` | Streamlit dashboard |

To run the steps by hand:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/01_clean_data.py
python scripts/02_run_sentiment.py
python scripts/03_analysis.py
streamlit run app.py
```

## Methods in short

- **Cleaning:** 476 articles from other sites, 1,103 duplicate texts, near-empty pages (videos),
  and articles written in English are removed.
- **Missing dates:** all Der Spiegel and about 20,000 Süddeutsche articles have no publish date.
  Süddeutsche article ids increase over time, so the date of an undated article is estimated from
  its 5 nearest neighbours by id (k-nearest-neighbours regression). Tested on 10% held-out dated
  articles before use. Spiegel urls carry no id, so Spiegel is left out of the timeline.
- **War filter:** an article counts as war-related if its headline contains a war keyword
  (Ukraine, Russland, Putin, Kiew, Selenskyj, ...) or the text mentions them at least 3 times.
- **Sentiment:** German BERT returns positive / neutral / negative with a probability for each.
  Polarity = P(positive) − P(negative).
- **Check:** our labels are compared with the labels that came with the dataset.
- **Statistics:** chi-square test of independence between outlet and headline sentiment,
  with Cramér's V as effect size.

## Dashboard

- **Over time:** monthly share of negative headlines per outlet with key events marked, and how much
  of each outlet's coverage was about the war
- **Outlets:** sentiment split per outlet, war vs other news
- **Words:** most frequent words in positive/negative headlines per outlet
- **Articles:** searchable table with links to the article and an English translation
- **Try the model:** type any German sentence and classify it live

## Limitations

- BERT reads at most 512 tokens, so text sentiment uses only the start of long articles.
- The keyword filter is simple and also catches economy articles that mention the war.
- Der Spiegel has no dates; Bild was scraped only until mid-September 2022.
- Most news headlines are neutral by style, so differences between outlets are in the
  negative share rather than positive vs negative.
