#!/bin/bash
# Double-click this file on a Mac (or run ./start.command) to set everything up and open the dashboard.
# The first run downloads the data and the model, so it takes about 10-15 minutes.

cd "$(dirname "$0")" || exit 1
set -e

if [ ! -d venv ]; then
    echo "Creating Python environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "Installing libraries (only slow the first time)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ ! -f data/raw/articles_raw.csv ]; then
    echo "Downloading the scraped articles (400 MB)..."
    mkdir -p data/raw
    curl -L -o data/raw/articles_raw.csv \
        "https://media.githubusercontent.com/media/LinaYorda/sentiment-analysis-dashboard/main/data/articles_with_text_and_title_sentiment%20copy.csv"
fi

[ -f data/processed/articles_clean.csv ] || python scripts/01_clean_data.py
[ -f data/processed/sentiment_titles.csv ] || python scripts/02_run_sentiment.py
[ -f data/processed/dashboard_data.csv.gz ] || python scripts/03_analysis.py

echo "Opening the dashboard at http://localhost:8501 (press Ctrl+C here to stop it)"
streamlit run app.py
