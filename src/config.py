from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw" / "articles_raw.csv"
PROCESSED = BASE / "data" / "processed"
REPORTS = BASE / "reports"
FIGURES = REPORTS / "figures"

RAW_URL = ("https://media.githubusercontent.com/media/LinaYorda/sentiment-analysis-dashboard/"
           "main/data/articles_with_text_and_title_sentiment%20copy.csv")

# first year of the full-scale invasion
START = "2022-02-24"
END = "2023-02-24"

SOURCES = {
    "sueddeutsche.de": "Süddeutsche Zeitung",
    "bild.de": "Bild",
    "spiegel.de": "Der Spiegel",
    "welt.de": "Die Welt",
}

SOURCE_COLORS = {
    "Süddeutsche Zeitung": "#2b6a99",
    "Bild": "#c8102e",
    "Der Spiegel": "#e07b00",
    "Die Welt": "#3c8d5a",
}

SENTIMENT_COLORS = {"negative": "#c0392b", "neutral": "#9aa5b1", "positive": "#2e8b57"}

MODEL_NAME = "oliverguhr/german-sentiment-bert"

# German words that mark an article as being about the war.
# Lowercase stems, so "ukrain" catches ukraine, ukrainisch, ukrainer...
WAR_KEYWORDS = [
    "ukrain", "russland", "russisch", "russen", "putin", "selenskyj", "kiew", "kyjiw",
    "kreml", "moskau", "charkiw", "mariupol", "cherson", "donbass", "donbas", "luhansk",
    "donezk", "bachmut", "krim", "butscha", "saporischschja", "wagner-gruppe",
    "invasion", "angriffskrieg",
]

# a few well known turning points, used as markers on the timeline
EVENTS = [
    ("2022-02-24", "Invasion begins"),
    ("2022-04-03", "Bucha images published"),
    ("2022-09-21", "Russian mobilisation"),
    ("2022-11-11", "Kherson retaken"),
]
