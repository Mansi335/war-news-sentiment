"""
Dashboard for the sentiment analysis of German news about the war in Ukraine.

Start it with:  streamlit run app.py
"""
# quote makes text safe to put inside a web link (used for the Google Translate links)
from urllib.parse import quote

# pandas = work with tables, plotly = interactive charts,
# streamlit = turns this Python file into a web page
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Get settings (dates, events, model name, colours, keywords) and the word-counting helper
from src.config import (END, EVENTS, MODEL_NAME, PROCESSED, SENTIMENT_COLORS,
                        SOURCE_COLORS, START, WAR_KEYWORDS)
from src.text import top_words

# Set the browser tab title and use the full width of the screen
st.set_page_config(page_title="War in Ukraine: German media sentiment", layout="wide")

# The three sentiment labels, always in this order
LABELS = ["negative", "neutral", "positive"]
# Shared look for all charts: white background, font size, small margins
PLOT = dict(template="plotly_white", font=dict(size=13), margin=dict(l=10, r=10, t=40, b=10))


# Streamlit re-runs this whole file every time you click something.
# @st.cache_data remembers the result, so the file is only read once (much faster).
@st.cache_data
def load_data():
    # Read the dashboard file made in step 3
    df = pd.read_csv(PROCESSED / "dashboard_data.csv.gz")
    # Turn the date text into real dates (missing dates become "NaT" = no date)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    # Replace empty months with an empty text
    df["month"] = df["month"].fillna("")
    return df


# Load the AI model only once and keep it in memory (it is big and slow to load)
@st.cache_resource(show_spinner="Loading the BERT model (first time takes a minute)...")
def load_model():
    # Imported here so the dashboard still opens even if the model is never used
    from transformers import pipeline
    return pipeline("text-classification", model=MODEL_NAME)


# Load all articles
df = load_data()

# ---------- sidebar ----------
# The filters on the left side of the page
st.sidebar.header("Filters")
# Choose which newspapers to show (all four are selected at the start)
outlets = st.sidebar.multiselect("Outlets", list(SOURCE_COLORS), default=list(SOURCE_COLORS))
# Switch: show only war articles, or all articles
only_war = st.sidebar.toggle("Only articles about the war", value=True)
# Slider to choose a date range (start and end of the study period)
d0, d1 = pd.Timestamp(START).date(), pd.Timestamp(END).date()
period = st.sidebar.slider("Period (dated articles)", min_value=d0, max_value=d1, value=(d0, d1),
                           format="DD.MM.YYYY")
# Small note explaining gaps in the data
st.sidebar.caption(
    "Der Spiegel articles have no publish date in the scraped data, so they appear in the "
    "outlet comparisons but not on the timeline. Bild was only scraped until mid-September 2022."
)

# Apply the filters.
# "view" = the articles that match the filters
view = df[df["source"].isin(outlets)]
if only_war:
    view = view[view["war_related"]]
# Keep articles inside the chosen dates (+1 day so the last day is included) ...
in_period = view["date"].between(pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1))
# ... and also keep articles that have no date at all
view = view[in_period | view["date"].isna()]
# "dated" = only the articles that have a date (needed for the timeline charts)
dated = view[view["date"].notna()]

# ---------- header ----------
# Big title and a short description at the top of the page
st.title("How German media covered the first year of the war in Ukraine")
st.markdown(
    f"Sentiment of **{len(df):,}** articles from Süddeutsche Zeitung, Bild, Der Spiegel and Die Welt, "
    f"24 Feb 2022 to 24 Feb 2023, classified with the German BERT model `{MODEL_NAME}`."
)

# If nothing matches the filters, show a warning and stop here
if view.empty:
    st.warning("No articles match these filters. Select at least one outlet.")
    st.stop()

# Share of negative and positive headlines (mean of True/False = share)
neg = (view["title_sentiment"] == "negative").mean()
pos = (view["title_sentiment"] == "positive").mean()
# Four boxes next to each other with the key numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles shown", f"{len(view):,}")
c2.metric("About the war", f"{view['war_related'].mean():.0%}" if not only_war else "100%")
c3.metric("Negative headlines", f"{neg:.1%}")
c4.metric("Positive headlines", f"{pos:.1%}")

# Create the five tabs of the dashboard
tab_time, tab_outlets, tab_words, tab_articles, tab_model = st.tabs(
    ["Over time", "Outlets", "Words", "Articles", "Try the model"])

# ---------- over time ----------
# Everything inside "with tab_time:" appears in the "Over time" tab
with tab_time:
    # Spiegel has no dates, so if only Spiegel is chosen there is nothing to draw
    if dated.empty:
        st.info("No dated articles for this selection (Der Spiegel has no dates).")
    else:
        # For each month and newspaper: % negative headlines and number of articles
        # (neg = 100 for a negative headline, 0 otherwise, so the average is a percentage)
        monthly = dated.assign(neg=dated["title_sentiment"].eq("negative") * 100) \
            .groupby(["month", "source"]).agg(negative=("neg", "mean"), articles=("id", "size")).reset_index()

        # Line chart: one line per newspaper
        fig = px.line(monthly, x="month", y="negative", color="source", markers=True,
                      color_discrete_map=SOURCE_COLORS,
                      labels={"negative": "% negative headlines", "month": "", "source": ""},
                      title="Share of negative headlines per month")
        # Add a dotted line and a label for each important event
        for day, name in EVENTS:
            # "2022-04-03" -> "2022-04" (just the month)
            m = day[:7]
            if m in set(monthly["month"]):
                fig.add_vline(x=m, line_dash="dot", line_color="#999")
                fig.add_annotation(x=m, y=1, yref="paper", text=name, showarrow=False,
                                   textangle=-90, xanchor="left", yanchor="top", font=dict(size=10, color="#666"))
        # Apply the shared chart style, put the legend below, and show the chart
        fig.update_layout(**PLOT, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # Second chart: how much of each newspaper's news was about the war.
        # This uses ALL articles (not only war ones), otherwise it would always be 100%
        all_dated = df[df["source"].isin(outlets) & df["date"].notna()]
        share = all_dated.groupby(["month", "source"])["war_related"].mean().mul(100).reset_index()
        fig = px.line(share, x="month", y="war_related", color="source", markers=True,
                      color_discrete_map=SOURCE_COLORS,
                      labels={"war_related": "% of all articles", "month": "", "source": ""},
                      title="How much of each outlet's coverage was about the war")
        fig.update_layout(**PLOT, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        # Short explanation under the charts
        st.caption("Attention to the war is highest in the first weeks and then falls, "
                   "while the share of negative headlines stays much more stable.")

# ---------- outlets ----------
with tab_outlets:
    # % negative / neutral / positive per newspaper.
    # melt() changes the table from "wide" (one column per label) to "long" (one row per label),
    # because plotly needs it that way
    split = pd.crosstab(view["source"], view["title_sentiment"], normalize="index") \
        .reindex(columns=LABELS, fill_value=0).mul(100).reset_index() \
        .melt(id_vars="source", var_name="sentiment", value_name="share")
    # Stacked bar chart: one bar per newspaper, split into the three sentiment colours
    fig = px.bar(split, y="source", x="share", color="sentiment", orientation="h",
                 color_discrete_map=SENTIMENT_COLORS, category_orders={"sentiment": LABELS},
                 labels={"share": "% of headlines", "source": "", "sentiment": ""},
                 title="Headline sentiment per outlet", text_auto=".1f")
    fig.update_layout(**PLOT, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    # Compare war articles with other news: % negative headlines for each group
    base = df[df["source"].isin(outlets)]
    comp = base.assign(group=base["war_related"].map({True: "War", False: "Other news"}),
                       neg=base["title_sentiment"].eq("negative") * 100) \
        .groupby(["source", "group"])["neg"].mean().reset_index()
    # Two bars next to each other per newspaper: red = war, grey = other news
    fig = px.bar(comp, x="source", y="neg", color="group", barmode="group",
                 color_discrete_map={"War": "#c0392b", "Other news": "#b8c4ce"},
                 labels={"neg": "% negative headlines", "source": "", "group": ""},
                 title="War articles vs other news")
    fig.update_layout(**PLOT, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

# ---------- words ----------
with tab_words:
    # Split the tab into a small left column (choices) and a wide right column (chart)
    col_a, col_b = st.columns([1, 3])
    with col_a:
        # Choose a newspaper and a sentiment
        w_outlet = st.selectbox("Outlet", outlets)
        w_sent = st.radio("Headlines", LABELS, index=0)
    # Get the headlines that match the choice
    subset = view[(view["source"] == w_outlet) & (view["title_sentiment"] == w_sent)]["title"]
    # Find the 20 most common words (skipping war words like "ukraine")
    words = top_words(subset, 20, extra_stop=WAR_KEYWORDS)
    with col_b:
        if words:
            # Make a small table of words and counts ([::-1] puts the most common word on top)
            wdf = pd.DataFrame(words, columns=["word", "count"]).iloc[::-1]
            # Horizontal bar chart in the newspaper's colour
            fig = px.bar(wdf, x="count", y="word", orientation="h",
                         color_discrete_sequence=[SOURCE_COLORS.get(w_outlet, "#555")],
                         title=f"Most frequent words in {w_sent} {w_outlet} headlines ({len(subset):,})")
            fig.update_layout(**PLOT, height=560, yaxis_title="", xaxis_title="headlines")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No headlines for this selection.")

# ---------- articles ----------
with tab_articles:
    # A search box and a sentiment filter next to each other
    q1, q2 = st.columns([3, 2])
    query = q1.text_input("Search headlines (German)", placeholder="e.g. selenskyj, gas, scholz")
    sent_filter = q2.multiselect("Sentiment", LABELS, default=LABELS)
    # Keep only headlines with the chosen sentiments
    table = view[view["title_sentiment"].isin(sent_filter)]
    # If the user typed something, keep only headlines that contain it
    if query:
        table = table[table["title"].str.contains(query.lower(), regex=False)]
    # Newest articles first
    table = table.sort_values("date", ascending=False)
    # Show at most 500 rows, and add a Google Translate link for each headline
    table = table.head(500).assign(
        translate=lambda t: "https://translate.google.com/?sl=de&tl=en&op=translate&text=" + t["title"].map(quote)
    )
    st.caption(f"{len(table):,} articles shown (max 500). Use the translate link to read a headline in English.")
    # Show the table with nicer column names, a confidence bar, and clickable links
    st.dataframe(
        table[["date", "source", "title", "title_sentiment", "title_confidence", "text_sentiment", "url", "translate"]],
        use_container_width=True, hide_index=True, height=520,
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD.MM.YYYY"),
            "source": "Outlet",
            "title": st.column_config.TextColumn("Headline", width="large"),
            "title_sentiment": "Headline sentiment",
            "title_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
            "text_sentiment": "Text sentiment",
            "url": st.column_config.LinkColumn("Article", display_text="open"),
            "translate": st.column_config.LinkColumn("English", display_text="translate"),
        },
    )

# ---------- try the model ----------
with tab_model:
    st.subheader("Classify any German sentence")
    # One example sentence for each label
    examples = {
        "negative": "Schrecklicher Angriff auf Kiew: Viele Menschen sterben, die Lage ist katastrophal",
        "neutral": "Bundestag berät über das Haushaltsgesetz",
        "positive": "Endlich gute Nachrichten: Die Hilfe kommt an und die Menschen sind dankbar",
    }
    # session_state is Streamlit's memory between clicks.
    # Start with an empty text box the first time
    if "model_text" not in st.session_state:
        st.session_state["model_text"] = ""
    st.caption("Type or paste any German text, or fill in an example:")
    # Three buttons; clicking one puts its example sentence in the text box
    for col, (label, sentence) in zip(st.columns(len(examples)), examples.items()):
        if col.button(f"{label.capitalize()} example", use_container_width=True):
            st.session_state["model_text"] = sentence
    # The text box where the user types (connected to the memory via key="model_text")
    text = st.text_area("German text", key="model_text", height=100,
                        placeholder="z. B. Die Verhandlungen geben Anlass zur Hoffnung")
    st.caption("Note: the model was trained on reviews and social media, so factual news headlines "
               "usually come out neutral. Emotional wording gives clearer positive/negative results.")
    # The "Classify" button (greyed out while the text box is empty)
    if st.button("Classify", type="primary", disabled=not text.strip()):
        # Try to load the model. The online version doesn't install the model libraries
        # (they are too big for the free hosting), so show a friendly message instead
        try:
            clf = load_model()
        except (ImportError, RuntimeError):
            st.info("The live model is not available in the online version of this dashboard. "
                    "Run the app on your own computer with transformers and torch installed to try it.")
        else:
            # Ask the model for the score of every label
            scores = clf(text, top_k=None, truncation=True, max_length=512)
            # Turn it into {label: score} and pick the label with the highest score
            probs = {s["label"]: s["score"] for s in scores}
            label = max(probs, key=probs.get)
            # Show the answer in bold
            st.markdown(f"**Prediction: {label}** ({probs[label]:.1%} confident)")
            # Small bar chart showing all three scores
            pdf = pd.DataFrame({"sentiment": LABELS, "probability": [probs.get(l, 0) for l in LABELS]})
            fig = px.bar(pdf, x="probability", y="sentiment", orientation="h", color="sentiment",
                         color_discrete_map=SENTIMENT_COLORS, range_x=[0, 1], text_auto=".2f")
            fig.update_layout(**PLOT, showlegend=False, height=240, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
