import re
from collections import Counter

# common German words that say nothing about the topic
GERMAN_STOPWORDS = set("""
aber alle allem allen aller alles als also am an ander andere anderen auch auf aus bei beim
bin bis bist da damit dann das dass dem den denn der des die dies diese diesem diesen dieser
dieses doch dort du durch ein eine einem einen einer eines er es etwas euch für gegen gibt hat
hatte hätte haben hier hin hinter ich ihr ihm ihn ihnen ihre im in ist ja jetzt kann kein keine
mal man mehr mein mit muss nach nach nicht nichts noch nun nur ob oder ohne sehr sein seine
sich sie sind so soll sollen über um und uns unser unter vom von vor war waren was weil welt
wenn wer werden wie wieder will wir wird wo wurde wurden zu zum zur zwei drei neue neuen
bild spiegel sz sagt sagte heute live news ticker liveticker newsblog alle wegen geht gibt
seit ab soll steht macht viele erst kommt schon neues video jahre jahr tag tage
immer weiter laut offenbar aktuellen aktuelle nacht morgen abend
""".split())

# recurring column and newsletter names that are not about the content
FORMAT_WORDS = set("""
nachrichten szpodcast podcast punkt lage kolumne kommentar tagebuch newsletter
fair the for and of to in on is
""".split())
GERMAN_STOPWORDS |= FORMAT_WORDS

WORD = re.compile(r"[a-zäöüß]{3,}")


def tokens(text, extra_stop=()):
    stop = GERMAN_STOPWORDS | set(extra_stop)
    words = WORD.findall(str(text).lower())
    return [w for w in words if w not in stop and not any(w.startswith(s) for s in extra_stop)]


def top_words(texts, n=15, extra_stop=()):
    counts = Counter()
    for t in texts:
        counts.update(tokens(t, extra_stop))
    return counts.most_common(n)
