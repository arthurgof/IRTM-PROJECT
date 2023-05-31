# %% [markdown]
# # 0. Imports

# %%
import gc
from multiprocessing import cpu_count, Pool
from plotly.offline import init_notebook_mode
from bertopic import BERTopic
from bertopic import *
import nltk

# nltk.download()
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from collections import Counter
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import string
import itertools
from time import sleep
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from plotly.offline import init_notebook_mode
from bertopic import BERTopic
from bertopic import *
from bertopic.representation import KeyBERTInspired
from bertopic.representation import ZeroShotClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection, cos_sim
import torch

# %% [markdown]
# # 1. Clustering to retrieve sexual sequences

# %% [markdown]
# Get the data

# %%
f = open("gameofthrones.txt", "r", encoding="utf-8")
text_list = [line for line in f.readlines()]
delta = 5
text = " ".join(text_list)

# %%
# Define a list of words in the lexical field of sex
sex_lexicon = [
    "sex",
    "sexual",
    "intimacy",
    "passion",
    "romance",
    "affair",
    "liaison",
    "fling",
    "tryst",
    "encounter",
    "activity",
    "connection",
    "episode",
    "intercourse",
    "coitus",
    "lust",
    "desire",
    "pleasure",
    "sensual",
    "erotic",
    "seduction",
    "seductive",
    "seduce",
    "seducer",
    "seductress",
    "arousal",
    "orgasm",
    "foreplay",
    "kiss",
    "naked",
    "nudity",
    "pornography",
    "prostitute",
    "whore",
    "brothel",
    "virginity",
    "impotence",
    "erection",
    "masturbation",
    "ejaculation",
    "contraception",
    "abortion",
    "homosexuality",
    "bisexuality",
    "transgender",
    "queer",
    "pansexual",
    "pansexuality",
]
synonyms = [
    "rape",
    "romance",
    "passion",
    "lust",
    "affair",
    "liaison",
    "kiss",
    "fuck",
    "fucked",
    "intercourse",
    "coitus",
    "breast",
    "penis",
    "masturbation",
    "nipples",
    "whore",
]
sex_lexicon.extend(synonyms)
f = open("gameofthrones.txt", "r", encoding="utf-8")
text = [line for line in f.readlines()]
text_Sexual = []
delta = 0
for i in range(len(text)):
    if any(x in text[i] for x in sex_lexicon) or any(x in text[i] for x in sex_lexicon):
        # text_Sexual.append(".".join(text[i-delta : i+delta]))
        text_Sexual.append(text[i])

# %%
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(text_Sexual)

# %%
clusters = community_detection(torch.tensor(embeddings), threshold=0.5)
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", text_Sexual[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", text_Sexual[sentence_id])

# %%
text_aggregation = []
for i in range(len(text)):
    sentence = text[i]
    id1 = id2 = i
    while len([*sentence]) < 250:
        id1 -= 1
        id2 += 1
        if id2 >= len(text):
            break
        if id1 < 0:
            break
        trsent1 = text[id1]
        trsent2 = text[id2]
        sentence = "".join([trsent1, sentence, trsent2])
    text_aggregation.append(sentence)
bookEmbeding = model.encode(text_aggregation)

# %%
cluster_descriptions = {
    8: "Intimate moments",
    10: "Brothels and desire",
    15: "Cunning remarks",
    21: "Disturbing coercion",
    23: "Traditions and sex",
    27: "Suggestive tension",
}

# %%
sexualCluster = [8, 10, 15, 21, 23, 27]
# now let make a classification model
df = pd.DataFrame(columns=["text", "cluster", "similarity"])
total = 0
for j in range(len(bookEmbeding)):
    maxcluster = -1
    maxsimilarity = -1
    for i, cluster in enumerate(clusters):
        similarity = cos_sim(embeddings[cluster], bookEmbeding[j])
        if similarity.mean() > maxsimilarity:
            maxsimilarity = similarity.mean()
            maxcluster = i
    if (maxcluster + 1) in sexualCluster:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        [
                            text_aggregation[j],
                            cluster_descriptions[maxcluster + 1],
                            maxsimilarity.item(),
                        ]
                    ],
                    columns=["text", "cluster", "similarity"],
                ),
            ],
            ignore_index=True,
        )
df.head()

# %%
df = df[df.similarity > 0.35]
df.reset_index(inplace=True)
df

# %%
df["Characters_nltk"] = ""
df["Places_nltk"] = ""
df["Characters_bert"] = ""
df["Places_bert"] = ""

# %% [markdown]
# # 3. NER

# %% [markdown]
# ### Scraping

# %% [markdown]
# In order to normalize our results, we need the true values of the names. We use a scraping method to get them.

# %%
# Make a request to the webpage

urls = "https://iceandfire.fandom.com/wiki/Category:Characters?from="
alphabet = list(string.ascii_uppercase)

soups = []
characters = []

for i in range(26):
    url = urls + (alphabet[i])
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        html = response.read()

    soup = BeautifulSoup(html, "html.parser")
    soups.append(soup)

characters = []

for i in range(26):
    links = soups[i].find_all("a", {"class": "category-page__member-link"})

    for link in links:
        name = link.get_text().strip()
        if name != "":
            characters.append(name)

    # Print the list of character names
print(characters)

# %% [markdown]
# ## 3.1 NER with NLTK


# %%
def nltk_ner(sentence):
    nltk_results = ne_chunk(pos_tag(word_tokenize(sentence)))

    person = []
    organization = []
    gpe = []
    facility = []
    gsp = []
    location = []

    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ""
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + " "
                match nltk_result.label():
                    case "PERSON":
                        person.append(name)
                    case "ORGANIZATION":
                        organization.append(name)
                    case "GPE":
                        gpe.append(name)
                    case "GSP":
                        gsp.append(name)
                    case "LOCATION":
                        location.append(name)
                    case "FACILITY":
                        facility.append(name)
        # print ('Type: ', nltk_result.label(), 'Name: ', name)
    return person, location


# %%
# Apply ner to each row

for index in tqdm(range(len(df))):
    pers, loc = nltk_ner(df.loc[index, "text"])
    df.loc[index, "Characters_nltk"] = str(pers)
    df.loc[index, "Places_nltk"] = str(loc)


# %%
def similarity(one, two):
    # Create a corpus of the two words
    corpus = [one, two]

    # Create a TfidfVectorizer with a custom tokenizer that returns the input as a list of characters
    vectorizer = TfidfVectorizer(tokenizer=lambda x: list(x), lowercase=False)

    # Compute the soft TF-IDF weights
    tfidf = vectorizer.fit_transform(corpus)

    # Compute the cosine similarity between the two words
    similarity = cosine_similarity(tfidf[0], tfidf[1])[0][0]

    return similarity


# %%
df.Characters_nltk = df.Characters_nltk.apply(lambda x: eval(x))
df.Places_nltk = df.Places_nltk.apply(lambda x: eval(x))

# %%
# Loop through rows to nomalize each list


def sim(persons):
    unique_chars = set()
    threshold = 0.7
    for i in range(len(characters)):
        for j in range(len(persons)):
            char1 = persons[j]
            char2 = characters[i]
            distance = similarity(char1, char2)
            if distance > threshold:
                unique_chars.add(char1)
                break
    gc.collect()
    return str(unique_chars)


print("Number of cpu : ", cpu_count() - 1)
pool = Pool(processes=7)
df["Characters_nltk"] = pool.map(sim, df["Characters_nltk"])
pool.close()
df.to_csv("df.csv", index=False)
# %%
df.Characters_nltk = df.Characters_nltk.apply(lambda x: eval(x))
df

# %% [markdown]
# ## 3.2 NER with BERT


# %%
def bertspeed(sentences):
    # Load the pre-trained model and the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    bert_ner = AutoModelForTokenClassification.from_pretrained(
        "Jean-Baptiste/roberta-large-ner-english"
    )

    # Create a NER pipeline consisting of the model and it's tokenizer
    ner_model = pipeline(
        "ner", model=bert_ner, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    bert_tags = []
    # Parse sentences one by one
    for sentence in sentences:
        result = ner_model(sentence)
        for entity in result:
            # Collect words and tags
            bert_tags.append((entity["word"], entity["entity_group"]))

    # Create list of names to add to the df
    bert_persons = []
    bert_places = []

    for i in range(len(bert_tags)):
        match bert_tags[i][1]:
            case "PER":
                bert_persons.append(bert_tags[i][0])
            case "LOC":
                bert_places.append(bert_tags[i][0])
    gc.collect()
    return str([str(bert_persons), str(bert_places)])


pool = Pool(processes=7)
df["tr"] = pool.map(bertspeed, df["text"])
pool.close()
df.to_csv("df.csv", index=False)

# %%
df.tr = df.tr.apply(lambda x: eval(x))
df["Characters_bert"] = df.tr.apply(lambda x: x[0])
df["Places_bert"] = df.tr.apply(lambda x: x[1])
df.to_csv("df.csv", index=False)


# %%
def bertspeed2(bert_persons):
    unique_bert = set()
    threshold = 0.65

    for i in tqdm(range(len(characters))):
        for j in range(len(bert_persons)):
            temp = []
            char1 = bert_persons[j]
            char2 = characters[i]
            distance = similarity(char1, char2)
            if distance > threshold:
                unique_bert.add(char1)
                break

    return str(unique_bert)


pool = Pool(processes=7)
df["Characters_bert"] = pool.map(bertspeed2, df["Characters_bert"])
df.Characters_bert = df.Characters_bert.apply(lambda x: eval(x))
pool.close()
df.to_csv("df.csv", index=False)

# %% RUN ALL ABOVE CELLS BEFORE THIS CELL
