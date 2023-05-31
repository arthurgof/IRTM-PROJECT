from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

import pandas as pd

from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import gc

gc.collect()

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")


true_relation_original = [
    ("Cersei Lannister", "Jaime Lannister"),
    ("Daenerys Targaryen", "Khal Drogo"),
    ("Jon Snow", "Ygritte"),
    ("Tyrion Lannister", "Shae"),
    ("Robb Stark", "Jeyne Westerling"),
    ("Sansa Stark", "Joffrey Baratheon"),
    ("Petyr Baelish"),
    ("Drogo", "Daenerys's handmaidens"),
    ("Renly Baratheon", "Loras Tyrell"),
    ("Viserys Targaryen"),
    ("Illyrio Mopatis", "Daenerys Targaryen"),
    ("Brienne of Tarth", "Renly Baratheon"),
]


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


dicoPerso = {
    "dany": "Daenerys Targaryen",
    "daenerys": "Daenerys Targaryen",
    "jon snow": "Jon Snow",
    "jon": "Jon Snow",
    "tyrion": "Tyrion Lannister",
    "bran": "Bran Stark",
    "khal drogo": "Drogo",
    "ned": "Eddard Stark",
    "viserys": "Viserys Targaryen",
    "loras": "Loras Tyrell",
    "ser jorah": "Jorah Mormont",
    "sam": "Samwell Tarly",
}


def evaluate_bertopic(
    models_name, train_set, characters, text_aggregation, threshold=0.6
):
    representation_model = KeyBERTInspired()
    vectorizer_model = CountVectorizer(stop_words="english")
    model_roberta = SentenceTransformer(models_name)

    model_roberta.encode(train_set)

    topic_model = BERTopic(
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        embedding_model=model_roberta,
        verbose=True,
        calculate_probabilities=True,
    )

    topics = topic_model.fit_transform(train_set)

    t, s = topic_model.find_topics(["fuck", "kiss"], top_n=10)

    topics, probs = topic_model.transform(text_aggregation)

    if -1 in t:
        t.remove(-1)

    rmt = []

    for i in range(len(t)):
        if s[i] < 0.52:
            rmt.append(t[i])

    for r in rmt:
        t.remove(r)

    if len(t) == 0:
        t.append(rmt[0])

    df = pd.DataFrame(columns=["text", "cluster", "similarity", "place"])
    for i in range(len(topics)):
        if topics[i] in t:
            if probs[i][topics[i]] > 0.999:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [[text_aggregation[i], topics[i], probs[i][topics[i]], i]],
                            columns=["text", "cluster", "similarity", "place"],
                        ),
                    ],
                    ignore_index=True,
                )

    df["Characters_nltk"] = ""
    df["Places_nltk"] = ""

    for index in range(len(df)):
        pers, loc = nltk_ner(df.loc[index, "text"])
        df.loc[index, "Characters_nltk"] = str(pers)

    df.Characters_nltk = df.Characters_nltk.apply(lambda x: eval(x))

    for index in tqdm(range(len(df))):
        persons = df.loc[index, "Characters_nltk"]
        unique_chars = set()
        for i in range(len(persons)):
            char1 = persons[i]
            if char1.lower() in dicoPerso:
                unique_chars.add(dicoPerso[char1.lower()])
                continue
            if char1.lower().replace(" ", "") in dicoPerso:
                unique_chars.add(dicoPerso[char1.lower().replace(" ", "")])
                continue
            elif char1 in characters:
                unique_chars.add(char1)
                dicoPerso[char1.lower()] = char1
                continue
            v = []
            for j in range(len(characters)):
                char2 = characters[j]
                distance = similarity(char1, char2)
                v.append(distance)
            unique_chars.add(characters[v.index(max(v))])
            dicoPerso[char1.lower()] = characters[v.index(max(v))]
        df.loc[index, "Characters_nltk"] = str(unique_chars)

    df.Characters_nltk = df.Characters_nltk.apply(lambda x: list(eval(x)))

    precision = 0
    recall = 0
    f1 = 0
    false = 0
    true_relation = list(true_relation_original)
    found = [False for i in range(len(true_relation))]
    for index in range(0, len(df)):
        characters_nltk = df.loc[index, "Characters_nltk"]
        for i in range(len(true_relation)):
            if found[i]:
                continue
            if len(true_relation[i]) == 2:
                p1, p2 = true_relation[i]
                for char1 in characters_nltk:
                    for char2 in characters_nltk:
                        if char1 == char2:
                            continue
                        if similarity(char1, p1) > threshold:
                            if similarity(char2, p2) > threshold:
                                found[i] = True
                                break
                    if found[i]:
                        break
                if not found[i]:
                    false += 1
            else:
                for char in characters_nltk:
                    if similarity(char, true_relation[i]) > threshold:
                        found[i] = True
                        break
                if not found[i]:
                    false += 1

    if recall + false == 0:
        recall = 0
    else:
        recall = sum(found) / (sum(found) + false)
    precision = sum(found) / len(true_relation_original)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print(
        "models",
        models_name,
        "Precision: ",
        precision,
        "Recall: ",
        recall,
        "F1: ",
        f1,
        "Len: ",
        len(df),
    )

    return precision, recall, f1
