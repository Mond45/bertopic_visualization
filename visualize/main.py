import numpy as np
import pandas as pd
import streamlit as st
from bertopic import BERTopic
from sklearn.preprocessing import normalize

st.set_page_config(layout="wide")


@st.cache_data
def load_stuff():
    topic_model = BERTopic(
        embedding_model=None,
        verbose=True,
        calculate_probabilities=True,
    ).load("mond45/panda")
    embeddings = np.load("embedding.npy")

    df = pd.read_json("cleaned.json")
    df_scraped = pd.read_json("scrap_data.json")
    df_scraped_dates = (
        df_scraped["date_delivered"]
        .str.extractall(
            r"(\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4})"
        )
        .groupby(level=0)
        .first()
    )
    df_scraped_dates = df_scraped_dates.iloc[:, [2, 1]]
    df_scraped_dates.columns = ["year", "month"]
    df_scraped_dates["month"] = df_scraped_dates["month"].map(
        {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }
    )
    df_scraped[["year", "month"]] = df_scraped_dates
    df_all = pd.concat(
        [
            df,
            df_scraped.drop(columns="date_delivered").rename(
                columns={"citation_title": "title"}
            ),
        ],
        ignore_index=True,
    )
    df_all["data"] = df_all["title"] + "\n" + df_all["abstract"]
    df_all["timestamp"] = pd.to_datetime(
        df_all["year"].astype(str) + df_all["month"].astype(str), format="%Y%m"
    )

    return topic_model, embeddings, df_all


topic_model, embeddings, df_all = load_stuff()


@st.cache_data
def get_topics_over_time():
    return topic_model.topics_over_time(
        df_all["data"], df_all["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )


st.write("# Paper Topics Visualization")

st.write("## Topic Info")
st.write(topic_model.get_topic_info().drop(columns=['Representative_Docs']).drop(index=[0]))

st.write("## Topic Clusters")
st.write(topic_model.visualize_topics())

st.write("## Topics Hierarchy")
st.write(topic_model.visualize_hierarchy(top_n_topics=50))

st.write("## Topic Over Time")
st.write(topic_model.visualize_topics_over_time(get_topics_over_time()))


def extract(row):
    filtered = [e["classification"] for e in row if e["@type"] == "SUBJABBR"][0]
    if not filtered:
        return pd.DataFrame()
    if isinstance(filtered, list):
        return [e["$"] for e in filtered]
    return [filtered]


st.write("## Topics per Class")
df_selected = df_all[df_all["classification"].notna()]
df_selected["classification"] = df_selected["classification"].apply(extract)
topics, _ = topic_model.transform(
    df_selected["data"], embeddings[df_selected.index.to_numpy()]
)


@st.cache_data
def get_topics_per_class() -> pd.DataFrame:

    documents = pd.DataFrame(
        {
            "Document": df_selected["data"],
            "Topic": topics,
            "Class": df_selected["classification"],
        }
    )
    global_c_tf_idf = normalize(topic_model.c_tf_idf_, axis=1, norm="l1", copy=False)
    # For each unique timestamp, create topic representations
    topics_per_class = []
    for _, class_ in enumerate(
        df_selected["classification"].explode().dropna().unique().tolist()
    ):

        # Calculate c-TF-IDF representation for a specific timestamp
        selection = documents[documents["Class"].map(lambda x: class_ in x)]
        documents_per_topic = selection.groupby(["Topic"], as_index=False).agg(
            {"Document": " ".join, "Class": "count"}
        )
        c_tf_idf, words = topic_model._c_tf_idf(documents_per_topic, fit=False)

        # Fine-tune the timestamp c-TF-IDF representation based on the global c-TF-IDF representation
        # by simply taking the average of the two
        c_tf_idf = normalize(c_tf_idf, axis=1, norm="l1", copy=False)
        c_tf_idf = (
            global_c_tf_idf[documents_per_topic.Topic.values + topic_model._outliers]
            + c_tf_idf
        ) / 2.0

        # Extract the words per topic
        words_per_topic = topic_model._extract_words_per_topic(
            words, selection, c_tf_idf, calculate_aspects=False
        )
        topic_frequency = pd.Series(
            documents_per_topic.Class.values, index=documents_per_topic.Topic
        ).to_dict()

        # Fill dataframe with results
        topics_at_class = [
            (
                topic,
                ", ".join([words[0] for words in values][:5]),
                topic_frequency[topic],
                class_,
            )
            for topic, values in words_per_topic.items()
        ]
        topics_per_class.extend(topics_at_class)

    topics_per_class = pd.DataFrame(
        topics_per_class, columns=["Topic", "Words", "Frequency", "Class"]
    )

    return topics_per_class


topics_per_class = get_topics_per_class()
st.write(
    topic_model.visualize_topics_per_class(
        topics_per_class, top_n_topics=25, height=600
    )
)

