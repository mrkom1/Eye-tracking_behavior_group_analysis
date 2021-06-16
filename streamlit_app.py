import base64
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from fast_histogram import histogram2d
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go

ROOT_DATA_FOLDER = Path('dataset_journalists')


DIR_RENAME_DICT = {
    "Нейтральні тексти": "neutral",
    "Позитивні тексти": "positive",
    "Негативні тексти": "negative",
}

CHAR_TYPE_RENAME_DICT = {
    "Рівномірний розподіл": "uniform",
    "Усереднений за групою": "main_trend",
}


def load_sess_results():
    sess_results = {}
    with open(ROOT_DATA_FOLDER / 'sessions_results.pickle', 'rb') as f:
        sess_results = pickle.load(f)

    return sess_results


def jensen_shannon_divergance(p, q):
    """
    method to compute the Jenson-Shannon Divergance
    between two probability distributions
    """
    assert p.shape == q.shape

    p = p.flatten()
    q = q.flatten()

    # normalization
    p = p / np.sum(p)
    q = q / np.sum(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (entropy(p, m) + entropy(q, m)) / 2

    # # compute the Jensen Shannon Distance
    # distance = np.sqrt(divergence)

    return divergence


def find_main_trend(sess_results, text_type, hist_shape):
    main_trend_hist_dict = {}
    for p, sess_res in sess_results.items():
        type_ = p.split("/")[2]

        if type_ == text_type:
            for page, page_data in sess_res["heatmap_points"].items():
                h = histogram2d(x=page_data['X'],
                                y=page_data['Y'],
                                bins=hist_shape,
                                range=[[0, 1], [0, 1]])
                if main_trend_hist_dict.get(page, np.array([])).any():
                    main_trend_hist_dict[page] += h
                else:
                    main_trend_hist_dict[page] = h
    return main_trend_hist_dict


def find_similarity(df_u_dict: dict,
                    sess_results: dict,
                    text_type: str = "Нейтральні тексти",
                    similarity_type: str = "uniform",
                    metric_type: str = "Jensen-Shennon",
                    hist_shape: tuple = (50, 50)):

    if similarity_type == "uniform":
        h1 = np.full(fill_value=(1 / (hist_shape[0]*hist_shape[1])),
                    shape=hist_shape)
    elif similarity_type == "main_trend":
        main_trend_hist_dict = find_main_trend(
            sess_results, text_type, hist_shape)

    for p, sess_res in sess_results.items():
        type_ = p.split("/")[2]
        name_ = p.split("/")[-2]

        if type_ == text_type:
            jensen_div = []
            for page, page_data in sess_res["heatmap_points"].items():
                h0 = histogram2d(x=page_data['X'],
                                 y=page_data['Y'],
                                 bins=hist_shape,
                                 range=[[0, 1], [0, 1]])
                if similarity_type == "main_trend":
                    h1 = main_trend_hist_dict[page]

                if metric_type == "DTW":
                    jensen_div.append(
                        fastdtw(h0, h1, dist=euclidean)[0]
                    )
                else:
                    jensen_div.append(
                        jensen_shannon_divergance(h0, h1)
                    )

            if jensen_div:
                if metric_type == "DTW":
                    jensen_div_mean = 1/np.mean(jensen_div)
                else:
                    jensen_div_mean = np.median(jensen_div)
                df_u_dict[text_type].loc[
                    name_, "point_distance"] = jensen_div_mean
    if similarity_type == "DTW":
        distances = df_u_dict[text_type].loc[
            :, "point_distance"].values
        if (np.max(distances) - np.min(distances)) != 0:
            df_u_dict[text_type].loc[
                :, "point_distance"] = (
                    (distances-np.min(distances))
                    / (np.max(distances) - np.min(distances))
                    )
    df_u_dict[text_type] = df_u_dict[text_type].astype(float)
    return df_u_dict[text_type]


def find_group_depndencies(sess_results, similarity_type, metric_type):
    neutral_idxs = [k.split("/")[-2] for k in sess_results.keys()
                    if k.split("/")[2] == "Нейтральні тексти"]
    negative_idxs = [k.split("/")[-2] for k in sess_results.keys()
                     if k.split("/")[2] == "Негативні тексти"]
    positive_idxs = [k.split("/")[-2] for k in sess_results.keys()
                     if k.split("/")[2] == "Позитивні тексти"]

    neutral_df = pd.DataFrame(index=neutral_idxs,
                              columns=["point_distance"])
    negative_df = pd.DataFrame(index=negative_idxs,
                               columns=["point_distance"])
    positive_df = pd.DataFrame(index=positive_idxs,
                               columns=["point_distance"])

    similarity_dict = {
        "Нейтральні тексти": neutral_df,
        "Негативні тексти": negative_df,
        "Позитивні тексти": positive_df
    }

    for ttype in ("Нейтральні тексти", "Негативні тексти", "Позитивні тексти"):
        find_similarity(similarity_dict, sess_results,
                        ttype, similarity_type, metric_type)

    return similarity_dict


def plot_similarity_hist(similarity_dict):
    group_labels = ["Нейтральні тексти",
                    "Негативні тексти", "Позитивні тексти"]
    colors = ['#4F57BB', '#A84C3A', '#499B79']
    hist_data = [similarity_dict[group_labels[0]].dropna(
    ).point_distance.values,
        similarity_dict[group_labels[1]].dropna(
    ).point_distance.values,
        similarity_dict[group_labels[2]].dropna(
    ).point_distance.values, ]

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels,
                             colors=colors, bin_size=.02,
                             )
    # Add title
    fig.update_layout(title_text=('Similarity:'),
                    #   xaxis_range=[0.0, 1.0],
                      bargap=0.2,  # gap between bars of adjacent location coordinates
                      bargroupgap=0.1,  # gap between bars of the same location coordinates
                      width=1200, height=800
                      )

    columns = st.beta_columns([1, 12, 1])
    columns[1].plotly_chart(fig)


def create_pdf_markdown(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="410" height="500" type="application/pdf"></iframe>'
    return pdf_display


def reading_speed_barplot(rs_dict: dict):
    fig = go.Figure(data=[
        go.Bar(name='All gazes',
               y=[*rs_dict["all"]],
               x=[*rs_dict["all"].values()],
               text=[*rs_dict["all"].values()],
               orientation='h'),
        go.Bar(name='Reading gazes',
               y=[*rs_dict["reading"]],
               x=[*rs_dict["reading"].values()],
               text=[*rs_dict["reading"].values()],
               orientation='h'),
    ])

    fig.update_layout(
        xaxis_title="reading speed",
        yaxis_title="page",
        width=500,
        height=400
    )
    fig['layout']['yaxis']['autorange'] = "reversed"
    return fig


def plot_user_markdown(column, q, sess_results, key):
    option = column.selectbox('Choose user:', q.index.tolist())
    rs_dict = (sess_results["dataset_journalists/eye_tracking_recordings"
                            f"/{key}/{option}/1"]["reading_speed"])
    column.plotly_chart(reading_speed_barplot(rs_dict))
    blinks_show = column.checkbox("show blinks", key=str(column))
    if blinks_show:
        file_name = (ROOT_DATA_FOLDER / "blinks_heatmaps_filtered" / "personal"
                     / DIR_RENAME_DICT[key] / f"{option}.pdf")
    else:
        file_name = (ROOT_DATA_FOLDER / "reading_heatmaps_filtered" / "personal"
                    / DIR_RENAME_DICT[key] / f"{option}.pdf")
    pdf_markdown = create_pdf_markdown(file_name)
    column.markdown(pdf_markdown, unsafe_allow_html=True)


def similarity_clusters_visualization(similarity_dict: dict,
                                      sess_results: dict):
    key = st.selectbox('Select group type:', [*similarity_dict.keys()])
    df = similarity_dict[key]

    quantiles = df.quantile([0.25, 0.5, 0.75])

    columns = st.beta_columns(3)

    q1 = df[(df <= quantiles.loc[0.25,
                                    "point_distance"]).values]
    q12 = df[(df >= quantiles.loc[0.25,
                                    "point_distance"]).values
                & (df <= quantiles.loc[0.75,
                                    "point_distance"]).values]
    q2 = df[(df >= quantiles.loc[0.75,
                                    "point_distance"]).values]

    columns[0].subheader("Highest similarity (25%)")
    plot_user_markdown(columns[0], q1, sess_results, key)

    columns[1].subheader("Medium similarity (50%)")
    plot_user_markdown(columns[1], q12, sess_results, key)

    columns[2].subheader("Lowest similarity (25%)")
    plot_user_markdown(columns[2], q2, sess_results, key)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    sess_results = load_sess_results()
    similarity_type = CHAR_TYPE_RENAME_DICT[
        st.selectbox('Choose characteristic of interest:',
                     [*CHAR_TYPE_RENAME_DICT.keys()])
    ]

    metric_type = st.selectbox('Choose metric:', ["Jensen-Shennon", "DTW"])

    similarity_dict = find_group_depndencies(sess_results, 
                                             similarity_type,
                                             metric_type)

    plot_similarity_hist(similarity_dict)
    similarity_clusters_visualization(similarity_dict, sess_results)
