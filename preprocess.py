import sys
import base64
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from fast_histogram import histogram2d
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from gazestatistics.session import PdfSession

from gazestatistics.reading import ContentBasedReadingModel

from gazestatistics.tracker.utils import group_points_by_block_id

from gazestatistics.constants.tracker import GAZE_REL_BLOCK_X_DF_COLUMN
from gazestatistics.constants.tracker import GAZE_REL_BLOCK_Y_DF_COLUMN
from gazestatistics.constants.tracker import BLOCK_ID_DF_COLUMN

from contents import PdfData


sys.path.append('..')

ROOT_DATA_FOLDER = Path('dataset_journalists')
READING_TR = 0.5
READING_MODEL_PATH = ("/Users/alexandrkravchenko/Documents/work/"
                      "eyepass-survey/server/bin/inception_model_graph")


def get_sess_dict(meta_df: pd.DataFrame, reload: bool = False):
    sess_dict = {}

    if reload:
        for _, sess_row in tqdm(meta_df.iterrows(), total=meta_df.shape[0]):
            folder = sess_row.folder
            sess_dict[folder] = PdfSession.from_folder(folder)

        with open(ROOT_DATA_FOLDER / 'sessions_dict.pickle', 'wb') as f:
            pickle.dump(sess_dict, f)

    else:
        with open(ROOT_DATA_FOLDER / 'sessions_dict.pickle', 'rb') as f:
            sess_dict = pickle.load(f)

    return sess_dict


def process_reading_gazes(tracker: pd.DataFrame, pdf_data: PdfData):
    reading_model_mask = (tracker.reading_probs.values
                          > READING_TR)

    tracker["reading_mask"] = reading_model_mask

    for page, page_data in tracker.groupby(BLOCK_ID_DF_COLUMN):

        not_image_gazes_mask = np.full(len(page_data), True)
        in_text_mask = np.full(len(page_data), False)

        for image in pdf_data.image_coordinates[page]:
            x_mask = ((page_data[GAZE_REL_BLOCK_X_DF_COLUMN].values
                       < image["x1"])
                      & (page_data[GAZE_REL_BLOCK_X_DF_COLUMN].values
                         > image["x0"]))
            y_mask = ((page_data[GAZE_REL_BLOCK_Y_DF_COLUMN].values
                       < image["y1"])
                      & (page_data[GAZE_REL_BLOCK_Y_DF_COLUMN].values
                         > image["y0"]))
            not_image_gazes_mask &= ~(x_mask & y_mask)

        for text_obj in pdf_data.text_objects_coordinates[page]:
            x_mask = ((page_data[GAZE_REL_BLOCK_X_DF_COLUMN].values
                       < text_obj["x1"])
                      & (page_data[GAZE_REL_BLOCK_X_DF_COLUMN].values
                         > text_obj["x0"]))
            y_mask = ((page_data[GAZE_REL_BLOCK_Y_DF_COLUMN].values
                       < text_obj["y1"])
                      & (page_data[GAZE_REL_BLOCK_Y_DF_COLUMN].values
                         > text_obj["y0"]))
            in_text_mask |= (x_mask & y_mask)

        tracker.loc[page_data.index,
                    "reading_mask"] &= (not_image_gazes_mask
                                        & in_text_mask)

    return tracker


def get_sess_results(sess_dict: dict, reload: bool = False):
    if reload:
        reading_model = ContentBasedReadingModel(READING_MODEL_PATH)
        sess_results = {}

        pbar = tqdm(sess_dict.items())
        for p, sess in pbar:
            pbar.set_description(f"Processing {p}")

            # process pdf
            pdf_path = ROOT_DATA_FOLDER / (p.split("/")[2] + ".pdf")
            pdf_data = PdfData(pdf_path)
            pdf_data.process()

            res = sess.process(scrolling_map=False, heatmap_points=False)
            res['tracker']["reading_probs"] = reading_model.predict(
                res['tracker'],
                content_changes_df=res['content']
                )
            tracker = process_reading_gazes(res["tracker"], pdf_data)
            res["heatmap_points"] = group_points_by_block_id(
                tracker.loc[tracker.reading_mask])
            sess_results[p] = res
        with open(ROOT_DATA_FOLDER / 'sessions_results.pickle', 'wb') as f:
            pickle.dump(sess_results, f)

    else:
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


def find_similarity(df_u_dict: dict,
                    sess_results: dict,
                    text_type: str = "Нейтральні тексти",
                    hist_shape: tuple = (50, 50)):

    h1 = np.full(fill_value=(1 / (hist_shape[0]*hist_shape[1])),
                 shape=hist_shape)

    for p, sess_res in sess_results.items():
        type_ = p.split("/")[2]
        name_ = p.split("/")[-2]

        if type_ == text_type:
            jensen_div = []
            for _, page_data in sess_res["heatmap_points"].items():
                h0 = histogram2d(x=page_data['X'],
                                 y=page_data['Y'],
                                 bins=hist_shape,
                                 range=[[0, 1], [0, 1]])

                jensen_div.append(
                    jensen_shannon_divergance(h0, h1)
                )
            if jensen_div:
                jensen_div_mean = np.median(jensen_div)
                df_u_dict[text_type].loc[
                    name_, "jensen_shennon_divergance"] = jensen_div_mean
    df_u_dict[text_type] = df_u_dict[text_type].astype(float)
    return df_u_dict[text_type]


def find_group_depndencies(sess_results):
    neutral_idxs = [k.split("/")[-2] for k in sess_results.keys()
                    if k.split("/")[2] == "Нейтральні тексти"]
    negative_idxs = [k.split("/")[-2] for k in sess_results.keys()
                    if k.split("/")[2] == "Негативні тексти"]
    positive_idxs = [k.split("/")[-2] for k in sess_results.keys()
                    if k.split("/")[2] == "Позитивні тексти"]
    
    neutral_df = pd.DataFrame(index=neutral_idxs, 
                              columns=["jensen_shennon_divergance"])
    negative_df = pd.DataFrame(index=negative_idxs,
                               columns=["jensen_shennon_divergance"])
    positive_df = pd.DataFrame(index=positive_idxs,
                               columns=["jensen_shennon_divergance"])

    similarity_dict = {
        "Нейтральні тексти": neutral_df,
        "Негативні тексти": negative_df,
        "Позитивні тексти": positive_df
    }

    for ttype in ("Нейтральні тексти", "Негативні тексти", "Позитивні тексти"):
        find_similarity(similarity_dict, sess_results, ttype)
    
    return similarity_dict


def plot_similarity_hist(similarity_dict):
    group_labels = ["Нейтральні тексти",
                    "Негативні тексти", "Позитивні тексти"]
    colors = ['#4F57BB', '#A84C3A', '#499B79']
    hist_data = [similarity_dict[group_labels[0]].dropna(
            ).jensen_shennon_divergance.values,
                 similarity_dict[group_labels[1]].dropna(
            ).jensen_shennon_divergance.values,
                 similarity_dict[group_labels[2]].dropna(
            ).jensen_shennon_divergance.values, ]

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels,
                             colors=colors, bin_size=.02,
                             )
    # Add title
    fig.update_layout(title_text=('Jensen Shennon Divergance similarity with '
                                  'uniform distribution'),
                      xaxis_range=[0.3, 0.7],
                      bargap=0.2,  # gap between bars of adjacent location coordinates
                      bargroupgap=0.1,  # gap between bars of the same location coordinates
                      width=900, height=700
                      )

    columns = st.beta_columns([1, 3, 1])
    columns[1].plotly_chart(fig)


def create_pdf_markdown(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="410" height="500" type="application/pdf"></iframe>'
    return pdf_display


def similarity_clusters_visualization(similarity_dict: dict):
    for key, df in similarity_dict.items():

        quantiles = df.quantile([0.25, 0.5, 0.75])

        st.header(key)
        columns = st.beta_columns(3)

        q1 = df[(df <= quantiles.loc[0.25,
                                     "jensen_shennon_divergance"]).values]
        q12 = df[(df >= quantiles.loc[0.25,
                                      "jensen_shennon_divergance"]).values
                 & (df <= quantiles.loc[0.75,
                                        "jensen_shennon_divergance"]).values]
        q2 = df[(df >= quantiles.loc[0.75,
                                     "jensen_shennon_divergance"]).values]

        columns[0].subheader("High level (< Q1)")
        option0 = columns[0].selectbox('choose user:', q1.index.tolist())
        pdf_markdown = create_pdf_markdown(
            "dataset_journalists/reading_heatmaps_filtered/personal"
            f"/{key}/{option0}.pdf")
        columns[0].markdown(pdf_markdown, unsafe_allow_html=True)

        columns[1].subheader("Medium level  (Q1 < X < Q2)")
        option1 = columns[1].selectbox('choose user:', q12.index.tolist())
        pdf_markdown = create_pdf_markdown(
            "dataset_journalists/reading_heatmaps_filtered/personal"
            f"/{key}/{option1}.pdf")
        columns[1].markdown(pdf_markdown, unsafe_allow_html=True)

        columns[2].subheader("Low level  (> Q2)")
        option2 = columns[2].selectbox('choose user:', q2.index.tolist())
        pdf_markdown = create_pdf_markdown(
            "dataset_journalists/reading_heatmaps_filtered/personal"
            f"/{key}/{option2}.pdf")
        columns[2].markdown(pdf_markdown, unsafe_allow_html=True)


if __name__ == '__main__':
    meta_df = pd.read_csv('meta.csv', index_col=0)
    sess_dict = get_sess_dict(meta_df, reload=False)
    sess_results = get_sess_results(sess_dict, reload=False)

    similarity_dict = find_group_depndencies(sess_results)

    # streamlit visualization
    st.set_page_config(layout="wide")
    plot_similarity_hist(similarity_dict)
    similarity_clusters_visualization(similarity_dict)
