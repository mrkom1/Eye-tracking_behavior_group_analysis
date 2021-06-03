import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

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
        old_pdf_path = ""
        for p, sess in pbar:
            pbar.set_description(f"Processing {p}")

            # process pdf
            pdf_path = ROOT_DATA_FOLDER / (p.split("/")[2] + ".pdf")
            if pdf_path != old_pdf_path:
                pdf_data = PdfData(pdf_path)
                pdf_data.process()
                old_pdf_path = pdf_path

            res = sess.process(reading_model=reading_model,
                               scrolling_map=False, 
                               heatmap_points=False,
                               aggregate_pages_stats=True,)

            res['tracker']["reading_probs"] = reading_model.predict(
                res['tracker'],
                content_changes_df=res['content']
                )
            tracker = process_reading_gazes(res["tracker"], pdf_data)
            # heatmap
            res["heatmap_points"] = group_points_by_block_id(
                tracker.loc[tracker.reading_mask])
            # reading speed
            reading_speed_dict = {}
            reading_speed_dict["all"] = {}
            reading_speed_dict["reading"] = {}
            for page_id, page_time in res["pages_stats"].items():
                reading_speed_dict["all"][page_id] = round(
                    np.nan_to_num(
                        pdf_data.words_count[page_id]
                        / (page_time['gaze'] / 60_000),
                        neginf=0),
                    1)
                reading_speed_dict["reading"][page_id] = round(
                    np.nan_to_num(
                        pdf_data.words_count[page_id]
                        / (page_time['reading'] / 60_000),
                        neginf=0),
                    1)
            res["reading_speed"] = reading_speed_dict

            sess_results[p] = res
        with open(ROOT_DATA_FOLDER / 'sessions_results.pickle', 'wb') as f:
            pickle.dump(sess_results, f)

    else:
        with open(ROOT_DATA_FOLDER / 'sessions_results.pickle', 'rb') as f:
            sess_results = pickle.load(f)

    return sess_results


if __name__ == '__main__':
    meta_df = pd.read_csv('meta.csv', index_col=0)
    sess_dict = get_sess_dict(meta_df, reload=False)
    sess_results = get_sess_results(sess_dict, reload=False)
