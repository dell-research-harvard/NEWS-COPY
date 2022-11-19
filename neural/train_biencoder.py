import pandas as pd
from datetime import datetime

from sentence_transformers import losses

from modified_sbert.train import train_biencoder


def extract_raw_data(dataset_path):

    raw_data = pd.read_csv(dataset_path, sep='\t', encoding='utf-8')

    sentence_1_list = [str(i) for i in list(raw_data["Text 1"])]
    sentence_2_list = [str(i) for i in list(raw_data["Text 2"])]
    labels = list(raw_data["Label"])

    return {'sentence_1': sentence_1_list, 'sentence_2': sentence_2_list, "labels": labels}


if __name__ == '__main__':

    data_path = ''

    train_biencoder(
        train_data=extract_raw_data(f'{data_path}/train_set.csv'),
        dev_data=extract_raw_data(f'{data_path}/dev_set.csv'),
        base_model='sentence-transformers/all-mpnet-base-v2',
        add_pooling_layer=False,
        train_batch_size=32,
        num_epochs=16,
        warmup_epochs=16,
        loss_fn='contrastive',
        loss_params={'distance_metric': losses.SiameseDistanceMetric.COSINE_DISTANCE, 'margin': 0.2},
        model_save_path=f'output/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
    )
