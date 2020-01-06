#!/usr/bin/env python3
from __future__ import unicode_literals
import pandas as pd
import json
import os
import numpy as np


def process_video(row):
    video_id = row['video_id']
    start = float(row['start time'])
    end = float(row['end time'])
    vid_features_path = 'video_features/train/'+video_id+'.npy'
    aud_features_path = 'audio_features/train/'+video_id+'.npy'
    if os.path.isfile(vid_features_path) and os.path.isfile(aud_features_path):
        video_feature = np.load(vid_features_path)
        audio_feature = np.load(aud_features_path)
        if video_feature.shape[0]  < audio_feature.shape[0]*2.5:
            print(video_id)
            print('vid_feature_shape:', video_feature.shape[0])
            print('aud_feature_shape:', audio_feature.shape[0])
            print('video_time:',end-start)
            print('\n')


def main():


    video_data_path = "videodatainfo_2017.json"

    with open(video_data_path) as f:
        data = json.load(f)

    video_data = pd.DataFrame.from_dict(data["videos"])
    video_data['video_path'] = video_data.apply(
        lambda row: row['split'] + '/' + row['video_id'] + '_' + str(row['start time']) + '_' + str(
            row['end time']) + '.%(ext)s', axis=1)

    video_data.apply(lambda row: process_video( row), axis=1)


# download_and_process_video(video_save_path,video_data.iloc[[1]])
if __name__ == "__main__":
    main()
