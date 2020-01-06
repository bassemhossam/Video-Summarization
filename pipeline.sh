#!/bin/sh
cd $code
video_name=$1
features_path="/lfs01/workdirs/shams010u1/test/features"
video_features_path="$features_path/$video_name-vid"
audio_features_path="$features_path/$video_name-aud"
download_path="/lfs01/workdirs/shams010u1/downloaded_videos_for_app/scenes/"
python3 ResNet.py --video-path "$download_path/$video_name.avi" --video-features-path "$video_features_path" & \
cd audioset && \
python3 ./vggish_extract_features.py --wav_file "$download_path/$video_name.wav" --video_path "$audio_features_path" &\
wait
cd ..
python3 ./Get_Summary.py --video-features-path "$video_features_path.npy" --audio-features-path "$audio_features_path.npy"
