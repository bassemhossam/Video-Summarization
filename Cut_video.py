import cv2
import os
import subprocess
from pydub import AudioSegment
import sys
import csv


def extract_audio(full_path):

    full_path_audio = full_path.replace(".mkv", ".wav")
    command = 'ffmpeg -y -i "' + full_path + '" -ac 1 -ar 8000 -acodec pcm_s16le "' + full_path_audio+'"'
    subprocess.call(command, shell=True)

    return full_path_audio


def split(full_path,id,start_frame,end_frame,start_time,end_time,full_path_audio):

    if os.path.isfile(full_path):

        cap = cv2.VideoCapture(full_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.VideoWriter_fourcc(*'XVID')))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_path_cut = full_path.replace(".mkv", id+".avi")
        full_path_cut = full_path_cut.replace('downloaded_videos_for_app/', 'downloaded_videos_for_app/scenes/')
        out = cv2.VideoWriter(full_path_cut, fourcc, fps, (w, h))


        frame_count = 0
        while frame_count < end_frame:
            ret, frame = cap.read()
            frame_count += 1

            if frame_count >= start_frame:
                out.write(frame)
        cap.release()
        out.release()
        t1 = start_time * 1000  # Works in milliseconds
        t2 = (end_time * 1000) + 1

        newAudio = AudioSegment.from_wav(full_path_audio)
        newAudio = newAudio[t1:t2]
        full_path_audio_cut=full_path_audio.replace(".wav", id+".wav")
        full_path_audio_cut = full_path_audio_cut.replace('downloaded_videos_for_app/', 'downloaded_videos_for_app/scenes/')
        newAudio.export(full_path_audio_cut, format="wav")
        # os.remove(full_path)
        # os.remove(full_path_audio)


def main():
    video_name = sys.argv[1]
    csv_path = video_name+"-Scenes.csv"
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        next(csv_reader)
        full_path = video_name+'.mkv'
        full_path_audio = extract_audio(full_path)
        for row in csv_reader:
            split(full_path, row[0], int(row[1]), int(row[4]), float(row[3]), float(row[6]),full_path_audio)


if __name__ == "__main__":
    main()