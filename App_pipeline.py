#!python
from __future__ import unicode_literals
import youtube_dl
import sys
from io import StringIO
import cv2
import os
import subprocess
from pydub import AudioSegment
import csv
import multiprocessing
import time
import tensorflow as tf
import spacy 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if type(tf.contrib) != type(tf): tf.contrib._warning = None


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
full_path=''
full_path_audio=''
vid_title=''


# split a list into evenly sized chunks
def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def do_job(data_slice, full_path, full_path_audio, vid_title):
    for item in data_slice:
        main_pipeline(full_path, item, full_path_audio, vid_title)


def dispatch_jobs(data, full_path, full_path_audio, vid_title):
    job_number = 4
    total = len(data)
    if total < job_number:
        chunk_size = 1
    else:
        chunk_size = total // job_number
    slice = chunks(data, chunk_size)
    # print('chunk_size:', chunk_size)
    # print('slice:', slice)
    jobs = []

    for i, s in enumerate(slice):
        j = multiprocessing.Process(target=do_job, args=(s, full_path, full_path_audio, vid_title))
        jobs.append(j)
    for j in jobs:
        j.start()


def time_stamp(secs):
    return time.strftime('%H:%M:%S', time.gmtime(secs))


def main_pipeline(row):
    global full_path, full_path_audio, vid_title
    # print(full_path, full_path_audio, vid_title)
    if len(row) > 0:
        if float(row[9]) < 1:
            return
        split(full_path, full_path_audio, row[0], int(row[1]), int(row[4]), float(row[3]), float(row[6]))
        video_title = vid_title + row[0]

    else:
        start,end = split(full_path,full_path_audio)
        video_title = vid_title
    command = '$code/pipeline.sh "' + video_title+'"'
    # subprocess.call(command, shell=True)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate() 
    p.wait()
    output=output.decode("utf-8")
    start=output.find('<START>')
    end=output.find('<END')
    if start != -1 and end != -1:
        return output[start+7:end]
    else:
        return
        
     
    


def download_video(url, save_path, stdout):
    ydl_opts = {
        'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
        'outtmpl': str(save_path + '.%(ext)s'),
        'continuedl': 'True',
        'merge_output_format': 'mkv',
        'subtitlesformat': 'srt'
    }

    ydl_optsw = {
        'listsubtitles': 'True'
    }
    with youtube_dl.YoutubeDL(ydl_optsw) as ydl:
        ydl.download([url])

    sublist = stdout.getvalue()
    availsub = sublist.find("Available subtitles")
    ensub = sublist.find("en", availsub)

    if ensub != -1:
        ydl_opts['writesubtitles'] = 'True'
    else:
        availsub = sublist.find("Available automatic")
        ensub = sublist.find("en", availsub)
        if ensub != -1:
            ydl_opts['writeautomaticsub'] = 'True'

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def extract_audio(full_path):
    full_path_audio = full_path.replace(".mkv", ".wav")
    command = 'ffmpeg -loglevel panic -y -i "' + full_path + '" -ac 1 -ar 8000 -acodec pcm_s16le "' + full_path_audio + '"'
    subprocess.call(command, shell=True)

    return full_path_audio


def split(full_path,full_path_audio, id="", start_frame=None, end_frame=None, start_time=None, end_time=None):
    if os.path.isfile(full_path):

        cap = cv2.VideoCapture(full_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.VideoWriter_fourcc(*'XVID')))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_path_cut = full_path.replace(".mkv", id + ".avi")
        full_path_cut = full_path_cut.replace('downloaded_videos_for_app/', 'downloaded_videos_for_app/scenes/')
        out = cv2.VideoWriter(full_path_cut, fourcc, fps, (w, h))
        if start_time is None:
            start_frame = 0
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_time = 0
            end_time = end_frame/fps

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
        full_path_audio_cut = full_path_audio.replace(".wav", id + ".wav")
        full_path_audio_cut = full_path_audio_cut.replace('downloaded_videos_for_app/',
                                                          'downloaded_videos_for_app/scenes/')
        newAudio.export(full_path_audio_cut, format="wav")
        # os.remove(full_path)
        # os.remove(full_path_audio)
        return start_time,end_time


def summary(sentences,threshold):

    nlp = spacy.load('en_core_web_lg')
    doc_sentences = []
    del_list = []
    for i in range(len(sentences)):
        doc_sentences.append(nlp(sentences[i]))

    for j in range(len(doc_sentences)):
        for k in range(j+1, len(doc_sentences)):
            score=doc_sentences[j].similarity(doc_sentences[k])
            # print('score between sentences', j, ',', k, 'is', score)
            if score > threshold:
                if k not in del_list:
                    del_list.append(k)
    del_list.sort(reverse=True)

    if del_list:
        for x in del_list:
            del(sentences[x])
    return sentences

def format_summary(summary_list):
    out=""
    for sentence in summary_list:
        out+=sentence.capitalize()+"."
    return out

def main():
    global full_path, full_path_audio, vid_title
    url = sys.argv[1]
    stdout = StringIO()
    sterr = StringIO()
    og_stdout = sys.stdout
    og_sterr = sys.stderr
    sys.stdout = stdout
    sys.stderr = sterr

    with youtube_dl.YoutubeDL({}) as ydl:
        meta = ydl.extract_info(url, download=False)

    save_path = "/lfs01/workdirs/shams010u1/downloaded_videos_for_app/" + meta['title']
    download_video(url, save_path, stdout)
    sys.stdout = og_stdout
    sys.stderr = og_sterr
    command = 'scenedetect -q -i ' + '"' + save_path + '.mkv" -o ' + "/lfs01/workdirs/shams010u1/downloaded_videos_for_app/" + ' detect-content list-scenes'
    subprocess.call(command, shell=True)
    csv_path = save_path + "-Scenes.csv"
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        next(csv_reader)
        data = []
        for row in csv_reader:
            data.append(row)
    full_path = save_path + '.mkv'
    full_path_audio = extract_audio(full_path)
    vid_title = meta['title']
    # print('data:', data)
    # dispatch_jobs(data, full_path, full_path_audio, meta['title'])
    # print(stdout)
    captions_list=[]
    if len(data) > 0:
        for x in data:
            result=main_pipeline(x)
            if result is not None:
                captions_list.append(result)
    else:
        result=main_pipeline([])
        if result is not None:
            captions_list.append(result)
    print('<START>')
    captions_summary_list=summary(captions_list,0.9)
    captions_summary_string=format_summary(captions_summary_list)
    print(captions_summary_string)
    print('<EndOfCaptions>')

    # pool = multiprocessing.Pool(processes=4)
    # print(pool.map(main_pipeline, data, chunksize=13))


if __name__ == "__main__":
    main()
