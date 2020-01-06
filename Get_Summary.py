#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from model import *
from utils import *
from data_iterator import *
from tensorboardX import SummaryWriter
import argparse
import time
import datetime
from evaluate import calculate_metrics
from baseline_model import *
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=1,
                    help="batch size for training (default: 20)")
parser.add_argument("--lr", type=float, default=1,
                    help="learning rate for training (default: 1e-3)")
parser.add_argument("--model", required=False,
                    help="model name to save (required)")
parser.add_argument("--epochs", type=int, default=5000,
                    help="number of epochs to be trained (default: 20)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="log interval (default: 10)")
parser.add_argument("--ss", type=float, default=None,
                    help="starting epsilon for scheduled sampling (default: None)")
parser.add_argument("--max-grad-norm", type=float, default=10,
                    help="maximum gradient clipping norm")
parser.add_argument("--captions-per-vid", type=int, default=20,
                    help="number of captions per video for training")
parser.add_argument("--optimizer", default="adadelta",
                    help="type of optimizer to use for training")
parser.add_argument("--max-videos", default=-1, type=int,
                    help="maximum number of videos in the training set (default: -1 -> use all the videos)")
parser.add_argument("--model-type", default="haca",
                    help="type of model to use")
parser.add_argument("--beam-search", action="store_true",
                    help="whether to use beam search while testing")
parser.add_argument("--max-caption-length", type=int, default=15)
parser.add_argument("--audio-features-path", type=str,required=True)
parser.add_argument("--video-features-path", type=str,required=True)
parser.add_argument("--video-id", type=str)
parser.add_argument("--start-time", type=str)
parser.add_argument("--end-time", type=str)
parser.add_argument("--start-time-in-sec", type=str)
parser.add_argument("--end-time-in-sec", type=str)

args = parser.parse_args()

# general parameters
batch_size = args.batch_size
max_caption_length = args.max_caption_length
lr = args.lr
# Used for Adam Optimizer
optim_eps = 1e-5
model_name = "HACAModel_bs80_maxmode"

# Scheduled sampling parameter (decides whether to teacher force, or auto regress) (None or 0 means complete teacher forcing)
ss_epsilon = None


# All configurations used for training
config = {
    "audio_input_size" : 128,
    "visual_input_size" : 2048,
    "chunk_size" : {"audio" : 4, "visual" : 10}, # Granularity of high level encoder
    "audio_hidden_size" : {"low" : 128, "high" : 64},
    "visual_hidden_size" : {"low" : 512, "high" : 256},
    "local_input_size" : 1024,
    "global_input_size" : 1024,
    "global_hidden_size" : 256,
    "local_hidden_size" : 1024,
    "vocab_size" : 10004,
    "embedding_size" : 512
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
haca_model = HACAModel(config, device, batch_size, max_caption_length)
data = torch.load('models/{}.pt'.format("HACAModel_bs80_maxmode"),map_location=device)
haca_model.load_state_dict(data["state_dict"])
if device.type == "cuda" :
    haca_model.cuda()
passed_epochs = data["epoch"]

def convert_to_batch(features, modality="video"):
    batch = np.zeros((1, len(features), features.shape[1]))
    batch[0,:len(features),:] = features[:]
    return batch

def test_model(model):
    # Loads the iterator, if in training mode, and max_video != -1, only that much videos will be loaded for training.
    # However, for validation/testing, all the videos will be loaded.

    model.eval()
    ss_epsilon = 0
    video_features = np.load(args.video_features_path)
    audio_features = np.load(args.audio_features_path)
    vid_batch = convert_to_batch(video_features, modality="video")
    aud_batch = convert_to_batch(audio_features, modality="audio")
    temp=np.array([[10001]])
    batch = {}
    batch["audio_features"] = torch.from_numpy(aud_batch).type(torch.float32).to(device)
    batch["visual_features"] = torch.from_numpy(vid_batch).type(torch.float32).to(device)
    batch["caption_x"] = torch.from_numpy(temp).type(torch.long).to(device)
    # output_captions obtained from model
    output_caption = model(batch, ss_epsilon)
    output_caption = torch.cat(output_caption, dim=1)
    output_caption = torch.max(output_caption, dim=2, keepdim=False)[1]
    # Converting output caption to words (from the vocabulary)
    candidate = [id_to_word[id.item()] for id in output_caption[0,:]]
    # If the caption itself ended, then truncating the caption after that, otherwise, taking the entire caption
    if "<END>" in candidate:
        length = candidate.index("<END>") + 1
    else:
        length = max_caption_length
    candidate = candidate[:length]
    candidate = " ".join(candidate)
    # Appending the output to the res list
    return candidate
#print("info: <ST>",args.start_time," <ET>",args.end_time," <STIS>",args.start_time_in_sec,"<EDIS>",args.end_time_in_sec)
#print(args.video_id + ': <START> '+ test_model(haca_model))
caption = test_model(haca_model)
if '<UNK>' not in caption:
    print('<START>'+caption)
