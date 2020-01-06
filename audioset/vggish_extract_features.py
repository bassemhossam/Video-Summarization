"""
 $ python vggish_extract_features.py --wav_file /path/to/a/wav/file
                                    --video_path /path/to/save/np/postprocessed_batch/array
"""


from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf
import json
import os
import pandas as pd

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if type(tf.contrib) != type(tf): tf.contrib._warning = None
flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

flags.DEFINE_string(
    'video_path', None,
    'Path to the input video wave file.')

FLAGS = flags.FLAGS


def OutputAudioEmbeddings(wav_file_path, save_path):

    # In this simple example, we run the examples from a single audio file through
    # the model. If none is provided, we generate a synthetic input.

    if os.path.isfile(wav_file_path) and not os.path.isfile(save_path+'.npy'):
        wav_file = wav_file_path
        print(wav_file_path)
        print(save_path+'.npy')
        examples_batch = vggish_input.wavfile_to_examples(wav_file)
        # print(examples_batch)

        # Prepare a postprocessor to munge the model embeddings.
        pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

        # If needed, prepare a record writer to store the postprocessed embeddings.
        writer = tf.python_io.TFRecordWriter(
            FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

        with tf.Graph().as_default(), tf.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: examples_batch})
            # print(embedding_batch)
            postprocessed_batch = pproc.postprocess(embedding_batch)
            # print(postprocessed_batch.shape)
            np.save(save_path, postprocessed_batch)


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if FLAGS.wav_file:
        wav_file = FLAGS.wav_file
    else:
        print("Error parsing the wav_file arg")

    if FLAGS.video_path:
        video_path = FLAGS.video_path
    else:
        print("Error parsing the video_path arg")

    OutputAudioEmbeddings(wav_file, video_path)


if __name__ == '__main__':
    main()
