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

FLAGS = flags.FLAGS


def OutputAudioEmbeddings(pathIn, row):
    video_id = row['video_id']
    video_path = row['video_path']
    split = row['split']
    full_path = os.path.join(pathIn, video_path)
    full_path = full_path.replace("%(ext)s", "wav")  # output file of the downloader path
    if split == 'train':
        full_path_cut = full_path.replace("train", "train/cut")
    elif split == 'test':
        full_path_cut = full_path.replace("test", "test/cut")

    # In this simple example, we run the examples from a single audio file through
    # the model. If none is provided, we generate a synthetic input.

    if os.path.isfile(full_path_cut):
      wav_file = full_path_cut

      examples_batch = vggish_input.wavfile_to_examples(wav_file)
      #print(examples_batch)

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
        #print(embedding_batch)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        print(postprocessed_batch)
        #print(postprocessed_batch.shape)
        np.save('audio_features/'+split+'/'+video_id,postprocessed_batch)




#np.save('video_features/'+split+'/'+video_id, input)

def main():
    video_data_path = "videodatainfo_2017.json"
    video_save_path = 'data/datasets/msrvtt/videos/'

    with open(video_data_path) as f:
        data = json.load(f)

    video_data = pd.DataFrame.from_dict(data["videos"])
    video_data['video_path'] = video_data.apply(
        lambda row: row['split'] + '/' + row['video_id'] + '_' + str(row['start time']) + '_' + str(
            row['end time']) + '.%(ext)s', axis=1)

    video_data.apply(lambda row: OutputAudioEmbeddings(video_save_path, row), axis=1)

if __name__ == '__main__':
    main()
