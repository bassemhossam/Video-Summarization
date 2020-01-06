from __future__ import unicode_literals

import torch
import argparse
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import numpy as np
import os


def extractFeatures(file_path, save_path):

    if os.path.isfile(file_path) and not os.path.isfile(save_path + '.npy'):
        count = 0
        vidcap = cv2.VideoCapture(file_path)
        success, image = vidcap.read()
        success = True
        img2vec = Img2Vec()
        features = []
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
            success, image = vidcap.read()
            if success:
                # cv2.imwrite(pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
                cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_im)
                vec = img2vec.get_vec(img)
                features.append(vec)

            count = count + (1 / 3)
        vidcap.release()
        features_npy = np.asarray(features)
        np.save(save_path, features_npy)


class Img2Vec():

    def __init__(self, model='resnet-152', layer='default', layer_output_size=2048):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        if self.device.type == "cuda":
            self.model.cuda()
        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)


        my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        if self.device.type == 'cuda':
            image = Variable(image).to(device=self.device)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:

            return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-152':
            model = models.resnet152(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path",
                        help="path of the input video")
    parser.add_argument("--video-features-path",
                        help="path of the video features")
    args = parser.parse_args()

    video_path = args.video_path
    video_features_path = args.video_features_path

    extractFeatures(video_path, video_features_path)


if __name__ == "__main__":
    main()
