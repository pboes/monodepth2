
from __future__ import absolute_import, division, print_function
from utils import download_model_if_doesnt_exist
from layers import disp_to_depth
import networks


import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

sys.path.append("monodepth2/")

STEREO_SCALE_FACTOR = 5.4


class DepthPredictor:
    def __init__(self) -> None:
        self.model_name = "mono+stereo_no_pt_640x192"
        self.device = torch.device("cpu")
        self.scale_factor = STEREO_SCALE_FACTOR
        self.load_model()

    def load_model(self):
        model_path = os.path.join("models", self.model_name)
        download_model_if_doesnt_exist(self.model_name)
        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        print("   Loading pretrained encoder")
        encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(self.device)
        encoder.eval()
        self.encoder = encoder

        print("   Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(self.device)
        depth_decoder.eval()
        self.depth_decoder = depth_decoder

    # def predict_depths(self, input_image):
        # depth_map = self.generate_depth_map(input_image)
        # return self.depth_from_depth_map(depth_map)

    def predict_depths(self, input_image):
        original_width, original_height = input_image.size
        input_image = input_image.resize(
            (self.feed_width, self.feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(self.device)
        features = self.encoder(input_image)
        outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        self.scaled_disp, self.depth = disp_to_depth(disp_resized, 0.1, 100)
        metric_depth = STEREO_SCALE_FACTOR * self.depth.cpu().detach().numpy()
        return metric_depth

    def save_depths_to_jpeg(self, depths, name):
        # Saving colormapped depth image
        disp_resized_np = self.scaled_disp.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(
            vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[
                          :, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        # name_dest_im = os.path.join(
        #     output_directory, "{}_disp.jpeg".format(output_name))
        im.save(name)
    # def depth_from_depth_map(self, depth_map):
    #     return self.scale_factor / depth_map
