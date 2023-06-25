"""
Show part query correspondence in single object images with AffCorrs
"""
# Standard imports
import os
import sys
from PIL import Image
import yaml
import numpy as np
import matplotlib.pyplot as plt
# Vision imports
import torch
import cv2
sys.path.append("..")
from UCL_AffCorrs.models.correspondence_functions import (overlay_segment, resize)
from UCL_AffCorrs.models.aff_corrs import AffCorrs_V1

# User-defined constants
# SUPPORT_DIR = "../affordance_database/usb/"
# TARGET_IMAGE_PATH = "./images/demo_affordance/eth.jpg"

SUPPORT_DIR = "../src/UCL_AffCorrs/affordance_database/hammer_handle/"
TARGET_IMAGE_PATH = "../src/UCL_AffCorrs/demos/images/demo_parts/basketball.jpeg"


# Other constants
PATH_TO_CONFIG  = "../src/UCL_AffCorrs/config/default_config.yaml"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLORS = [[255,0,0],[255,255,0],[255,0,255],
          [0,255,0],[0,0,255],[0,255,255]]

# Load arguments
with open(PATH_TO_CONFIG) as f:
    args = yaml.load(f, Loader=yaml.CLoader)
args['low_res_saliency_maps'] = False
args['load_size'] = 256

class ShowPartCorrespondence():
    def __init__(self):
        self.part_out_i = []
        return
    
    # Helper functions
    def load_rgb(self, path):
        """ Loading RGB image with OpenCV
        : param path: string, image path name. Must point to a file.
        """
        assert os.path.isfile(path), f"Path {path} doesn't exist"
        return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

    def viz_correspondence(self, im_a, im_b, parts_a, parts_b):
        """ Visualizes the correspondences
        : param im_a: np.ndarray, RGB image a
        : param im_b: np.ndarray, RGB image b
        : param parts_a: List[np.ndarray], list of part masks in a
        : param parts_b: List[np.ndarray], list of part masks in b
        """
        quer_img = im_a.astype(np.uint8)
        corr_img = im_b.astype(np.uint8)
        for i, part_i in enumerate(parts_a):
            quer_img = overlay_segment(quer_img, part_i,
                                    COLORS[i], alpha=0.3)
            part_out_i = resize(parts_b[i],corr_img.shape[:2]) > 0
            self.part_out_i = part_out_i
            corr_img = overlay_segment(corr_img, part_out_i,
                                    COLORS[i], alpha=0.3)

        _fig, ax = plt.subplots(1,2)
        ax[0].imshow(quer_img)
        ax[1].imshow(corr_img)
        plt.show()

    def run_result(self,origin_rgb):
        with torch.no_grad():
            model = AffCorrs_V1(args)

            # Prepare inputs
            img1_path = f"{SUPPORT_DIR}/prototype.png"
            aff1_path = f"{SUPPORT_DIR}/affordance.npy"
            rgb_a = self.load_rgb(img1_path)
            parts = np.load(aff1_path, allow_pickle=True).item()['masks']
            affordances = [None for _ in parts]
            rgb_b = origin_rgb
            # rgb_b = load_rgb(TARGET_IMAGE_PATH)
            # print(rgb_b)
            # print(rgb_b.shape)
            ## Produce correspondence
            print("start to find correspondences")
            model.set_source(Image.fromarray(rgb_a), parts, affordances)
            model.generate_source_clusters()
            model.set_target(Image.fromarray(rgb_b))
            model.generate_target_clusters()
            parts_out, aff_out = model.find_correspondences()
            print("found it!")
            ## Display correspondence
            self.viz_correspondence(rgb_a, rgb_b, parts, parts_out)
