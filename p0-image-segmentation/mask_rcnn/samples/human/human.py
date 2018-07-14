"""
Mask R-CNN
Train on the toy Human dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 human.py train --dataset=/path/to/human/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 human.py train --dataset=/path/to/human/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 human.py train --dataset=/path/to/human/dataset --weights=imagenet

    # Apply color splash to an image
    python3 human.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 human.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from glob import glob
import cv2
import zlib
import io
from PIL import Image
import json
import base64
import matplotlib.pyplot as plt


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class HumanConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "human"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Human

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class HumanDataset(utils.Dataset):

    def base64_2_mask(self, s):
        z = zlib.decompress(base64.b64decode(s))
        n = np.fromstring(z, np.uint8)
        mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
        return mask

    def mask_2_base64(self, mask):
        img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
        img_pil.putpalette([0,0,0,255,255,255])
        bytes_io = io.BytesIO()
        img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
        bytes = bytes_io.getvalue()
        return base64.b64encode(zlib.compress(bytes)).decode('utf-8')

    def to_annotation_path(self, path):
        return path.replace("/img", "/ann").replace(".png", ".json").replace(".jpg", ".json")

    def load_human(self, dataset_dir, subset):
        """Load a subset of the Human dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("human", 1, "human")

        # Train or test dataset?
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        all_image_paths = glob("{}/img/*.*g".format(dataset_dir))

        # Add images
        for image_path in all_image_paths:
            ann_path = self.to_annotation_path(image_path)
            filename = image_path.split("/")[-1].split(".")[0]
            
            annotation = json.load(open(ann_path, "r"))
            width = annotation["size"]["width"]
            height = annotation["size"]["height"]

            assert len(image_path) > 0, "image path should be non-empty but found {}".format(image_path)
            
            self.add_image(
                "human",
                image_id=filename,
                path=image_path,
                width=width, 
                height=height,
                annotation_path=ann_path)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Human dataset image, delegate to parent class.
        image_info = self.image_info[image_id]

        if image_info["source"] != "human":
            return super(self.__class__, self).load_mask(image_id)

        img_shape = [image_info["height"], image_info["width"]]
        js = json.load(open(image_info["annotation_path"], "r"))

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        masks = list()
        for obj in js['objects']:
            if obj['bitmap']:
                o = obj['bitmap']['origin']
                m = self.base64_2_mask(obj['bitmap']['data'])
                mask = np.zeros(shape=img_shape[:2])
                c, r = o
                h, w = m.shape
                mask[r:r+h,c:c+w] = m
                masks.append(mask)
            else:
                exterior = np.vstack(obj['points']['exterior'])
                interior = [np.vstack(x) for x in obj['points']['interior']]        
                # draw outer mask
                rr, cc = skimage.draw.polygon(exterior[:, 1], exterior[:,0], shape=img_shape[:2])        
                mask = np.zeros(shape=img_shape[:2])
                mask[rr, cc] = True
                # carve out holes
                for hole in interior:
                    rr, cc = skimage.draw.polygon(hole[:, 1], hole[:,0], shape=img_shape[:2])        
                    mask[rr, cc] = False
                masks.append(mask.astype(np.bool))

        masks = np.rollaxis(np.array(masks), 0, 3)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return masks.astype(np.bool), np.ones([masks.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "human":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, epochs, layers="heads"):
    """Train the model."""
    # Training dataset.
    dataset_train = HumanDataset()
    dataset_train.load_human(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = HumanDataset()
    dataset_val.load_human(args.dataset, "test")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers=layers)



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Humans.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Human/dataset/",
                        help='Directory of the Human dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--epochs', required=False,
                        default=30,
                        metavar="N",
                        help='Number of epochs to train')
    parser.add_argument('--layers', required=False,
                        default="heads",
                        metavar="/path/to/logs/",
                        help="layers: Allows selecting wich layers to train. It can be: (1) A regular expression to match layer names to train, (2) heads: The RPN, classifier and mask heads of the network, (3) all: All the layers, (4) 3+: Train Resnet stage 3 and up, (5) 4+: Train Resnet stage 4 and up, (6) 5+: Train Resnet stage 5 and up")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = HumanConfig()
    else:
        class InferenceConfig(HumanConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.epochs, args.layers)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
