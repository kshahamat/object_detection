# Install pyyaml + detectron2(from github)
# !python -m pip install pyyaml==5.1
# Detectron2 has not released pre-built binaries for the latest pytorch (https://github.com/facebookresearch/detectron2/issues/4053)
# so we install from source instead. This takes a few minutes.
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import detectron2
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


class ObjectFeatures:

    def __init__(self):

        # Create config
        cfg = get_cfg()
        
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.DEVICE = "cpu"


        self.predictor = DefaultPredictor(cfg)

        self.index_to_class_dict = {0: u'__background__',
            1: u'person',
            2: u'bicycle',
            3: u'car',
            4: u'motorcycle',
            5: u'airplane',
            6: u'bus',
            7: u'train',
            8: u'truck',
            9: u'boat',
            10: u'traffic light',
            11: u'fire hydrant',
            12: u'stop sign',
            13: u'parking meter',
            14: u'bench',
            15: u'bird',
            16: u'cat',
            17: u'dog',
            18: u'horse',
            19: u'sheep',
            20: u'cow',
            21: u'elephant',
            22: u'bear',
            23: u'zebra',
            24: u'giraffe',
            25: u'backpack',
            26: u'umbrella',
            27: u'handbag',
            28: u'tie',
            29: u'suitcase',
            30: u'frisbee',
            31: u'skis',
            32: u'snowboard',
            33: u'sports ball',
            34: u'kite',
            35: u'baseball bat',
            36: u'baseball glove',
            37: u'skateboard',
            38: u'surfboard',
            39: u'tennis racket',
            40: u'bottle',
            41: u'wine glass',
            42: u'cup',
            43: u'fork',
            44: u'knife',
            45: u'spoon',
            46: u'bowl',
            47: u'banana',
            48: u'apple',
            49: u'sandwich',
            50: u'orange',
            51: u'broccoli',
            52: u'carrot',
            53: u'hot dog',
            54: u'pizza',
            55: u'donut',
            56: u'cake',
            57: u'chair',
            58: u'couch',
            59: u'potted plant',
            60: u'bed',
            61: u'dining table',
            62: u'toilet',
            63: u'tv',
            64: u'laptop',
            65: u'mouse',
            66: u'remote',
            67: u'keyboard',
            68: u'cell phone',
            69: u'microwave',
            70: u'oven',
            71: u'toaster',
            72: u'sink',
            73: u'refrigerator',
            74: u'book',
            75: u'clock',
            76: u'vase',
            77: u'scissors',
            78: u'teddy bear',
            79: u'hair drier',
            80: u'toothbrush'}

    
    def get_predictions(self, image):


        # output for each image is class : list of confidence scores

        output = self.predictor(image)
        instances = output["instances"]
        scores = instances.get_fields()["scores"].tolist()
        pred_classes = instances.get_fields()["pred_classes"].tolist()
        output_classes = list(map(lambda x: self.index_to_class_dict[x+1], pred_classes))

        output_dict = {}
        for i, classes in enumerate(pred_classes):

            if classes in output_dict:
                output_dict[classes].append(scores[i])
            else:
                output_dict[classes] = [scores[i]]

        return output_dict
