import os
from pathlib import Path
from model.tool.darknet2pytorch import Darknet

import inspect


YOLOV4_PATH = Path(inspect.getfile(inspect.currentframe())).parent / "model"



def load_model():
    cfgfile = os.path.join(YOLOV4_PATH, 'cfg/yolov4.cfg')
    weightfile = os.path.join(YOLOV4_PATH, 'weight/yolov4.weights')

    yolo4_model = Darknet(cfgfile)
    yolo4_model.load_weights(weightfile)

    return yolo4_model
