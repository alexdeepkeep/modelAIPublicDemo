# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from gan_models.tiny_yolo import TinyYoloNet
if not(__name__ == "demo") and not(__name__ == "__main__"):
    import sys
    sys.path.append('pytorchYOLOv4/')
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch.nn.functional as F
import argparse
# from ipdb import set_trace as st


"""hyper parameters"""
use_cuda = False

class MaxProbExtractor(torch.nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id=0, cls_id_tgt=1, num_cls=80):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.cls_id_tgt = cls_id_tgt
        self.num_cls = num_cls

    def set_cls_id_attacked(self, cls_id):
        self.cls_id = cls_id

    def set_cls_id_tgt(self, cls_id_tgt):
        self.cls_id_tgt = cls_id_tgt

    def forward(self, YOLOoutput):
        ## YOLOoutput size : torch.Size([1, 22743, 80])
        output_class = YOLOoutput[:,:,self.cls_id]
        output_class_tgt = YOLOoutput[:,:,self.cls_id_tgt]
        max_conf_target_obj_cls, max_conf_indexes  = torch.max(output_class, dim=1)
        # st()
        return max_conf_target_obj_cls, output_class, output_class_tgt

class DetectorYolov4():
    def __init__(self, cfgfile='./pytorchYOLOv4/cfg/yolov4.cfg', weightfile='./pytorchYOLOv4/weight/yolov4.pth', show_detail=True, tiny=False, epoch_save=-1):
        if tiny:
            cfgfile    = './pytorchYOLOv4/cfg/yolov4-tiny.cfg'
            weightfile = './pytorchYOLOv4/weight/yolov4-tiny.weights'
        self.show_detail = show_detail
        self.epoch_save = epoch_save
        if(self.show_detail):
            start_init        = time.time()
            self.m = Darknet(cfgfile)
            finish_init       = time.time()
            start_w           = time.time()
            # m.print_network()
            self.m.load_weights(weightfile)
            finish_w          = time.time()
            print(f'Loading Tiny-YOLOv4 weights from %s... Done!' % (weightfile)) if tiny else print(
                f'Loading YOLOv4 weights from %s... Done!' % (weightfile))
            start_d           = time.time()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                use_cuda = True
                self.m.cuda()
                # init MaxProbExtractor
                self.max_prob_extractor = MaxProbExtractor().cuda()
            else:
                use_cuda = False
                # init MaxProbExtractor
                self.max_prob_extractor = MaxProbExtractor()
            finish_d          = time.time()
            
            print('Yolov4 init model  Predicted in %f seconds.' % (finish_init - start_init))
            print('Yolov4 load weight Predicted in %f seconds.' % (finish_w - start_w))
            print('Yolov4 load device Predicted in %f seconds.' % (finish_d - start_d))
            print('Total time :%f ' % (finish_d - start_init))
        else:
            self.m = Darknet(cfgfile)
            # m.print_network()
            self.m.load_weights(weightfile)
            print('Loading Yolov4 weights from %s... Done!' % (weightfile))
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                use_cuda = True
                self.m.cuda()
                 # init MaxProbExtractor
                self.max_prob_extractor = MaxProbExtractor().cuda()
            else:
                use_cuda = False
                # init MaxProbExtractor
                self.max_prob_extractor = MaxProbExtractor()
        # 
    def detect(self, input_imgs, cls_id_attacked, epoch, clear_imgs=None, with_bbox=True):
        input_imgs        = F.interpolate(input_imgs, size=self.m.width).to(self.device)
        if with_bbox and (epoch % self.epoch_save == 0):
            boxes         = do_detect(self.m, input_imgs, 0.4, 0.6, use_cuda)
            bbox          = [torch.Tensor(box) for box in boxes] ## [torch.Size([3, 7]), torch.Size([2, 7]), ...]
        else:
            self.m.eval()
        # detections_tensor
        output            = self.m(input_imgs)
        detections_tensor_xy    = output[0]    # xy1xy2, torch.Size([1, 22743, 1, 4])
        detections_tensor_class = output[1] # conf, torch.Size([1, 22743, 80])
        self.max_prob_extractor.set_cls_id_attacked(cls_id_attacked)
        max_prob_obj_cls, output_class, output_class_target = self.max_prob_extractor(detections_tensor_class)
        # output_objectness, output_class, output_class_target = self.max_prob_extractor(output)
        # print("detections_tensor_class:",detections_tensor_class[1,:10,0])
        # print("max_prob_obj_cls",max_prob_obj_cls)

        # get overlap_score
        if not(clear_imgs == None):
            # resize image
            clear_imgs = F.interpolate(clear_imgs, size=self.m.width).to(self.device)
            # detections_tensor
            output_clear = self.m(clear_imgs)
            detections_tensor_xy_clear    = output_clear[0]    # xy1xy2, torch.Size([1, 22743, 1, 4])
            detections_tensor_class_clear = output_clear[1] # conf, torch.Size([1, 22743, 80])
            #
            max_prob_obj_cls_clear = self.max_prob_extractor(detections_tensor_class_clear)
            # count overlap
            output_score       = max_prob_obj_cls
            output_score_clear = max_prob_obj_cls_clear
            # overlap_score      = torch.abs(output_score-output_score_clear)
            # TODO: its a patch, the above line is the original
            overlap_score = torch.tensor(0).to(self.device)
        else:
            overlap_score = torch.tensor(0).to(self.device)
        # st()
        if with_bbox and (epoch % self.epoch_save == 0):
            return max_prob_obj_cls, output_class_target, overlap_score, bbox
        else:
            return max_prob_obj_cls, output_class_target, overlap_score, []
    

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco_new.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco_new.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./weight/yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/000000042070.jpg',
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.imgfile:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
