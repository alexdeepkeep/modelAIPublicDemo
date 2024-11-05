import torch
from torchvision.ops import nms

class Postprocessor:

    def __init__(self, confidence_threshold = 0.5, nms_threshold = 0.6):
        self.conf_thresh = confidence_threshold
        self.nms_thresh = nms_threshold

    def execute(self, output):
        box_array = output[0]
        # [batch, num, num_classes]
        confs = output[1]

        num_classes = confs.shape[2]

        # [batch, num, 4]
        box_array = box_array[:, :, 0]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = torch.max(confs, dim=2)[0]
        max_id = torch.argmax(confs, dim=2)

        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > self.conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            bboxes = []
            # NMS for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]

                # Apply NMS using torchvision's function
                keep = nms(ll_box_array, ll_max_conf, self.nms_thresh)
                
                if keep.size(0) > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]

                    # Append the detected objects as a tensor with 6 columns [x1, y1, x2, y2, confidence, class]
                    class_column = torch.full((ll_box_array.shape[0], 1), j, dtype=torch.float32, device=output[0].device)
                    bboxes.append(torch.cat([ll_box_array, ll_max_conf.unsqueeze(1), class_column], dim=1))

            # Append the list of detected objects as a tensor for this image
            bboxes_batch.append(torch.cat(bboxes, dim=0) if bboxes else torch.empty(0, 6))  # 6 columns

        # Stack the list of tensors into a batch tensor
        return bboxes_batch
