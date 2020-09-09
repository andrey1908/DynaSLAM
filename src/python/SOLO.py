import os
import os.path as osp
import shutil
import tempfile
from scipy import ndimage
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, get_dist_info, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model, tensor2imgs, get_classes
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import cv2

import numpy as np

from mmdet.datasets.pipelines.compose import Compose
from mmdet.utils.registry import build_from_cfg
from mmdet.datasets.registry import PIPELINES
from mmcv.parallel.collate import collate

class Mask:
    """
    """
    def __init__(self, network_name):
        print('Initializing SOLO network...')

        # Root directory of the project
        ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
        cfg = mmcv.Config.fromfile(os.path.join(ROOT_DIR, network_name + '.py'))
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        # Path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, network_name + '.pth')

        # Build the model and load checkpoint
        self.model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
        checkpoint = load_checkpoint(self.model, COCO_MODEL_PATH, map_location='cpu')
        self.model.CLASSES = checkpoint['meta']['CLASSES']
        self.class_names = self.model.CLASSES
        self.model = MMDataParallel(self.model, device_ids=[0])
        self.model.eval()
        self.pipeline = Compose(cfg.test_pipeline[1:])
        self.classes_to_cut_out = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
        print('Initialated SOLO network...')

    def GetDynSeg(self,image,image2=None):
        h = image.shape[0]
        w = image.shape[1]
        if len(image.shape) == 2:
            im = np.zeros((h,w,3))
            im[:,:,0]=image
            im[:,:,1]=image
            im[:,:,2]=image
            image = im
        data = {'img_info': {'height': h, 'width': w}, 'filename': None, 'seg_prefix': None, 'proposal_file': None, 'bbox_fields': [], 'mask_fields': [], 'seg_fields': [], 'img': image, 'img_shape': (h, w, 3), 'ori_shape': (h, w, 3)}
        data = self.pipeline(data)
        data = collate([data])
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)
        # Visualize results
        mask = np.zeros((h,w))
        cur_result = results[0]
        if cur_result is None:
            return mask
        seg_label = cur_result[0]
        seg_label = seg_label.cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1]
        cate_label = cate_label.cpu().numpy()
        num_masks = seg_label.shape[0]
        for i in range(num_masks):
            if self.class_names[cate_label[i]] in self.classes_to_cut_out:
                cur_mask = seg_label[i, :, :]
                assert (cur_mask.shape[0] == h) and (cur_mask.shape[1] == w)
                mask[cur_mask == 1] = 1.
        return mask

