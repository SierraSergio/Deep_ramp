import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

#registro dataset 
register_coco_instances("cap_breton",{},"C:/Users/ser/detectron2/CapBreton_train/prueba.json", "C:/Users/ser/detectron2/CapBreton_train/CapBre_train/img")
cap_breton_metadata = MetadataCatalog.get("cap_breton")
dataset_dicts = DatasetCatalog.get("cap_breton")




cfg = get_cfg()
cfg.merge_from_file("C:/Users/ser/detectron2/cap_breton_rcnn_x101_weights/config.yaml")
cfg.DATASETS.TRAIN = ("cap_breton",)
cfg.DATASETS.TEST = ("cap_breton",)   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "C:/Users/ser/detectron2/cap_breton_rcnn_x101_weights/model_final.pth" # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has one class (ballon)
cfg.OUTPUT_DIR= "C:/Users/ser/detectron2/cap_breton_rcnn_x101_weights/"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()