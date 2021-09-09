#version 18 de marzo dde 2021, funciona.

#inferimos con detectron2 para conseguir las mascaras(binarias) y aplicamos cv2.inpaint para tratar de suavizar el color.
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from PIL import Image
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os


register_coco_instances("all_especies",{},"C:/Users/ser/detectron2/circarock_entrenamiento_4especies/train_4especies.json", "C:/Users/ser/detectron2/circarock_entrenamiento_4especies/img")
all_especies_metadata = MetadataCatalog.get("all_especies")
dataset_dicts = DatasetCatalog.get("all_especies")

cfg = get_cfg()
cfg.merge_from_file("C:/Users/ser/review_object_detection_metrics/4especie_val_TF23/4especies_entreno_280721_prueba3/config.yaml")

# load weights
cfg.DATASETS.TRAIN = ("all_especies",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "C:/Users/ser/review_object_detection_metrics/4especie_val_TF23/mejo_lepto/model_final.pth"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 3 classes (data, fig, hazelnut)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
cfg.DATASETS.TEST = ("all_especies", )
predictor = DefaultPredictor(cfg)

#image
im = "C:/Users/ser/Desktop/IA418_TF23_0311.JPG"
im=cv2.imread(im)

#inference
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1],
                   metadata=all_especies_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
    )
v,imagen,nuevo_negro = v.draw_instance_predictions(outputs["instances"].to("cpu"))
v.save("C:/Users/ser/Desktop/IA418_TF23_0311_salida_4especies.jpg")