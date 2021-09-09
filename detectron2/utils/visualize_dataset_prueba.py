import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import cv2
import random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "C:/Users/ser/Desktop/hasty_leptometras/coco_circarock.json", "C:/Users/ser/Desktop/hasty_leptometras")
register_coco_instances("my_dataset_val", {}, "C:/Users/ser/Desktop/hasty_leptometras/coco_circarock.json", "C:/Users/ser/Desktop/hasty_leptometras")
register_coco_instances("my_dataset_test", {}, "C:/Users/ser/Desktop/hasty_leptometras/coco_circarock.json", "C:/Users/ser/Desktop/hasty_leptometras")

#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")
import random
from detectron2.utils.visualizer import Visualizer
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("prueba",vis.get_image()[:, :, ::-1])