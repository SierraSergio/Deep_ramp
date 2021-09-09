from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "C:/Users/ser/Desktop/hasty_leptometras/coco_circarock.json", "C:/Users/ser/Desktop/hasty_leptometras")
register_coco_instances("my_dataset_val", {}, "C:/Users/ser/Desktop/hasty_leptometras/coco_circarock.json", "C:/Users/ser/Desktop/hasty_leptometras")
register_coco_instances("my_dataset_test", {}, "C:/Users/ser/Desktop/hasty_leptometras/coco_circarock.json", "C:/Users/ser/Desktop/hasty_leptometras")