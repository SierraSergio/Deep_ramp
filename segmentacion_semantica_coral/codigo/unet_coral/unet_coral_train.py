#28ago2020  

''' opciones de entrenamiento
def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adadelta',
          do_augment=False,
          augmentation_name="aug_all")
'''


from keras_segmentation.models.unet import resnet50_unet_896

images_path = "C:/Users/ser/Desktop/dataset_motas"
segs_path = "C:/Users/ser/Desktop/dataset_motas"

model = resnet50_unet_896(n_classes=2 , input_height=896, input_width=896  )

if True:
 model.train(
    train_images =  images_path,
    train_annotations = segs_path,  
    checkpoints_path = "C:/Users/ser/Desktop/dataset_motas/pesos/resnet_mota" , epochs=5, auto_resume_checkpoint=True , 
    #load_weights= 'C:/Users/ser/Desktop/15_imagenes_vivo_muerto_gaviera/checkpoints_8clases/resnet_local_8clases.4',
    do_augment=False,
    gen_use_multiprocessing=False,
    #augmentation_name="aug_all"
 )


