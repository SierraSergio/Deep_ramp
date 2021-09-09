colores = [(180, 180, 180),(0,0,255) ,(255,244,227), (30, 164, 255), (0,50,200) ,(25,10,135), (255, 18, 0), (30,10,5) ,(128,0,255), (241, 222, 170), (8,8,45) ,(55,60,89),(200,8,103) ,(100,40,245), (0,64,128)]

from keras_segmentation.models.unet import resnet50_unet_896

images_path = "C:/Users/ser/Desktop/UNET_keras_coral/coral_dataset_unet/img/"
segs_path = "C:/Users/ser/Desktop/UNET_keras_coral/coral_dataset_unet/masks_inf_896"
test_path = "C:/Users/ser/Desktop/E0717_TF14/E0717_TF14_9595.JPG"


model = resnet50_unet_896(n_classes=13 ,  input_height=896, input_width=896 )

#inferencia 

weights_file = 'C:/Users/ser/Desktop/dataset_coral_varias_clases/checkpoints_coral_resnet_896x896_reales/vgg_resnet_local_14clases.2'
from keras_segmentation.predict import predict

pr = predict( 
	checkpoints_path = "C:/Users/ser/Desktop/dataset_coral_varias_clases/checkpoints_coral_resnet_896x896_reales\\vgg_resnet_local_14clases", 
	inp= test_path,
	out_fname="C:/Users/ser/Desktop/E0717_TF14_9595_1.png" ,
    #out_mask = "C:/Users/ser/Desktop/E0717_TF14_9579_1.png" ,
    overlay_img=True,
    class_names = ['unclassified','dead coral','Madrepora','Lophelia', 'pez','erizo cidaris', 'ceriantus','cangrejo','ceriantus2','estrella','regradella','otros corales','munida', 'basura','leiopathes'],
    show_legends = True,
    colors = colores
)