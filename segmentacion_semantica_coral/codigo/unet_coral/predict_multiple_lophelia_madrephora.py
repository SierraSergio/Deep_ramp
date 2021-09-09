#en el caso del coral tenemos: 0=unclassified 1=live   2=dead

colores = [(180, 180, 180),(0,0,255) ,(244,244,244), (0,244,20), (0,50,200) ,(25,10,135), (255, 18, 0), (30,10,5) , (255, 18, 0), (100,40,245) ,(55,60,89),(200,8,103) ,(100,40,245),(0,64,128)]

from keras_segmentation.models.unet import resnet50_unet_896

inp_dir="C:/Users/ser/Desktop/E0717_TF16/"
out_dir="C:/Users/ser/Desktop/salidas/E0717_TF16_2corales/" 

path_fot="C:/Users/ser/Desktop/E0717_TF16/E0717_TF016.FOT"
nombre_fot="E0717_TF16"

model = resnet50_unet_896(n_classes=7,  input_height=5440, input_width=3616)

#inferencia 

weights_file = 'C:/Users/ser/Desktop/dataset_coral_varias_clases/checkpoints_coral_resnet_896x896_reales/vgg_resnet_local_14clases.2'

from keras_segmentation.predict_multiple_laser_porcentaje_conteo_2corales import predict_multiple

pr = predict_multiple( 
	checkpoints_path = "C:/Users/ser/Desktop/dataset_coral_varias_clases/checkpoints_coral_resnet_896x896_reales\\vgg_resnet_local_14clases", 
    #checkpoints_path = "C:/Users/ser/Desktop/15_imagenes_vivo_muerto_gaviera/checkpoints_7clases_vivo_muerto\\resnet_7clases_vivo_muerto",
	#inp= test_path,
    inp_dir=inp_dir,
	out_dir=out_dir,
    out_mask = out_dir+"masks/" ,
    overlay_img=True,
    class_names = ['unclassified','dead coral','Madrepora','Lophelia', 'pez','erizo cidaris', 'ceriantus','cangrejo','ceriantus2','estrella','regradella','otros corales','munida', 'basura','leiopathes'],
    show_legends = False,
    colors = colores,
    path_fot=path_fot,
    nombre_fot=nombre_fot
)