# coding=utf-8

colores = [(180, 180, 180),(0,0,255) ,(0,244,20), (0,244,20), (0,50,200) ,(25,10,135), (255, 18, 0), (30,10,5) , (255, 18, 0), (100,40,245) ,(55,60,89),(200,8,103) ,(100,40,245),(0,64,128)]

from keras_segmentation.models.unet import resnet50_unet_896

video_path = "C:/Users/ser/Desktop/E0717_TV16/E0717_TV16.mp4"
out_video="C:/Users/ser/Desktop/E0717_TV16/salida/E0717_TV16_inferido.mp4" 

#en out_dir se iran guardando los txt
out_dir="C:/Users/ser/Desktop/E0717_TV16/salida/txt/" 

#path al archivo de excel con las coordenadas y más datos
excel="C:/Users/ser/Desktop/E0717_TV16/E0717_TV16_posi.xlsx"

model = resnet50_unet_896(n_classes=14 ,  input_height=5440, input_width=3616)

#inferencia 

weights_file = 'C:/Users/ser/Desktop/dataset_coral_varias_clases/checkpoints_coral_resnet_896x896_reales/vgg_resnet_local_14clases.2'
from keras_segmentation.predict_video import predict_video

pr = predict_video( 
	checkpoints_path = "C:/Users/ser/Desktop/dataset_coral_varias_clases/checkpoints_coral_resnet_896x896_reales\\vgg_resnet_local_14clases", 
    inp=video_path,
	output=out_video,
    overlay_img=True,
    class_names = ['unclassified','dead coral','Madrepora','Lophelia', 'pez','erizo cidaris', 'ceriantus','cangrejo','ceriantus2','estrella','regradella','otros corales','munida', 'basura','leiopathes'],
    show_legends = False,
    colors = colores,
    prediction_width=1920, prediction_height=1080,out_dir=out_dir,name_excel=excel
)