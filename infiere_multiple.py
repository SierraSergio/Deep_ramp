#version 18 de marzo dde 2021


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from PIL import Image
import numpy as np
import os
import glob
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import csv
import errno




def distancia_laser(img_real,a,b,col_real,row_real,laseres_x,laseres_y):
    pixel_real=img_real[a,b]
          
    if pixel_real[0]>245 and pixel_real[1]==255 and pixel_real[2]>240 :
        #print("he entrado", pixel_real, b,a)
        laseres_x.append(b)
        laseres_y.append(a)
        #opciones: guardar un vector con todos los valores que esten entre estos puntos y ver cual es el mayor y el menor y entonces restar.
    return laseres_x,laseres_y
    
#registro dataset 
register_coco_instances("cap_breton",{},"C:/Users/ser/detectron2/4especies_coco/coco_4especies.json", "C:/Users/ser/detectron2/4especies_coco/")
cap_breton_metadata = MetadataCatalog.get("cap_breton")
dataset_dicts = DatasetCatalog.get("cap_breton")

cfg = get_cfg()

# load weights
cfg.MODEL.WEIGHTS = "C:/Users/ser/detectron2/pesos_4especies_AP94/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.merge_from_file("C:/Users/ser/detectron2/pesos_4especies_AP94/config.yaml")

# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

transectos=["IA418_TF23","IA418_TF24"]
for nombre_transecto in transectos:
    try:
        os.mkdir('C:/Users/ser/detectron2/salidas/salida_inferencia/'+nombre_transecto)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
    #image
    im_path = "C:/Users/ser/detectron2/salidas/"+nombre_transecto+"/"
    with open('C:/Users/ser/detectron2/salidas/'+nombre_transecto+"/"+nombre_transecto+'.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(['Transect', 'Photo', 'Sp1','Sp2','Sp3','Sp4','Sp5','Sp6','Area','Dens_Sp1','Dens_Sp2','Dens_Sp3','Dens_Sp4','Dens_Sp5','Dens_Sp6'])
        for imagen in glob.glob(im_path+"*.jpg"):
            nombre=imagen.split("/"+nombre_transecto+"\\")
            nombre=nombre[1]
            print(nombre)
            im=cv2.imread(imagen)
            #inference
            outputs = predictor(im)
            print(outputs)
            '''
            #quiero la subclase de las clses. contar cuantos de cada clase hay y pasarlo a csv
            instances=outputs.get('instances')
            classes_tensor=instances.pred_classes
            classes_tensor=classes_tensor.cpu()
            classes=classes_tensor.numpy()
            
            sp1=sp2=sp3=sp4=sp5=sp0=0
            d_sp1=d_sp2=d_sp3=d_sp4=d_sp5=d_sp0=0
            transecto='ICB19'
        
            #calcular el area
            #bucle que recorre la imagen para sacar la distancia de los punteros.
            imagen_real = Image.open(imagen)
            col_real,row_real =  imagen_real.size
            laseres_x=[]
            laseres_y=[]
            
            for a in np.arange((row_real/2-row_real/4),(row_real/2+row_real/4)):
                for b in np.arange((col_real/2-col_real/4),(col_real/2+col_real/4)):
                    laseres_x,laseres_y=distancia_laser(im,int(a),int(b),col_real,row_real,laseres_x,laseres_y)  
                    
            
            distancia_real_laser=0.25#m
            
            if len(laseres_x)>0 and len(laseres_y)>0:
                min_x=min(laseres_x)
                max_x=max(laseres_x)
                min_y=min(laseres_y)
                max_y=max(laseres_y)
                distancia_x=max_x-min_x
                distancia_y=max_y-min_y
                if distancia_x>0 and distancia_y>0:
                    area="{:.2f}".format((col_real*distancia_real_laser/distancia_x)*(row_real*distancia_real_laser/distancia_y))
                    area=float(area)
                else:
                    area="NULL"
            else:
                area="NULL"
                distancia_x="NULL"
                distancia_y="NULL"
            
            
            #conteo de especies e cada clase
            for a in classes:
                if a==0:
                    sp0=sp0+1
                if a==1:
                    sp1=sp1+1
                if a==2:
                    sp2=sp2+1
                if a==3:
                    sp3=sp3+1
                if a==4:
                    sp4=sp4+1
                if a==5:
                    sp5=sp5+1

            
            if area=="NULL":
                d_sp0=d_sp1=d_sp2=d_sp3=d_sp4=d_sp5=d_sp6="NULL"

            else:
                d_sp0=sp0/area
                d_sp1=sp1/area
                d_sp2=sp2/area
                d_sp3=sp3/area
                d_sp4=sp4/area
                d_sp5=sp5/area

                
                d_sp0="{:.2f}".format(d_sp0)            
                d_sp1="{:.2f}".format(d_sp1)
                d_sp2="{:.2f}".format(d_sp2)
                d_sp3="{:.2f}".format(d_sp3)
                d_sp4="{:.2f}".format(d_sp4)
                d_sp5="{:.2f}".format(d_sp5)

            
            #csv
            employee_writer.writerow([transecto,nombre,sp1,sp2,sp3,sp4,sp5,sp0,area,d_sp1,d_sp2,d_sp3,d_sp4,d_sp5,d_sp0])
            '''
            v = Visualizer(im[:, :, ::-1],
                               metadata=cap_breton_metadata, 
                               scale=1, 
                               instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
                )
            
            v,imagen,nuevo_negro = v.draw_instance_predictions(outputs["instances"].to("cpu"))
               
        
            v.save("C:/Users/ser/detectron2/salidas/salida_inferencia/"+nombre_transecto+"/"+nombre)