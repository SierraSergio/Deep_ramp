#version 12 de enero de 2021. En este caso, cambiamos de donde obtenemos las coordenadas, no es un fot es un excel que tiene tomadas las medidas cada segundo,en teoría estos videos son
#de 25fps en formato wmv, al pasarlo a mp4 se convierte en 30fps. es decir, tengo que leer una línea del excel cada 30 fps y meterla en un txt
#por un motivo que desconozco la salida del video era a 60fps lo he forzado a que sea a 30fps con la función set_video.

#version final 28 de enero con conteo de ceriantus.
import glob
import random
import json
import os
import six
import pandas as pd
import xlrd
import pyshine as ps
import sys

import cv2
import numpy as np
from tqdm import tqdm
from time import time
import skimage
from skimage.color import rgb2gray

from keras_segmentation.train import find_latest_checkpoint
from .data_utils.data_loader import get_image_array, get_segmentation_array,\
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
from .models.config import IMAGE_ORDERING


random.seed(DATA_LOADER_SEED)

def write_txt(out_dir,name_video,num_especies, area,fotograma,excel,coral_vivo,coral_muerto,unclassified,cont_total,distancia_x,ceriantus,regradella,leiopathes):
        #Abrimos el fichero excel
        datos = xlrd.open_workbook(excel) 
        sh = datos.sheet_by_index(0)
        n_col= sh.ncols
        dato=[]
        for n in range(n_col):
            cell = sh.cell(fotograma+1,n) 
            #print(cell)
            dato.append(cell)
            
        #print(dato)
        f = open(out_dir+"/"+str(fotograma)+".txt", "w")
        f.write("Transecto,  SHIP_Lon, SHIP_SOG, SHIP_COG, Water_depth, SUB1Lon, SUB1_Lat, SUB1_Altitude, SUB1_COG,SUB1_Pitch,SUB1_Roll, Salinity, SUB1_Depth, Density, Temperature")
        f.write("\n")
        f.write(str(dato[0])+","+str(dato[3])+","+str(dato[4])+str(dato[5])+","+str(dato[6])+","+str(dato[7])+","+str(dato[8])+","+str(dato[9])+str(dato[10])+","+str(dato[11])+","+str(dato[12])+","+str(dato[13])+","+str(dato[14])+","+str(dato[15])+","+str(dato[16])+","+str(dato[17]))
        f.write("\n")
        f.write("Transecto,second_frame,area,live_area,dead_area,live_%,dead_%,ceriantus,regradella,leiopathes")
        f.write("\n")
        if area=="NULL":
            f.write(name_video+","+str(fotograma)+", NULL, NULL, NULL" +","+str(coral_vivo/cont_total)+","+str(coral_muerto/cont_total)+","+str(ceriantus)+","+str(regradella)+","+str(leiopathes)+", NULL, NULL, NULL")
            f.write("\n")
            
        else:
            f.write(name_video+","+str(fotograma)+","+str(area)+","+str((coral_vivo/cont_total)*area)+","+str((coral_muerto/cont_total)*area) +","+str(coral_vivo/cont_total)+","+str(coral_muerto/cont_total)+","+str(ceriantus)+","+str(regradella)+","+str(leiopathes)+","+str(ceriantus/area)+","+str(regradella/area)+","+str(leiopathes/area))
            f.write("\n") 
        f.write("Transecto: "+str(name_video))
        f.write("\n")
        f.write("segundo: "+str(fotograma))
        f.write("\n")
        

        if area=="NULL":
            f.write("Image surface area(m2): NULL ")
            f.write("\n")
            f.write("Live coral surface area(m2): NULL " )
            f.write("\n")
            f.write("Dead coral surface area(m2):: NULL" )
            f.write("\n")
           
        else:
            f.write("Image surface area(m2): "+str(area) )
            f.write("\n")
            f.write("Live coral surface area(m2): "+str((coral_vivo/cont_total)*area) )
            f.write("\n")
            f.write("Dead coral surface area(m2): "+str((coral_muerto/cont_total)*area) )
            f.write("\n")
            
        f.write("Live coral(%): "+str(coral_vivo/cont_total))
        f.write("\n")
        f.write("Dead coral (%): "+str(coral_muerto/cont_total))
        f.write("\n")
        f.write("Amount of Ceriantus: "+str(ceriantus))
        f.write("\n")
        f.write("Amount of Regradella: "+str(regradella))
        f.write("\n")
        f.write("Amount of leiopathes: "+str(leiopathes))
        f.write("\n")
        if area=="NULL":
            f.write("Density of Ceriantus: NULL unit/m2")
            f.write("\n")
            f.write("Density of Regradella: NULL unit/m2")
            f.write("\n")
            f.write("Density of Leiopathes: NULL unit/m2")
        else:
            f.write("Density of Ceriantus: "+str(ceriantus/area)+" unit/m2")
            f.write("\n")
            f.write("Density of Regradella: "+str(regradella/area)+" unit/m2")
            f.write("\n")
            f.write("Density of Leiopathes: "+str(leiopathes/area)+" unit/m2")

def distancia_laser(img_real,a,b,col_real,row_real,laseres_x):
    pixel_real=img_real[a,b]      
    #if pixel_real[0]>235 and pixel_real[1]>252 and pixel_real[2]>235:
    if  pixel_real[1]>253 :
        #print("he entrado", pixel_real, b,a)
        #print(pixel_real)
        laseres_x.append(b)
        
        #opciones: guardar un vector con todos los valores que esten entre estos puntos y ver cual es el mayor y el menor y entonces restar.
    return laseres_x

def cuenta_pixel (imagen_inf, cont_0, cont_1, cont_2 ,cont_3,cont_4,cont_5,cont_6,cont_7,cont_8,cont_9,cont_10,cont_11,cont_12,cont_13, j, k):
   
   pixel_inf = imagen_inf[j,k] #pixel de la imagen inferida
   
   
      
   if pixel_inf == 0 :
      cont_0 = cont_0 + 1
      #cuenta los pixeles inferidas de la clase "unclassified"
      
   if pixel_inf==1 :
      cont_1 = cont_1+ 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf== 2 :
      cont_2= cont_2 + 1
      #cuenta los pixeles inferidas de la clase "dead"   
      
   if pixel_inf==3 :
      cont_3 = cont_3+ 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf == 4 :
      cont_4 = cont_4 + 1
      #cuenta los pixeles inferidas de la clase "dead"   
   if pixel_inf == 5:
      cont_5= cont_5 + 1
      #cuenta los pixeles inferidas de la clase "unclassified"
      
   if pixel_inf==6 :
      cont_6 = cont_6 + 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf == 7 :
      cont_7 = cont_7+ 1
      #cuenta los pixeles inferidas de la clase "dead"  
      
   if pixel_inf == 8 :
      cont_8 = cont_8 + 1
      #cuenta los pixeles inferidas de la clase "unclassified"
      
   if pixel_inf==9 :
      cont_9 = cont_9 + 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf == 10 :
      cont_10 = cont_10 + 1
      #cuenta los pixeles inferidas de la clase "dead"   
      
   if pixel_inf == 11 :
      cont_11= cont_11 + 1
      #cuenta los pixeles inferidas de la clase "unclassified"
      
   if pixel_inf==12 :
      cont_12 = cont_12 + 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf == 13 :
      cont_13 = cont_13+ 1
      #cuenta los pixeles inferidas de la clase "dead"   

      
   return  cont_0, cont_1, cont_2,cont_3,cont_4,cont_5,cont_6,cont_7,cont_8,cont_9,cont_10,cont_11,cont_12,cont_13

def model_from_checkpoint_path(checkpoints_path):

    from .models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    #OJO!! el problema es con un cambio en las extensiones de los chekcpoints in TF>2.1
    #estoy probando a cambiar manualmente la extensión .index del último a un número cualquiera
    #latest_weights = find_latest_checkpoint(checkpoints_path)
    #latest_weights = 'checkpoints_prueba/vgg_unet_1.9'
    #latest_weights = 'checkpoints_grass1/vgg_unet_grass.last9_sinDataAug'
    from video_predict_14clases_laser_porcentaje import weights_file # importa la variable global 
    latest_weights = weights_file  # variable global para el apaño
    print('OJO!! apaño con el fichero de checkpoint ', latest_weights)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes-1):
        
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img


def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img


def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):

    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)

        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img


def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None,contador=None,area=None,cont_0=None,cont_1=None,cont_2=None,cont_3=None,cont_total=None,distancia_x=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)
    
    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"
    #print("holaaaa",inp)
    #inp ya es la matriz rgb
    '''
    if isinstance(inp, six.string_types):
    '''
    #inp = cv2.imread(inp)
    assert len(inp.shape) == 3, "Image should be h,w,3 "
    shape=inp.shape
    
    output_width = model.output_width
    output_height = model.output_height
    
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    
    #calculo de la distancia entre laseres
    if contador%25==0 or contador==1:
        laseres_x=[]
        #print("he entrado!!!")
        
        for a in np.arange((shape[0]/2-shape[0]/100),(shape[0]/2+shape[0]/100)):
            for b in np.arange((shape[1]/2-shape[1]/16),(shape[1]/2+shape[1]/16)):
                laseres_x=distancia_laser(inp,int(a),int(b),shape[0],shape[1],laseres_x)  
        
        #cálculo del area abarcada en la imagen
        distancia_real_laser=0.2#metros
        #print(laseres_x)
        
        #si no ha detectado ningun láser el área sera NULL
        if len(laseres_x)>0 :
            min_x=min(laseres_x)
            max_x=max(laseres_x)
            
            distancia_x=max_x-min_x
            
            if distancia_x>0:
                area=(shape[0]*distancia_real_laser/distancia_x)*(shape[1]*distancia_real_laser/distancia_x)
            else:
                area="NULL"
        else:
            area="NULL"
            distancia_x="NULL"
            distancia_y="NULL"
        
    #generamos la matriz de clases
    x = get_image_array(inp, input_width, input_height,
                        ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    #en este punto se tiene la matriz de clases
   
    ceriantus=leiopathes=regradella=0
    if contador%25==0 or contador==0 or contador==1:
        #creo una matriz vacia para rellenarla con los nuevos valores.
        conteo=np.empty((output_width, output_height))
        #A partir de aqui tendriamos la mascara, pero esta va a ser practicamente negra.La paso a un color blanco los que sean ceriantus para probar.
        for y in range(output_width):
            for z in range(output_height):
                if pr[y,z]==6 or pr[y,z]==8:
                    conteo[y,z]=50
                else:
                    conteo[y,z]=250
        masks='C:/Users/ser/Desktop/E0717_TV16/masks/1.jpg'        
        cv2.imwrite(masks, conteo)      
        #funcion para conteo de ceriantus en cada segundo
        im = cv2.imread(masks, cv2.IMREAD_GRAYSCALE)
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 20
        params.maxThreshold = 250
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 0
        params.maxArea=5000

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)
        # Detect blobs.
        keypoints = detector.detect(im)
        ceriantus=len(keypoints)
        ceriantus=int(ceriantus)
        print(ceriantus)
        #conteo de regradella
        conteo=np.empty((output_width, output_height))
        for z in range(output_height):
            if pr[y,z]==10:
                    conteo[y,z]=50
            else:
                    conteo[y,z]=250
        masks='C:/Users/ser/Desktop/count/2.jpg'        
        cv2.imwrite(masks, conteo)      
       
        
        #funcion para conteo de ceriantus en cada segundo
        im = cv2.imread(masks, cv2.IMREAD_GRAYSCALE)
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 20
        params.maxThreshold = 250
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 0
        params.maxArea=5000

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)
        # Detect blobs.
        keypoints = detector.detect(im)
        regradella=len(keypoints)
        regradella=int(regradella)
        
        #conteo de leiopathes
        conteo=np.empty((output_width, output_height))
        for z in range(output_height):
            if pr[y,z]==14:
                    conteo[y,z]=50
            else:
                    conteo[y,z]=250
        masks='C:/Users/ser/Desktop/count/3.jpg'        
        cv2.imwrite(masks, conteo)      
       
        
        #funcion para conteo de ceriantus en cada segundo
        im = cv2.imread(masks, cv2.IMREAD_GRAYSCALE)
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 20
        params.maxThreshold = 250
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 0
        params.maxArea=5000
        
        
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)
        # Detect blobs.
        keypoints = detector.detect(im)
        leiopathes=len(keypoints)
        leiopathes=int(leiopathes)
    
    #print(pr)
    #calcular el porcentaje de cada clase
    if contador%25==0 or contador==1:
        cont_0=cont_1=cont_2=cont_3=cont_4=cont_5=cont_6=cont_7=cont_8=cont_9=cont_10=cont_11=cont_12=cont_13=0 
        for a in range(output_width):
            for b in range(output_height):
               cont_0, cont_1, cont_2,cont_3,cont_4,cont_5,cont_6,cont_7,cont_8,cont_9,cont_10,cont_11,cont_12,cont_13 = cuenta_pixel (pr, cont_0, cont_1, cont_2,cont_3,cont_4,cont_5,cont_6,cont_7,cont_8,cont_9,cont_10,cont_11,cont_12,cont_13, a, b)
        
        #print(cont_0, cont_1, cont_2,cont_3,cont_4,cont_5,cont_6,cont_7,cont_8,cont_9,cont_10,cont_11,cont_12,cont_13)
        cont_total=cont_0+ cont_1+ cont_2+cont_3+cont_4+cont_5+cont_6+cont_7+cont_8+cont_9+cont_10+cont_11+cont_12+cont_13
        
    
    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height)
    

    
    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr,area,cont_0,cont_1,cont_2,cont_3,cont_total,distancia_x,ceriantus,regradella,leiopathes


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))
        inps = sorted(inps)

    assert type(inps) is list
    #print(inps)
    all_prs = []
    
    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")
        #print("esta es la imagen que entra: ", inp)
        #print("tamaño de la imagen",imp.size)
        pr = predict(model, inp, out_fname,
                     overlay_img=overlay_img, class_names=class_names,
                     show_legends=show_legends, colors=colors,
                     prediction_width=prediction_width,
                     prediction_height=prediction_height)

        all_prs.append(pr)

    return all_prs


def set_video(inp, video_name):
    cap = cv2.VideoCapture(inp)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps=25
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return cap, video, fps


def predict_video(model=None, inp=None, output=None,
                  checkpoints_path=None, display=False, overlay_img=True,
                  class_names=None, show_legends=False, colors=class_colors,
                  prediction_width=None, prediction_height=None,out_dir=None,name_excel=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)
    n_classes = model.n_classes
    name_video=inp
    num_especies=len(class_names)
    num_frame=0
    fotograma=0
    leiopathes=regradella=ceriantus=0
    distancia_x="NULL"
    cont_0=cont_1=cont_2=cont_3=cont_total=0
    area="NULL"
    excel=name_excel
    cap, video, fps = set_video(inp, output)
    while(cap.isOpened()):
        prev_time = time()
        ret, frame = cap.read()
        if frame is not None:
            num_frame=num_frame+1
            pr,area,cont_0, cont_1,cont_2, cont_3,cont_total,distancia_x,ceriantus,regradella,leiopathes = predict(model=model, inp=frame,contador=num_frame,area=area,cont_0=cont_0,cont_1=cont_1,cont_2=cont_2,cont_3=cont_3,cont_total=cont_total,distancia_x=distancia_x)
            coral_vivo=cont_2+cont_3
            coral_muerto=cont_1
            unclassified=cont_0
                
            fused_img = visualize_segmentation(
                pr, frame, n_classes=n_classes,
                colors=colors,
                overlay_img=overlay_img,
                show_legends=show_legends,
                class_names=class_names,
                prediction_width=prediction_width,
                prediction_height=prediction_height
                )
            coral_vivo=cont_2+cont_3
            
            if num_frame==1:
                cv2.imwrite('C:/Users/ser/Desktop/E0717_TV16/frames/'+str(num_frame)+'.jpg',fused_img)
            if num_frame%25==0:
                cv2.imwrite('C:/Users/ser/Desktop/E0717_TV16/frames/'+str(num_frame/25)+'.jpg',fused_img)
            if num_frame==1 or num_frame%25==0 :                 
                write_txt(out_dir,name_video,num_especies, area,fotograma,excel,coral_vivo,coral_muerto,unclassified,cont_total,distancia_x,ceriantus,regradella,leiopathes)
                fotograma=fotograma+1
            
            if area=="NULL":
                #fused_img=cv2.putText(fused_img,'Image surface area(m2): NULL',(1200,910), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,0), 2)
                fused_image =  ps.putBText(fused_img,'Image surface area(m2): NULL',text_offset_x=prediction_width-400,text_offset_y=prediction_height-220,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            else:
                #fused_img=cv2.putText(fused_img,'Image surface area(m2): {:.2f}'.format(area),(1200,910), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,0), 2)
                fused_image =  ps.putBText(fused_img,'Image surface area(m2): {:.2f}'.format(area),prediction_width-400,text_offset_y=prediction_height-220,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)

            text='Proportions of cold-water coral colonies: '
            text1='Living coral(%): {:.2f}  '.format(((coral_vivo)/cont_total)*100)
            text2='Dead coral(%): {:.2f}'.format((cont_1/cont_total)*100)
            text3='Amount of Ceriantus:  {}'.format(ceriantus)
            text4='Amount of Regradellas:  {}'.format(regradella)
            text5='Amount of Leiopathes:  {}'.format(leiopathes)
            fused_image=ps.putBText(fused_img,text,text_offset_x=prediction_width-400,text_offset_y=prediction_height-190,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            fused_image=ps.putBText(fused_img,text1,text_offset_x=prediction_width-400,text_offset_y=prediction_height-160,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            fused_image=ps.putBText(fused_img,text2,text_offset_x=prediction_width-400,text_offset_y=prediction_height-130,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            fused_image=ps.putBText(fused_img,text3,text_offset_x=prediction_width-400,text_offset_y=prediction_height-100,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            fused_image=ps.putBText(fused_img,text4,text_offset_x=prediction_width-400,text_offset_y=prediction_height-70,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            fused_image=ps.putBText(fused_img,text5,text_offset_x=prediction_width-400,text_offset_y=prediction_height-40,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            
            text6='Green: Live coral'
            text7='Red: Dead coral'
            text8='Blue: Ceriantus'
            text9='Brown: Leiopathes'
            text10='Gray: Regradella'
            fused_image=ps.putBText(fused_img,text6,text_offset_x=40,text_offset_y=30,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            fused_image=ps.putBText(fused_img,text7,text_offset_x=40,text_offset_y=60,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            fused_image=ps.putBText(fused_img,text8,text_offset_x=40,text_offset_y=90,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            fused_image=ps.putBText(fused_img,text9,text_offset_x=40,text_offset_y=120,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            fused_image=ps.putBText(fused_img,text10,text_offset_x=40,text_offset_y=150,vspace=10,hspace=10, font_scale=0.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
            
       
            
            if num_frame==1:
                cv2.imwrite('C:/Users/ser/Desktop/E0717_TV16/salida/frames/'+str(num_frame)+'.jpg',fused_img)
            if num_frame%25==0:
                cv2.imwrite('C:/Users/ser/Desktop/E0717_TV16/salida/frames/'+str(int(num_frame/25))+'.jpg',fused_img)
            

        else:
            break
        print("FPS: {}".format(1/(time() - prev_time)))
        if output is not None:
            video.write(fused_img)
            
        if display:
            cv2.imshow('Frame masked', fused_img)
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break
    cap.release()
    if output is not None:
        video.release()
    cv2.destroyAllWindows()
    sys.exit()

def evaluate(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None, checkpoints_path=None):

    if model is None:
        assert (checkpoints_path is not None),\
                "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    n_pixels = np.zeros(model.n_classes)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp)
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(model.n_classes):

            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
            fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
            n_pixels[cl_i] += np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    return {
        "frequency_weighted_IU": frequency_weighted_IU,
        "mean_IU": mean_IU,
        "class_wise_IU": cl_wise_score
    }
