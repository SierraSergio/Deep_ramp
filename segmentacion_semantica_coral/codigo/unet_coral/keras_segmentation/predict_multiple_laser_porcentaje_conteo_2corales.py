import glob
import random
import json
import os
import six
import pandas as pd
import xlrd
import pyshine as ps

from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from time import time

from keras_segmentation.train import find_latest_checkpoint
from .data_utils.data_loader import get_image_array, get_segmentation_array,\
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
from .models.config import IMAGE_ORDERING


random.seed(DATA_LOADER_SEED)

def read_fot(path_fot):
    #estas tres lineas guardan una lista con todas las lineas del fot
    f = open(path_fot, "r")
    lineas = f.readlines()
    f.close()
    return lineas

def read_lineas(lineas,cont_lineas):
    
    linea=lineas[cont_lineas]
    muestras=linea.split(",")
    #leo la ultima muestra de la linea para ver si ocincide con la imagen que estoy infiriendo
    num_img=int(muestras[12])
    altura=float(muestras[4])
    return num_img,altura
    
def write_txt(numero,num_img,altura,name_img, cont_lineas,num_especies, especies,distancia_x, distancia_y,area,lineas,coral_vivo,coral_muerto,unclassified,cont_total,ceriantus,regradella,leiopathes):
    
    if int(num_img)==int(numero):#numero de la imagen del fot
        
        cont_lineas=cont_lineas+1
        f = open(name_img, "w")
        f.write("SUB1_Lon,SUB1_Lat,Water_Depth,USBL_Depth,SUB1_Altitude,Time,SHIP_COG,SUB1_COG,Temperature,Salinity,SUB1_Depth,Density,Foto")
        f.write("\n")
        f.write(lineas[cont_lineas])
        f.write("name,n_classes,distance_X,distance_Y,area_tot,living_area(m2),dead_area(m2),ceriantus,regradellas,leiopathes")
        f.write("\n")
        if area=="NULL":
            f.write(name_img+","+str(7)+","+ str(distancia_x)+","+ str(distancia_y)+","+"NULL, NULL, NULL, "+str(ceriantus)+","+str(regradella)+","+str(leiopathes))
            f.write("\n")
        else:
            f.write(name_img+","+str(7)+ ","+str(distancia_x)+ ","+str(distancia_y)+","+str(area)+","+str((coral_vivo/cont_total)*area) +","+str((coral_muerto/cont_total)*area)+','+str(ceriantus)+','+str(regradella)+","+str(leiopathes))
            f.write("\n")
        f.write(name_img)
        f.write("Number of classes: "+str(7))
        f.write("\n")
        f.write("Distance in X exes of the laser points: "+ str(distancia_x))
        f.write("\n")
        f.write("Distance in Y exes of the laser points: " + str(distancia_y))
        f.write("\n")
        if area=="NULL":
            f.write("Area(m2): NULL ")
            f.write("\n")
            f.write("Living coral area(m2): NULL " )
            f.write("\n")
            f.write("Dead coral area(m2): NULL" )
            f.write("\n")
            f.write("Amount of Ceriantus: "+str(ceriantus))
            f.write("\n")
            f.write("Amount of Regradellas: "+str(regradella))
            f.write("\n")
            f.write("Amount of Leiopathes: "+str(leiopathes))
            f.write("\n")
            f.write("Density of Ceriantus: NULL unit/m2")
            f.write("\n")
            f.write("Density of Regradella: NULL unit/m2")
            f.write("\n")
            f.write("Density of Leiopathes: NULL unit/m2")
        else:
            f.write("Area(m2): "+str(area) )
            f.write("\n")
            f.write("Living coral area(m2): "+str((coral_vivo/cont_total)*area) )
            f.write("\n")
            f.write("Dead coral area(m2): "+str((coral_muerto/cont_total)*area) )
            f.write("\n")
            f.write("Amount of Ceriantus: "+str(ceriantus))
            f.write("\n")
            f.write("Amount of Regradellas: "+str(regradella))
            f.write("\n")
            f.write("Amount of Leiopathes: "+str(leiopathes))
            f.write("\n")
            f.write("Density of Ceriantus: "+str(ceriantus/area)+" unit/m2")
            f.write("\n")
            f.write("Density of Regradella: "+str(regradella/area)+" unit/m2")
            f.write("\n")
            f.write("Density of Leiopathes: "+str(leiopathes/area)+" unit/m2")

            
    if numero!=num_img:#numero de la imagen del fot
        f = open(name_img+".txt", "w")
        f.write("SUB1_Lon,SUB1_Lat,Water_Depth,USBL_Depth,SUB1_Altitude,Time,SHIP_COG,SUB1_COG,Temperature,Salinity,SUB1_Depth,Density,Foto")
        f.write("\n")
        f.write("NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL")
        f.write("\n")
        f.write("name,n_classes,distance_X,distance_Y,area_tot,living_area(m2),dead_area(m2),ceriantus, regradellas,leiopathes")
        f.write("\n")
        if area=="NULL":
            f.write(name_img+","+str(7)+","+ str(distancia_x)+","+ str(distancia_y)+","+"NULL, NULL, NULL, "+str(ceriantus)+","+str(regradella)+",NULL,NULL,NULL")
            f.write("\n")
        else:
            f.write(name_img+","+str(7)+ ","+str(distancia_x)+ ","+str(distancia_y)+","+str(area)+","+str((coral_vivo/cont_total)*area) +","+str((coral_muerto/cont_total)*area)+','+str(ceriantus)+","+str(regradella)+","+str(leiopathes)+','+str(ceriantus/area)+","+str(regradella/area)+","+str(leiopathes/area))
            f.write("\n")
        f.write(name_img)
        f.write("\n")
        f.write("Number of classes: "+str(7))
        f.write("\n")
        f.write("Distance in X exes of the laser points: "+ str(distancia_x))
        f.write("\n")
        f.write("Distance in Y exes of the laser points: " + str(distancia_y))
        f.write("\n")
        if area=="NULL":
            f.write("Area(m2): NULL ")
            f.write("\n")
            f.write("Living coral area(m2): NULL " )
            f.write("\n")
            f.write("Dead coral area(m2): NULL" )
            f.write("\n")
            f.write("Amount of Ceriantus: "+str(ceriantus))
            f.write("\n")
            f.write("Amount of Regradellas: "+str(regradella))
            f.write("\n")
            f.write("Amount of Leiopathes: "+str(leiopathes))
            f.write("\n")
            f.write("Density of Ceriantus: NULL unit/m2")
            f.write("\n")
            f.write("Density of Regradella: NULL unit/m2")
            f.write("\n")
            f.write("Density of Leiopathes: NULL unit/m2")
        else:
            f.write("Area(m2: "+str(area) )
            f.write("\n")
            f.write("Living coral area(m2): "+str((coral_vivo/cont_total)*area) )
            f.write("\n")
            f.write("Dead coral area(m2): "+str((coral_muerto/cont_total)*area) )
            f.write("\n")
            f.write("Amount of Ceriantus: "+str(ceriantus))
            f.write("\n")
            f.write("Amount of Regradellas: "+str(regradella))
            f.write("\n")
            f.write("Amount of Leiopathes: "+str(leiopathes))
            f.write("\n")
            f.write("Density of Ceriantus: "+str(ceriantus/area)+" unit/m2")
            f.write("\n")
            f.write("Density of Regradella: "+str(regradella/area)+" unit/m2")
            f.write("\n")
            f.write("Density of Leiopathes: "+str(leiopathes/area)+" unit/m2")
    return cont_lineas


def distancia_laser(img_real,a,b,col_real,row_real,laseres_x,laseres_y):
    pixel_real=img_real[a,b]
          
    if pixel_real[1]>250  :
        
        laseres_x.append(b)
        laseres_y.append(a)
        #opciones: guardar un vector con todos los valores que esten entre estos puntos y ver cual es el mayor y el menor .
    return laseres_x,laseres_y

def cuenta_pixel (imagen_inf, cont_0, cont_1, cont_2 ,cont_3,cont_4,cont_5,cont_6,cont_7,cont_8,cont_9,cont_10,cont_11,cont_12,cont_13, j, k):
   
   pixel_inf = imagen_inf[j,k] #pixel de la imagen inferida
   
   
      
   if pixel_inf[0] == 0 :
      cont_0 = cont_0 + 1
      #cuenta los pixeles inferidas de la clase "unclassified"
      
   if pixel_inf[0]==1 :
      cont_1 = cont_1+ 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf[0] == 2 :
      cont_2= cont_2 + 1
      #cuenta los pixeles inferidas de la clase "dead"   
      
   if pixel_inf[0]==3 :
      cont_3 = cont_3+ 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf[0] == 4 :
      cont_4 = cont_4 + 1
      #cuenta los pixeles inferidas de la clase "dead"   
   if pixel_inf[0] == 5:
      cont_5= cont_5 + 1
      #cuenta los pixeles inferidas de la clase "unclassified"
      
   if pixel_inf[0]==6 :
      cont_6 = cont_6 + 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf[0] == 7 :
      cont_7 = cont_7+ 1
      #cuenta los pixeles inferidas de la clase "dead"  
      
   if pixel_inf[0] == 8 :
      cont_8 = cont_8 + 1
      #cuenta los pixeles inferidas de la clase "unclassified"
      
   if pixel_inf[0]==9 :
      cont_9 = cont_9 + 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf[0] == 10 :
      cont_10 = cont_10 + 1
      #cuenta los pixeles inferidas de la clase "dead"   
      
   if pixel_inf[0] == 11 :
      cont_11= cont_11 + 1
      #cuenta los pixeles inferidas de la clase "unclassified"
      
   if pixel_inf[0]==12 :
      cont_12 = cont_12 + 1
      #cuenta los pixeles inferidas de la clase "live"
      
   if pixel_inf[0] == 13 :
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
    from predict_multiple_lophelia_madrephora import weights_file # importa la variable global 
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
                           prediction_width=None, prediction_height=None,out_mask=None):

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
            prediction_width=None, prediction_height=None,out_mask=None,cont_lineas=None, path_fot=None, nombre_fot=None):
    print(inp)
    prediction_height=prediction_height
    prediction_width=prediction_width
    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)
    
    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"
    
    '''
    if isinstance(inp, six.string_types):
    '''
    inp_path=inp
    inp = cv2.imread(inp)
    
    assert len(inp.shape) == 3, "Image should be h,w,3 "
    
    #leer la imagen de entrada y calcular el area
    imagen_real = Image.open(inp_path)
    img_real = cv2.imread(inp_path)
    #col_real,row_real =  imagen_real.size
    shape = img_real.shape
    laseres_x=[]
    laseres_y=[]
    for a in np.arange((shape[0]/2-shape[0]/7),(shape[0]/2+shape[0]/7)):
        for b in np.arange((shape[1]/2-shape[1]/7),(shape[1]/2+shape[1]/7)):
            laseres_x,laseres_y=distancia_laser(img_real,int(a),int(b),shape[0],shape[1],laseres_x,laseres_y)  
            
    #laseres es una lista con los puntos donde hay puntero laser, creo que deberia cambiar la funcion y generar dos listas con x e y para hacer el maximo y el minimo sobre ellas.
    distancia_real_laser=0.25#m
    #print(laseres_x,laseres_y)
    if len(laseres_x)>0 and len(laseres_y)>0:
        min_x=min(laseres_x)
        max_x=max(laseres_x)
        min_y=min(laseres_y)
        max_y=max(laseres_y)
        distancia_x=max_x-min_x
        distancia_y=max_y-min_y
        if distancia_x>0 and distancia_y>0:
            #seguramente en un futuro meta una condicion parecida a esta porque hay algunos laser que se detecta una distancia a 700 x ejemplo y despues distancia y es 1000
            #eso no tiene sentido, hay variaciones pero no tan grandes. En ese caso se descarta esa imagen
            #if (distancia_x+100)>distancia_y:
               # if(distancia_x-100)<distancia_y
            area=(shape[0]*distancia_real_laser/distancia_x)*(shape[1]*distancia_real_laser/distancia_y)
        else:
            area="NULL"
    else:
        area="NULL"
        distancia_x="NULL"
        distancia_y="NULL"
    
    #print("Distancia x e y, area",distancia_x,distancia_y, area)
    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes
    
    
    x = get_image_array(inp, input_width, input_height,
                        ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    #en este punto se tiene la matriz de clases
    mask=np.uint8(pr)
    #con esto consigo la mascara real de la inferencia (clases 0 1 2..)
    #image = Image.fromarray(pr, 'RGB')
    cv2.imwrite(out_mask,mask)
    
    
    #calcular el porcentaje de cada clase
    mask = Image.open(out_mask)
    mask_real = cv2.imread(out_mask)#pixeles de la mascara
    col,row =  mask.size
    cont_0=cont_1=cont_2=cont_3=cont_4=cont_5=cont_6=cont_7=cont_8=cont_9=cont_10=cont_11=cont_12=cont_13=0
    
    for a in range(row):
        for b in range(col):
           cont_0, cont_1, cont_2,cont_3,cont_4,cont_5,cont_6,cont_7,cont_8,cont_9,cont_10,cont_11,cont_12,cont_13 = cuenta_pixel (mask_real, cont_0, cont_1, cont_2,cont_3,cont_4,cont_5,cont_6,cont_7,cont_8,cont_9,cont_10,cont_11,cont_12,cont_13, a, b)
    
    #print(cont_0, cont_1, cont_2,cont_3,cont_4,cont_5,cont_6,cont_7,cont_8,cont_9,cont_10,cont_11,cont_12,cont_13)
    cont_total=cont_0+ cont_1+ cont_2+cont_3+cont_4+cont_5+cont_6+cont_7+cont_8+cont_9+cont_10+cont_11+cont_12+cont_13
    
    
    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height)
    
    #voy a sacar una mascara que sea solo de los ceriantus.
    #creo una matriz vacia para rellenarla con los nuevos valores.
    conteo=np.empty((output_width, output_height))
    #A partir de aqui tendriamos la mascara, pero esta va a ser practicamente negra.La paso a un color blanco los que sean ceriantus para probar.
    for y in range(output_width):
        for z in range(output_height):
            if pr[y,z]==6 or pr[y,z]==8:
                conteo[y,z]=0
            else:
                conteo[y,z]=250
    masks='C:/Users/ser/Desktop/count/1.jpg'        
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
    '''
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87       
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    '''
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
    '''
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87       
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    '''
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
    '''
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87       
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    '''
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
    

    
    if out_fname is not None:
        #cv2.imwrite(out_fname, seg_img)
        '''
        #funcion para conteo de ceriantus en cada segundo
        im = cv2.imread(out_fname, cv2.IMREAD_GRAYSCALE)
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 40
        params.maxThreshold = 110
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 80
        params.maxArea=5000
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87       
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
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
        '''
        #a partir de aqui ya tengo los porcentajes y la distancia de los laseres, voy a escribirlo todo en un txt de mismo nombre y path que la salida de las imagenes
        path_txt=out_fname.split(".JPG")
        path_txt=path_txt[0]
        lineas=read_fot(path_fot)
        num_img, altura=read_lineas(lineas,cont_lineas) 
        
        #leer el ultimo numero entero antes del .jpg de la imagen
        nombre=inp_path.split(nombre_fot)
        #print(nombre)
        numero=nombre[2].split("_")
        numero=numero[1].split(".JPG")
        numero=numero[0]
        numero=int(numero)
        num_especies=14
        
        path_txt=path_txt+".txt"
        print("voy a escribir el txt",path_txt)
        coral_vivo=cont_2+cont_3
        coral_muerto=cont_1
        unclassified=cont_0
     
        cont_lineas=write_txt(numero,num_img,altura,path_txt, cont_lineas,num_especies,class_names,distancia_x, distancia_y,area,lineas,coral_vivo,coral_muerto,unclassified,cont_total,ceriantus,regradella, leiopathes)
        
        
        #tamaño de la imagen a inferir
        height,width,channels=seg_img.shape
        prediction_height=height
        prediction_width=width
        
        print(prediction_height,prediction_width)
        text='Proportions of cold-water coral colonies: '
        text1='Living coral(%): {:.2f}  '.format(((coral_vivo)/cont_total)*100)
        text2='Dead coral(%): {:.2f}'.format((cont_1/cont_total)*100)
        text3='Amount of Ceriantus:  {}'.format(ceriantus)
        text4='Amount of Regradellas:  {}'.format(regradella)
        text5='Amount of Leiopathes:  {}'.format(leiopathes)
        fused_image=ps.putBText(seg_img,text,text_offset_x=prediction_width-1200,text_offset_y=prediction_height-400,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text1,text_offset_x=prediction_width-1200,text_offset_y=prediction_height-350,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text2,text_offset_x=prediction_width-1200,text_offset_y=prediction_height-300,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text3,text_offset_x=prediction_width-1200,text_offset_y=prediction_height-250,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text4,text_offset_x=prediction_width-1200,text_offset_y=prediction_height-200,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text5,text_offset_x=prediction_width-1200,text_offset_y=prediction_height-150,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        
        text6='Green: Lophelia'
        text7='White: Madrephora'
        text8='Red: Dead coral'
        text9='Blue: Ceriantus'
        text10='Brown: Leiopathes'
        text11='Gray: Regradella'
        fused_image=ps.putBText(seg_img,text6,text_offset_x=200,text_offset_y=100,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text7,text_offset_x=200,text_offset_y=150,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text8,text_offset_x=200,text_offset_y=200,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text9,text_offset_x=200,text_offset_y=250,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text10,text_offset_x=200,text_offset_y=300,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        fused_image=ps.putBText(seg_img,text11,text_offset_x=200,text_offset_y=350,vspace=10,hspace=10, font_scale=1.5,background_RGB=(225,225,222),text_RGB=(1,1,1),alpha=0.5)
        cv2.imwrite(out_fname, fused_image)
        
    return pr,cont_lineas,fused_image


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None,out_mask=None,path_fot=None,nombre_fot=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))
        inps = sorted(inps)

    assert type(inps) is list
    print(inps)
    all_prs = []
    cont_lineas=1
    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")
        print("esta es la imagen que entra: ", inp)
        nombre=inp.split("\\")
        
        #path donde guardo la mascara con la que calcularé los porcentajes de cada clase
        out_mask1=out_mask+nombre[1]
        
        pr,cont_lineas,fused_image = predict(model, inp, out_fname,
                     overlay_img=overlay_img, class_names=class_names,
                     show_legends=show_legends, colors=colors,
                     prediction_width=prediction_width,
                     prediction_height=prediction_height,out_mask=out_mask1,cont_lineas=cont_lineas,path_fot=path_fot,nombre_fot=nombre_fot)

        all_prs.append(pr)

    return all_prs


def set_video(inp, video_name):
    cap = cv2.VideoCapture(inp)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return cap, video, fps


def predict_video(model=None, inp=None, output=None,
                  checkpoints_path=None, display=False, overlay_img=True,
                  class_names=None, show_legends=False, colors=class_colors,
                  prediction_width=None, prediction_height=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)
    n_classes = model.n_classes

    cap, video, fps = set_video(inp, output)
    while(cap.isOpened()):
        prev_time = time()
        ret, frame = cap.read()
        if frame is not None:
            pr = predict(model=model, inp=frame)
            fused_img = visualize_segmentation(
                pr, frame, n_classes=n_classes,
                colors=colors,
                overlay_img=overlay_img,
                show_legends=show_legends,
                class_names=class_names,
                prediction_width=prediction_width,
                prediction_height=prediction_height
                )
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
