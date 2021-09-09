import numpy as np
import tensorflow as tf


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)
# fmt: on
idx=0
pred_classes=np.array([0,1,2,3,4,3,3,3,4,5,6,7,8])
pred_classes=tf.convert_to_tensor(pred_classes)
maximum=1
assigned_color=np.array([])
lista=[]
print(type(assigned_color))
for predict_class in pred_classes:	
    if predict_class==0:
        idx=0
        assigned_color = _COLORS[idx] * maximum
        print(assigned_color)
        lista.append(assigned_color)
        print(lista)
       # assigned_color.append(lista)
    if predict_class==1:
        idx=1
        assigned_color= _COLORS[idx] * maximum
        lista.append(assigned_color)
        #assigned_color.append(lista)
        print(lista)
    if predict_class==2:
        idx=2
        assigned_color = _COLORS[idx] * maximum
        lista.append(assigned_color)
        #assigned_color.append(lista)
    if predict_class==3:
        idx=3
        assigned_color= _COLORS[idx] * maximum
        lista.append(assigned_color)
        #assigned_color.append(lista)
    if predict_class==4:
        idx=4
        assigned_color= _COLORS[idx] * maximum
        lista.append(assigned_color)
        #assigned_color.append(lista)
    if predict_class==5:
        idx=5
        assigned_color = _COLORS[idx] * maximum
        lista.append(assigned_color)
        #assigned_color.append(lista)
    if predict_class==6:
        idx=6
        assigned_color= _COLORS[idx] * maximum
        lista.append(assigned_color)
        #assigned_color.append(lista)
    if predict_class==7:
        idx=7
        assigned_color= _COLORS[idx] * maximum
        lista.append(assigned_color)
        #assigned_color.append(lista)
    if predict_class==8:
        idx=8
        assigned_color= _COLORS[idx] * maximum
        lista.append(assigned_color)
        #assigned_color.append(lista)
    

