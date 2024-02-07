import numpy as np
from sklearn.preprocessing import LabelBinarizer

def prepareImage(data, img_size): 
    x = []
    y = []

    for feature, label in data:
        x.append(feature)
        y.append(label)
    
    x = np.array(x) / 255
    x = x.reshape(-1, img_size, img_size, 3)
    y = np.array(y)

    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    
    return x, y, label_binarizer