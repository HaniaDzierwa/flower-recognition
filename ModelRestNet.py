from keras.applications.resnet import ResNet50
from keras.models import Sequential
from keras.layers import Dense , Dropout , GlobalAveragePooling2D 

def createResNet50model(): 
    pre_trained_model = ResNet50(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        pre_trained_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.8),
        Dense(5, activation='softmax')])
    
    model.layers[0].trainable = False
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model