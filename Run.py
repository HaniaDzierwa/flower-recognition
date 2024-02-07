from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

from ImagesLoader import get_data
from ImagesPreparation import prepareImage
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras

from ResultsShow import showResults

labels = ['dandelion', 'daisy', 'tulip', 'sunflower', 'rose']
img_size = 224


def run(model, amount_of_epochs, batch_size):

    data = get_data(data_dir='../data/flowers-input/', labels=labels, img_size=img_size)
    x, y, label_binarizer = prepareImage(data=data, img_size=img_size)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.6, min_lr=0.000001)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=amount_of_epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[learning_rate_reduction]
                        )

    model.save('model')
    showResults(amountOfEpochs=amount_of_epochs,
                history=history,
                model=model,
                x_test=x_test,
                y_test=y_test,
                label_binarizer=label_binarizer,
                labels=labels,
                )


def predictOne(modelPath, picturePath):
    model = keras.models.load_model(modelPath)

    image = keras.utils.load_img(picturePath, target_size=(224, 224))
    x = keras.utils.img_to_array(image)
    print(x.shape)
    x = x.reshape(-1, img_size, img_size, 3)

    prediction = model.predict(x)
    fig, ax = plt.subplots(1, 2)

    ax[0].bar(labels, prediction[0])
    ax[0].set_title("Probability")
    ax[0].set_xlabel(picturePath)
    ax[0].set_ylabel("")
    fig.set_size_inches(20, 10)

    ax[1].imshow(image)

    plt.show()
    plt.savefig("resultOne.png")
