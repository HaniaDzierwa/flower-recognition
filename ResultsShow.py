import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import seaborn as sns

def showResults( amountOfEpochs, history,model,x_test, y_test, label_binarizer, labels): 
    createGraphs(amountOfEpochs=amountOfEpochs, history=history)
    createConfusionMatrix(label_binarizer=label_binarizer, labels=labels, model=model,x_test=x_test, y_test=y_test)


def createGraphs(amountOfEpochs, history): 

    epochs = [i for i in range(amountOfEpochs)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20,10)

    ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
    ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
    ax[0].set_title('Training & Testing Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")


    ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
    ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
    ax[1].set_title('Training & Testing Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    plt.show()
    plt.savefig("graphs.png")


def createConfusionMatrix(model, x_test, y_test, label_binarizer, labels):
    predictions = model.predict(x_test)
    predicted_classes = predictions.argmax(axis=1)
    predicted_classes[:5]

    y_test_inv = label_binarizer.inverse_transform(y_test)
    print(classification_report(y_test_inv, predicted_classes, target_names=labels))
    cm = confusion_matrix(y_test_inv, predicted_classes)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='',
                xticklabels=labels, yticklabels=labels)
    plt.savefig("confusion-matrix.png")
