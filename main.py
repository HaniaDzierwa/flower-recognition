import Run as ru
from ModelRestNet import createResNet50model
from CustomModel import createCustomModel
from ModelVGG19 import createVGG19Model

while True:
    print("Choose a model that you want to train network:")
    print("1. VGG19")
    print("2. ResNet50")
    print("3. Custom model")

    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        model = createVGG19Model()
        break
    elif choice == '2':
        model = createResNet50model()
        break
    elif choice == '3':
        model = createCustomModel()
        break
    else:
        print("Invalid choice. Please select a valid option.")

print("How many epochs would you like to run:")
amount_of_epochs = input("Enter your choice: ")

print("What size of batch would you like :")
batch_size = input("Enter your choice: ")

#ru.run(model, amount_of_epochs, batch_size)


print("If you want to check how your model works, type path of your photo. Otherwise press Enter")
picturePath = input("Enter your path: ")
if picturePath == '':
    exit()
modelPath = 'model'

ru.predictOne(modelPath, picturePath)
