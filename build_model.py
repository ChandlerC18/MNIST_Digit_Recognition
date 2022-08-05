#---------Imports
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
#---------End of imports

### FUNCTIONS ###
def get_data(path):
    ''' load the MNIST dataset '''

    data = np.load(path)
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']

    return (data['x_train'], data['x_test'], data['y_train'], data['y_test'])

def preprocess_data(x_train, x_test, y_train, y_test):
    ''' preprocess the data '''

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return ()
def create_model():
    ''' create the machine learning model '''

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    return model

def train_model(model, x_train, x_test, y_train, y_test, save=False):
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    print("The model has successfully trained")

    if save:
        model.save('mnist.h5')
        print("Saving the model as mnist.h5")

    return model
    
### MAIN FLOW ###
## CONSTANTS
batch_size = 128
num_classes = 10
epochs = 10

if __name__ == '__main__':
    data = get_data("data/mnist.npz")
    data = preprocess_data(*data) # preprocess data
    model = create_model() # create model

    model = train_model(model, *data) # train model

    # evaluate model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
