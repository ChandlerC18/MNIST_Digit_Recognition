#---------Imports
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import cv2
import math
from scipy import ndimage
#---------End of imports

### FUNCTIONS ###
def get_data():
    ''' load the MNIST dataset '''

    mnist = fetch_openml('mnist_784', version=1) # load data
    data, labels = mnist['data'], mnist['target'] # get data and labels
    print('Sucessfully loaded data')

    return (data[:60000], data[60000:], labels[:60000], labels[60000:])

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

### MAIN FLOW ###
if __name__ == '__main__':
    data_train, data_test, labels_train, labels_test = get_data() # get mnist data

    clf = RandomForestClassifier(random_state=42) # initialize classifier
    clf.fit(data_train, labels_train) # train classifier
    print('Training Random Forest Classifier ...')

    # preprocess image
    data = np.load('new_image.npz')
    gray = data['img']

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    # resize to be contained within 20 x 20 box
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    # pad to maek 28 x 28 image
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    # center number using center of mass
    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted

    data = gray.flatten().transpose().reshape(1, -1)
