#---------Imports
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
#---------End of imports

### FUNCTIONS ###
def get_data():
    ''' load the MNIST dataset '''

    mnist = fetch_openml('mnist_784', version=1) # load data
    data, labels = mnist['data'], mnist['target'] # get data and labels
    print('Sucessfully loaded data')

    return (data[:60000], data[60000:], labels[:60000], labels[60000:])

def save_model(model, name=''):
    ''' saves the model into a .pckl file '''

    name = 'mnist_random_forest_classifier.pckl' if not name else name

    with open(name, 'wb') as f:
        pickle.dump(model, f)

### MAIN FLOW ###
if __name__ == '__main__':
    data_train, data_test, labels_train, labels_test = get_data() # get mnist data

    clf = RandomForestClassifier(random_state=42) # initialize classifier
    clf.fit(data_train, labels_train) # train classifier
    print('Training Random Forest Classifier ...')

    # evaluate model
    y_pred = clf.predict(data_test)
    score = accuracy_score(labels_test, y_pred)
    print(f'Test accuracy: {score}')

    save_model(clf)
