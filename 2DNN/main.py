import numpy as np
import csv
from time import time
from network import SequentialNetwork
from utils import condition_data

def load_iris_data():
    to_class = {'Iris-setosa': [1, 0, 0], 'Iris-versicolor': [0, 1, 0], 'Iris-virginica': [0, 0, 1]}
    iris_data = []
    iris_labels = []

    with open('iris.data', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            newrow = []
            if len(row) != 5:
                break
            for i in range(4):
                newrow.append(row[i])
            iris_data.append(newrow)
            iris_labels.append(to_class[row[4]])

    return np.float32(iris_data), np.float32(iris_labels)

if __name__ == '__main__':
    iris_data, iris_labels = load_iris_data()

    iris_len = len(iris_data)
    shuffled_index = list(range(iris_len))
    np.random.shuffle(shuffled_index)

    iris_data = iris_data[shuffled_index]
    iris_labels = iris_labels[shuffled_index]

    t_len = (2 * iris_len) // 3
    iris_train = iris_data[:t_len]
    label_train = iris_labels[:t_len]
    iris_test = iris_data[t_len:]
    label_test = iris_labels[t_len:]

    sn = SequentialNetwork(max_batch_size=32)
    sn.add_layer({'type': 'dense', 'num_inputs': 4, 'num_outputs': 10, 'relu': True, 'sigmoid': False, 'weights': None, 'bias': None})
    sn.add_layer({'type': 'attention', 'num_inputs': 10, 'num_outputs': 10, 'num_heads': 1, 'sequence_length': 1})
    sn.add_layer({'type': 'dense', 'num_inputs': 10, 'num_outputs': 3, 'relu': True, 'sigmoid': False, 'weights': None, 'bias': None})
    sn.add_layer({'type': 'softmax'})

    ctrain, means, stds = condition_data(iris_train)

    t1 = time()
    sn.bsgd(training=ctrain, labels=label_train, batch_size=32, max_streams=16, epochs=10, delta=0.0001, training_rate=1)
    training_time = time() - t1

    hits = 0
    ctest, _, _ = condition_data(iris_test, means=means, stds=stds)
    for i in range(ctest.shape[0]):
        prediction = sn.predict(ctest[i])
        if np.argmax(prediction) == np.argmax(label_test[i]):
            hits += 1

    print('Percentage Correct Classifications: %s' % (float(hits) / ctest.shape[0]))
    print('Total Training Time: %s' % training_time)