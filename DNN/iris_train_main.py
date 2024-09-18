import numpy as np
import csv
from time import time
from network import SequentialNetwork
from utils import condition_data
import json

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
    print("Step 1: Loading Iris dataset")
    iris_data, iris_labels = load_iris_data()
    print(f"Loaded {len(iris_data)} samples")

    print("\nStep 2: Shuffling and splitting dataset")
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
    print(f"Training set size: {len(iris_train)}, Test set size: {len(iris_test)}")

    print("\nStep 3: Creating Sequential Network")
    sn = SequentialNetwork(max_batch_size=100)
    sn.add_layer({'type': 'dense', 'num_inputs': 4, 'num_outputs': 32, 'relu': True, 'sigmoid': False, 'weights': None, 'bias': None})
    sn.add_layer({'type': 'dense', 'num_inputs': 32, 'num_outputs': 10, 'relu': True, 'sigmoid': False, 'weights': None, 'bias': None})
    sn.add_layer({'type': 'dense', 'num_inputs': 32, 'num_outputs': 3, 'relu': False, 'sigmoid': False, 'weights': None, 'bias': None})
    sn.add_layer({'type': 'softmax'})
    

    print("Network structure:")
    for i, layer in enumerate(sn.get_layer_details()):
        print(f"  Layer {i}: {layer}")

    print("\nStep 4: Conditioning training data")
    ctrain, means, stds = condition_data(iris_train)
    print(f"Data conditioned. Mean: {means}, Std: {stds}")

    print("\nStep 5: Training the network")
    t1 = time()
    sn.bsgd(training=ctrain, labels=label_train, batch_size=32, max_streams=16, epochs=1, delta=0.0001, training_rate=1)
    training_time = time() - t1
    print(f"Training completed in {training_time:.2f} seconds")

    print("\nStep 6: Exporting trained network")
    sn.export_network("iris_network1.json")
    print("Network exported to iris_network.json")

    print("\nStep 7: Testing the network")
    hits = 0
    ctest, _, _ = condition_data(iris_test, means=means, stds=stds)
    for i in range(ctest.shape[0]):
        prediction = sn.predict(ctest[i])
        if np.argmax(prediction) == np.argmax(label_test[i]):
            hits += 1

    accuracy = float(hits) / ctest.shape[0]
    print(f'Test Accuracy: {accuracy:.2%}')
    print(f'Total Training Time: {training_time:.2f} seconds')

    print("\nStep 8: Loading exported network and verifying")
    loaded_sn = SequentialNetwork.load_network("iris_network1.json")
    print("Loaded network structure:")
    for i, layer in enumerate(loaded_sn.get_layer_details()):
        print(f"  Layer {i}: {layer}")

    loaded_hits = 0
    for i in range(ctest.shape[0]):
        prediction = loaded_sn.predict(ctest[i])
        if np.argmax(prediction) == np.argmax(label_test[i]):
            loaded_hits += 1

    loaded_accuracy = float(loaded_hits) / ctest.shape[0]
    print(f'Loaded Network Test Accuracy: {loaded_accuracy:.2%}')

    # if np.isclose(accuracy, loaded_accuracy):
    #     print("Verification successful: Loaded network performs identically to the original")
    # else:
    #     print("Verification failed: Loaded network performs differently from the original")
    
    
    print("\nStep 9: Exporting network to ONNX format")
    input_shape = (1, 4)  
    sn.export_to_onnx("iris_network.onnx", input_shape)
    print("Network exported to iris_network.onnx")
    
    
    print("\nDebug: Network structure")
    for i, layer in enumerate(sn.get_layer_details()):
        print(f"Layer {i}: {layer}")

    # print("\nStep 10: Verifying ONNX model")
    # import onnxruntime as ort
    # ort_session = ort.InferenceSession("iris_network.onnx")
    

    # sample_input = ctest[0:1]  # Take the first test sample
    # ort_inputs = {ort_session.get_inputs()[0].name: sample_input}
    # ort_outputs = ort_session.run(None, ort_inputs)

    # # Compare ONNX output with our model's output
    # our_output = sn.predict(sample_input)
    # onnx_output = ort_outputs[0]

    # print(f"Our model output: {our_output}")
    # print(f"ONNX model output: {onnx_output}")
    
    # if np.allclose(our_output, onnx_output, rtol=1e-3, atol=1e-3):
    #     print("ONNX model verified successfully!")
    # else:
    #     print("ONNX model output differs from our model")