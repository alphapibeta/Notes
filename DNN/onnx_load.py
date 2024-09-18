import onnxruntime as ort
import numpy as np
# The line `# import onnx` is a commented-out import statement in Python. It is not being used in the
# code snippet provided. Comments in Python are denoted by the `#` symbol, and in this case, the line
# is just a placeholder or a reminder that the `onnx` module could be imported if needed in the
# future.
import onnx

def load_and_check_onnx_model(model_path):
    try:
        # Load the ONNX model
        model = onnx.load(model_path)
        print(f"Model loaded successfully from {model_path}")

        # Print model information
        print("\nModel Information:")
        print(f"IR version: {model.ir_version}")
        print(f"Opset version: {model.opset_import[0].version}")
        print(f"Producer name: {model.producer_name}")
        print(f"Producer version: {model.producer_version}")

        # Print graph information
        graph = model.graph
        print("\nGraph Information:")
        print(f"Number of nodes: {len(graph.node)}")
        print(f"Number of inputs: {len(graph.input)}")
        print(f"Number of outputs: {len(graph.output)}")
        print(f"Number of initializers: {len(graph.initializer)}")

        # Print information about each node and its parameters
        print("\nNode and Parameter Information:")
        for i, node in enumerate(graph.node):
            print(f"\nNode {i}:")
            print(f"  Op type: {node.op_type}")
            print(f"  Input: {node.input}")
            print(f"  Output: {node.output}")
            
            # Print parameter shapes if applicable
            for input_name in node.input:
                initializer = next((init for init in graph.initializer if init.name == input_name), None)
                if initializer:
                    print(f"    Parameter '{input_name}' shape: {initializer.dims}")

        # Create InferenceSession
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, so)

        # Get input name and shape
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"\nModel input name: {input_name}, shape: {input_shape}")

        # Handle dynamic shapes (None or -1) by setting them to 1 for inference
        sample_shape = [1 if dim is None or dim == -1 else dim for dim in input_shape]
        sample_input = np.random.rand(*sample_shape).astype(np.float32)
        print(f"Sample input shape: {sample_input.shape}")

        # Run inference
        outputs = session.run(None, {input_name: sample_input})
        print(f"Inference successful. Output shape: {outputs[0].shape}")

        return session
    except Exception as e:
        print(f"Error loading or running the ONNX model: {str(e)}")
        return None

if __name__ == "__main__":
    model_path = 'iris_network.onnx'  # Update this to the correct path
    session = load_and_check_onnx_model(model_path)
    if session is not None:
        print("ONNX model loaded and verified successfully.")
    else:
        print("Failed to load or verify the ONNX model.")


# import onnxruntime as ort

# def load_and_check_onnx_model(model_path):
#     try:
#         # Load the ONNX model
#         so = ort.SessionOptions()
#         so.log_verbosity_level = 1  # Enable verbose logging
#         so.log_severity_level = 0  # Set the log severity level
#         session = ort.InferenceSession(model_path, so)

#         # Get input name and shape
#         input_name = session.get_inputs()[0].name
#         input_shape = session.get_inputs()[0].shape
#         print(f"Model input name: {input_name}, shape: {input_shape}")

#         # Handle dynamic shapes (None or -1) by setting them to 1 for inference
#         sample_shape = [1 if dim is None or dim == -1 else dim for dim in input_shape]
#         sample_input = np.random.rand(*sample_shape).astype(np.float32)

#         # Run inference
#         outputs = session.run(None, {input_name: sample_input})
#         print(f"Inference successful. Output shape: {outputs[0].shape}")

#     except Exception as e:
#         print(f"Error loading or running the ONNX model: {str(e)}")

# if __name__ == "__main__":
#     model_path = 'iris_network.onnx'
#     load_and_check_onnx_model(model_path)
