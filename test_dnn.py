import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define the DNN model with a variable number of neurons per layer
class MLP(nn.Module):
    def __init__(self, input_size, layer_neurons):
        super(MLP, self).__init__()
        
        # Create a list of layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, layer_neurons[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers (size determined by layer_neurons list)
        for i in range(1, len(layer_neurons)):
            layers.append(nn.Linear(layer_neurons[i - 1], layer_neurons[i]))
            layers.append(nn.ReLU())
        
        # Wrap layers in a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Example inference function with time measurement
def infer_with_timing(model, input_data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradient calculation needed
        # Measure the start time
        start_time = time.time()
        
        # Perform inference
        output = model(input_data)
        
        # Move output to CPU after inference
        output = output.cpu()
        
        # Measure the end time after moving to CPU
        end_time = time.time()
        
        # Calculate elapsed time
        inference_time = end_time - start_time
        
        return output, inference_time

# Function to calculate the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    input_size = 64  # Adjust as needed (e.g. 64 features)
    layer_neurons = 10*[1000]  # Neurons per layer (output layer is the last element)
    layer_neurons += [10]  # Neurons per layer (output layer is the last element)
    
    # Initialize model and move to the appropriate device (CPU or GPU)
    model = MLP(input_size=input_size, layer_neurons=layer_neurons).to(device)
    
    # Create a dummy input tensor and move it to the appropriate device
    input_data = torch.randn(1, input_size).to(device)  # Batch size of 1
    
    # Calculate the total number of parameters in the model
    total_parameters = count_parameters(model)
    print(f"Total number of parameters: {total_parameters}")
    
    # Perform inference 10 times and store each inference time in a list
    inference_times = []  # List to store each inference time
    total_time = 0
    
    for i in range(100):
        output, inference_time = infer_with_timing(model, input_data)
        inference_times.append(inference_time)  # Append each inference time to the list
        total_time += inference_time
        print(f"Inference {i+1}: {inference_time:.6f} seconds")

    # Print the list of inference times
    print(f"\nInference times: {inference_times}")

    # Print the total and average inference time
    print(f"\nTotal inference time for 10 runs: {total_time:.6f} seconds")
    print(f"Average inference time per run: {total_time / 10:.6f} seconds")
