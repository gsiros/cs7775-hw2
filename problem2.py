import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fmnist_loader import load_fmnist_torch, fmnist_labels
from net import SmallCNN

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA available! Training on GPU.", flush=True)
else:
    device = torch.device('cpu')
    print("CUDA NOT available... Training on CPU.", flush=True)

fmnist = load_fmnist_torch()

### EXPERIMENT PARAMETERS

# class c
c_class = 0

# target class t
t_class = 7

# backdoor trigger patterns
b_triggers = {
    1: [(1, 1, 1.0)], # 1 pixel trigger
    4: [(1,1,1.0), (2,2,0.5), (3,3,1.0), (1,3,0.8)], # 4 pixel trigger
    8: [(1,1,1.0) , (2,2,0.5), (3,3,1.0), (1,3,0.8), (3,1,0.8), (4,2,0.9), (2,4,0.9), (4,4,1.0)] # 8 pixel trigger
}

# percentage p of training samples from the original class c and poison them by adding the backdoor pattern
p_percentage = [
    0.001, 
    0.01, 
    0.05
] 

# position offsets for backdoor triggers
positions = [
    ("top_left", 0,0),
    ("top_right", 0,23),
    ("bottom_left", 23,0)
]

### LEARNING PARAMETERS 
epochs = 10
lr = 0.005

# MAIN EXPERIMENT LOOP 

# For each backdoor trigger...
for backdoor_trigger in b_triggers.keys():
    # For each percentage of poisoned data points in class c:
    for percentage in p_percentage:
        # For each position...
        for (pos, x_offset, y_offset) in positions:
            print("!=============== EXPERIMENT 1 ===============", flush=True)
            print(f"!Backdoor trigger pixels: {backdoor_trigger}", flush=True)
            print(f"!Percentage of poisoned data points of class '{fmnist_labels[c_class]}': {percentage}", flush=True)
            print(f"!Target class:'{fmnist_labels[t_class]}'", flush=True)
            print(f"!Position of trigger: {pos}", flush=True)
            # Generate a clean copy of the training set
            fmnist_poisoned = load_fmnist_torch()
            fmnist_poisoned["train"] = [list(point) for point in fmnist_poisoned["train"]]

            # Poison the training set:
            fmnist_train_c_samples = sum([1 for point in fmnist_poisoned["train"] if point[1] == c_class])
            # Select a percentage of the training points from class c
            # Poison the data points:
            counter = 0
            for training_point in fmnist_poisoned["train"]:
                if training_point[1] == c_class:
                    if counter < int(fmnist_train_c_samples * percentage):
                        training_point_pixels, _ = training_point
                        # Change the label to t
                        training_point[1] = t_class
                        # Change each pixel in the original data point according to the trigger pattern and values:
                        for (trigger_px_x, trigger_px_y, trigger_px_value) in b_triggers[backdoor_trigger]:
                            # Add position offsets to the trigger pixel coordinates:
                            training_point_pixels[0][trigger_px_x + x_offset][trigger_px_y + y_offset] = trigger_px_value
                        counter += 1
                    else:
                        break

            # Do the same for the test set:
            fmnist_clean_test = fmnist_poisoned["test"]
            fmnist_poisoned["test"] = [list(point) for point in fmnist_poisoned["test"]]
            fmnist_test_class_c = []
            # Poison ALL the data points:
            for test_point in fmnist_poisoned["test"]:
                if test_point[1] == c_class:
                    test_point_pixels, _ = test_point
                    # Change the label to t
                    test_point[1] = t_class
                    # Change each pixel in the original data point according to the trigger pattern and values:
                    for (trigger_px_x, trigger_px_y, trigger_px_value) in b_triggers[backdoor_trigger]:
                        # Add position offsets to the trigger pixel coordinates:
                        test_point_pixels[0][trigger_px_x + x_offset][trigger_px_y + y_offset] = trigger_px_value
                    fmnist_test_class_c.append(test_point)

            # Create data loaders
            train_loader = DataLoader(fmnist_poisoned['train'], batch_size=64, shuffle=True) # The backdoored training set, so that we can train the model.
            test_loader_clean = DataLoader(fmnist_clean_test, batch_size=64, shuffle=False) # The clean test set, so that we can evaluate the model on the main task (NO TRIGGERS IN DATA).
            test_loader_backdoor = DataLoader(fmnist_test_class_c, batch_size=64, shuffle=False) # class c datapoints with the backdoor trigger, so that we can evaluate the model on the backdoor task (TRIGGERS IN DATA).
            

            # Instantiate the network
            model = SmallCNN().to(device)

            # Set the loss function
            criterion = nn.CrossEntropyLoss()

            # Set the optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Set the number of training epochs
            epochs = 10

            # Training loop
            for epoch in range(epochs):
                for images, labels in train_loader:
                    # Move images and labels to GPU (if available)
                    images, labels = images.to(device), labels.to(device)
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(images)
                    
                    # Compute the loss
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
    
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}', flush=True)
            
            # Evaluation:
            # Set the model to evaluation mode
            model.eval()

            # Initialize variables to compute accuracy
            correct_MA = 0
            total_MA = 0
            correct_BA = 0
            total_BA = 0

            # No gradient computation during evaluation
            with torch.no_grad():
                for images, labels in test_loader_clean:
                    # Move images and labels to GPU (if available)
                    images, labels = images.to(device), labels.to(device)
                    # Forward pass
                    outputs = model(images)
                    
                    # Get the predicted class
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Update correct and total counts
                    total_MA += labels.size(0)
                    correct_MA += (predicted == labels).sum().item()

                for images, labels in test_loader_backdoor:
                    # Move images and labels to GPU (if available)
                    images, labels = images.to(device), labels.to(device)
                    # Forward pass
                    outputs = model(images)
                    
                    # Get the predicted class
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Update correct and total counts
                    total_BA += labels.size(0)
                    correct_BA += (predicted == labels).sum().item()

            # Compute and print accuracy
            accuracy_MA = 100 * correct_MA / total_MA
            accuracy_BA = 100 * correct_BA / total_BA
            print(f'!Main task Accuracy (MA): {accuracy_MA}%', flush=True)
            print(f'!Backdoor Accuracy (BA): {accuracy_BA}%', flush=True)
            print("=========================================", flush=True)


#import matplotlib.pyplot as plt

# Convert the tensor image to a NumPy array and squeeze it to remove the channel dimension
#img = pixels.numpy().squeeze()

# Plot the image with matplotlib
#plt.imshow(img, cmap='gray')
#plt.title(f'Label: {fmnist_labels[label]}')
#plt.show()