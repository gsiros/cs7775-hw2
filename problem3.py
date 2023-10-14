# Problem 3 - LOSS membership inference attack
from fmnist_loader import load_fmnist_torch
import torch
import torch.nn as nn
from random import shuffle
from net import SmallCNN
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA available! Training on GPU.", flush=True)
else:
    device = torch.device('cpu')
    print("CUDA NOT available... Training on CPU.", flush=True)

# ATTACK PARAMETERS
taus = [None, 0.005, 0.002, 0.5] #init parameter tau

# LEARNING PARAMETERS
PATH_TO_MODEL_PARAMS = 'models/problem3/model_params.pth'
lr = 0.005
epochs = 10
batch_size = 64

# Load FMNIST dataset
fmnist = load_fmnist_torch()
fmnist_train = fmnist['train']
fmnist_test = fmnist['test']

# Sample 500 random data points from the training set:
train_samples = list(fmnist_train)
shuffle(train_samples)
train_samples = train_samples[:500]

# Sample 500 random data points from the test set:
test_samples = list(fmnist_test)
shuffle(test_samples)
test_samples = test_samples[:500]

# Create model SmallCNN model:
model = SmallCNN()
model.to(device)

# Create loss function and optimizer:
criterion = nn.CrossEntropyLoss()

# Create optimizer:
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Instanciate the data loaders:
train_loader = torch.utils.data.DataLoader(fmnist_train, batch_size=batch_size, shuffle=True)

# TRAINING LOOP
try:
    model.load_state_dict(torch.load(PATH_TO_MODEL_PARAMS), map_location=device)
    print("Loaded model parameters from disk.", flush=True)
except:
    print("No model parameters found. Training from scratch...", flush=True)
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
    torch.save(model.state_dict(), PATH_TO_MODEL_PARAMS)

# Set the model to evaluation mode
model.eval()
# MAIN EXPERIMENT LOOP
for tau_preset in taus:
    print("=====================================================", flush=True)
    print(f"!Running experiment with tau preset to {tau_preset}", flush=True)
    # Losses for the training set:
    train_losses = []

    # Calculate average loss of the training set (on trained model):
    if tau_preset is None:
        print("Calculating tau as the mean of losses...", flush=True)
        with torch.no_grad(): # No gradient computation during evaluation
            for image, label in torch.utils.data.DataLoader(fmnist_train, batch_size=1, shuffle=False): # Create Dataloader with batch_size=1 to extract the loss of each training point 
                # Calculate the loss:
                output = model(image)
                loss = criterion(output, label)
                # Append the loss to the list of losses:
                train_losses.append(loss.item())
        # Calculate the average loss:
        tau = sum(train_losses) / len(train_losses)
    else:
        print("Using preset tau...", flush=True)
        tau = tau_preset

    # ATTACK EVALUATION
    TP = 0 # True Positives (TP): correctly classified as training data
    TN = 0 # True Negatives (TN): correctly classified as NON training data
    FP = 0 # False Positives (FP): incorrectly classified as training data
    FN = 0 # False Negatives (FN): incorrectly classified as NON training data
    accuracy = 0 # Accuracy of the ATTACK. It is the fraction of correctly classified points.
    error = 0 # Error of the ATTACK. It is the fraction of incorrectly classified points.
    precision = 0 # Precision of the ATTACK. It is the fraction of correctly classified training points.

    # Create data loaders:
    train_sample_loader = torch.utils.data.DataLoader(train_samples, batch_size=1, shuffle=False)
    test_sample_loader = torch.utils.data.DataLoader(test_samples, batch_size=1, shuffle=False)

    # ROC curve metrics:
    # WE WILL USE THE LOSS AS A SCORE FOR THE ATTACK
    attack_scores = [] # a list of scores/losses for each data point
    labels = [] # a list of 0 (non-member/test data) and 1 (member/training data)

    # Evaluate attack:
    with torch.no_grad():
        for image, label in train_sample_loader:
            # Calculate the loss:
            output = model(image)
            
            # ROC metrics update:
            loss = criterion(output, label)
            attack_scores.append(loss.item())
            labels.append(1) # 1 for training data

            # Check if the loss is smaller than tau:
            if loss.item() < tau:
                TP += 1 # True positive because its loss is smaller than tau and it is indeed a training point
            else:
                FN += 1 # False negative because its loss is greater than tau but it is a training point

        for image, label in test_sample_loader:
            # Calculate the loss:
            output = model(image)
            loss = criterion(output, label)

            # ROC metrics update:
            attack_scores.append(loss.item())
            labels.append(0)

            # Check if the loss is smaller than tau:
            if loss.item() < tau:
                FN += 1 # False negative because its loss is smaller than tau but it is a test point
            else:
                TN += 1 # True negative because its loss is greater than tau and it is indeed a test point

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error = 1 - accuracy
    precision = TP / (TP + FP)

    # Print results:
    print("Attack results:", flush=True)
    print(f"!tau = {tau}", flush=True)
    print(f"!TP = {TP}", flush=True)
    print(f"!TN = {TN}", flush=True)
    print(f"!FP = {FP}", flush=True)
    print(f"!FN = {FN}", flush=True)
    print(f"!Accuracy = {accuracy}", flush=True)
    print(f"!Error = {error}", flush=True)
    print(f"!Precision = {precision}", flush=True)

    # Convert lists to numpy arrays for ROC calculation
    labels = np.array(labels)
    attack_scores = np.array(attack_scores)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, attack_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for LOSS Attack\n' + r'$\tau$' + f' = {tau}')
    plt.legend(loc="lower right")
    path = "./results/problem3/"
    if tau_preset is None:
        file = 'roc_LOSS_tau_' + 'avg' + '.png'
    else:
        file = 'roc_LOSS_tau_' + str(tau) + '.png'
    plt.savefig(path + file)