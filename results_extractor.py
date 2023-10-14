import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch
import numpy as np
from fmnist_loader import *

def extract_prob2():
    # Initialize an empty list to store the tuples
    data_list = []

    # Initialize experiment number
    experiment_number = 0

    # Open the file and read it line by line
    with open('results/problem2/out.txt', 'r') as file:
        for line in file:
            # Check if the line starts with '!'
            if line.startswith('!'):
                if "EXPERIMENT" in line:
                    experiment_number += 1
                elif "Backdoor trigger pixels" in line:
                    b = int(line.split(":")[1].strip())
                elif "Percentage of poisoned data points" in line:
                    p = float(line.split(":")[1].strip())
                elif "Position of trigger" in line:
                    trigger_location = line.split(":")[1].strip()
                elif "Main task Accuracy (MA)" in line:
                    ma = float(line.split(":")[1].split('%')[0].strip())
                elif "Backdoor Accuracy (BA)" in line:
                    ba = float(line.split(":")[1].split('%')[0].strip())
                    # Once we have BA, we can be sure that we have all the other data as well for this experiment
                    # So, construct the tuple and append to the list
                    data_tuple = (experiment_number, b, p, trigger_location, ma, ba)
                    data_list.append(data_tuple)

    # Group data by p value
    grouped_data = defaultdict(list)
    for entry in data_list:
        grouped_data[entry[2]].append(entry)

    # For each unique p, plot a figure
    for p, experiments in grouped_data.items():
        # Separate the data
        b_values = [t[1] for t in experiments]
        trigger_locations = [t[3] for t in experiments]
        ma_values = [t[4] for t in experiments]
        ba_values = [t[5] for t in experiments]
        
        # Create x-axis labels combining b and trigger location
        x_labels = [f"{b} ({loc})" for b, loc in zip(b_values, trigger_locations)]
        
        # Set the width of a bar
        barWidth = 0.3
        r1 = np.arange(len(b_values))
        r2 = [x + barWidth for x in r1]
        
        # Plotting
        fig, ax = plt.subplots()
        
        # Create bars
        ax.bar(r1, ma_values, width=barWidth, color='blue', edgecolor='grey', label='MA')
        ax.bar(r2, ba_values, width=barWidth, color='red', edgecolor='grey', label='BA')
        
        # Title & Subtitle
        plt.title(f'Accuracy vs. Number of Pixels & Trigger Location for p = {p}')
        plt.xlabel('Number of Pixels (b) with Trigger Location', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        
        # X axis
        plt.xticks([r + barWidth for r in range(len(b_values))], x_labels, rotation=45, ha='right')

        
        # Create legend & Show graphic
        plt.legend()
        plt.tight_layout()  # Adjust layout for better view
        plt.show()

    # Print the list
    return data_list

def extract_prob1a():
    # Load the data:
    results = torch.load('results/problem1/saved_data_c1.pt')

    # Calculate perturbation sizes (L2 norm)
    perturbation_sizes = [torch.norm(entry[2]).item() for entry in results]

    # Define the range of epsilon values
    min_epsilon = min(perturbation_sizes)
    max_epsilon = max(perturbation_sizes)
    epsilons = np.linspace(min_epsilon, max_epsilon, 10)

    attack_success_rates = []

    # Calculate attack success rate for each epsilon
    for epsilon in epsilons:
        successful_attacks = sum(1 for pert, lab, pred in zip(perturbation_sizes, 
                                                            [entry[3] for entry in results], 
                                                            [entry[4] for entry in results])
                                if pert <= epsilon and lab != pred)
        success_rate = successful_attacks / len(results)
        attack_success_rates.append(success_rate)

    # Plot
    plt.plot(epsilons, attack_success_rates, marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Attack Success Rate')
    plt.title('Attack Success Rate vs Epsilon (CDF)\n(C = 1)')
    plt.grid(True)
    plt.show()

def extract_prob1b():

    # Load the data:
    results1 = torch.load('results/problem1/saved_data_c01.pt')
    results2 = torch.load('results/problem1/saved_data_c1.pt')
    results3 = torch.load('results/problem1/saved_data_c10.pt')

    def calculate_success_rates(results):
        # Calculate perturbation sizes (L2 norm)
        perturbation_sizes = [torch.norm(entry[2]).item() for entry in results]

        # Define the range of epsilon values
        min_epsilon = min(perturbation_sizes)
        max_epsilon = max(perturbation_sizes)
        epsilons = np.linspace(min_epsilon, max_epsilon, 10)

        attack_success_rates = []

        # Calculate attack success rate for each epsilon
        for epsilon in epsilons:
            successful_attacks = sum(1 for pert, lab, pred in zip(perturbation_sizes, 
                                                                [entry[3] for entry in results], 
                                                                [entry[4] for entry in results])
                                    if pert <= epsilon and lab != pred)
            success_rate = successful_attacks / len(results)
            attack_success_rates.append(success_rate)

        return epsilons, attack_success_rates

    # Calculate success rates for each experiment
    epsilons1, success_rates1 = calculate_success_rates(results1)
    epsilons2, success_rates2 = calculate_success_rates(results2)
    epsilons3, success_rates3 = calculate_success_rates(results3)

    # Plot results for all experiments on the same graph
    plt.plot(epsilons1, success_rates1, marker='o', label='C = 0.1')
    plt.plot(epsilons2, success_rates2, marker='o', label='C = 1')
    plt.plot(epsilons3, success_rates3, marker='o', label='C = 10')

    plt.xlabel('Epsilon')
    plt.ylabel('Attack Success Rate')
    plt.title('Attack Success Rate vs Epsilon (CDF)')
    plt.legend()
    plt.grid(True)
    plt.show()

def extract_prob1c(TARGET_CLASS=3):

    # Load the data:
    results = torch.load('results/problem1/saved_data_c10.pt')

    # Filter data such that the label is not equal to the target class and the prediction is equal to the target class
    results = list(filter(lambda x: x[3] != TARGET_CLASS and x[4] == TARGET_CLASS, results))

    # Calculate perturbation sizes (L2 norm)
    perturbation_sizes = [torch.norm(entry[2]).item() for entry in results]

    # Get indices of top 3 minimum and maximum perturbations
    min_indices = np.argsort(perturbation_sizes)[:3]
    max_indices = np.argsort(perturbation_sizes)[-3:]

    # Gather those samples
    min_samples = [results[i] for i in min_indices]
    max_samples = [results[i] for i in max_indices]

    # Setting up the figure and axes
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 3*4))
    
    
    for i, (og, ex, pert, _, _) in enumerate(min_samples):
        # Original Image
        axes[i, 0].imshow(og.squeeze().numpy(), cmap='gray')  
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Perturbation
        axes[i, 1].imshow(pert.squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title('Perturbation')
        axes[i, 1].axis('off')

        # Adversarial Example
        axes[i, 2].imshow(ex.squeeze().numpy(), cmap='gray')
        axes[i, 2].set_title('Adversarial Example')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

    for i, (og, ex, pert, _, _) in enumerate(max_samples):
        # Original Image
        axes[i, 0].imshow(og.squeeze().numpy(), cmap='gray')  # Assuming grayscale images; adjust if needed
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Perturbation
        axes[i, 1].imshow(pert.squeeze().numpy(), cmap='gray')  # Assuming grayscale images; adjust if needed
        axes[i, 1].set_title('Perturbation')
        axes[i, 1].axis('off')

        # Adversarial Example
        axes[i, 2].imshow(ex.squeeze().numpy(), cmap='gray')  # Assuming grayscale images; adjust if needed
        axes[i, 2].set_title('Adversarial Example')
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.show()

extract_prob2()