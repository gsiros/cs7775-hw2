# Problem 1: Evasion Attack using the Carlini-Wagner L2 norm method
# Source: http://arxiv.org/abs/1608.04644
# Acknowledgements:
#   https://github.com/carlini/nn_robust_attacks.git (Carlini's original implementation in TensorFlow)
#   https://github.com/kkew3/pytorch-cw2/ (kkew3's implementation of Carlini's attack, used as guide)
#   https://github.com/rwightman/pytorch-nips2017-attack-example 

from cw import AttackCarliniWagnerL2
from fmnist_loader import load_fmnist_torch
from net import SmallCNN
import torch
from torch.utils.data import DataLoader

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA available! Training on GPU.", flush=True)
else:
    device = torch.device('cpu')
    print("CUDA NOT available... Training on CPU.", flush=True)

fmnist = load_fmnist_torch()

# Load SmallCNN model and set it to evaluation mode
model = SmallCNN()
# Load parameters and not the architecture:
model.load_state_dict(torch.load('models/problem3/model_params.pth'), torch.device(device))
# PUT MODEL TO EVAL MODE SO THAT THE WEIGHTS ARE NOT CHANGED:
model.eval()

TARGET_CLASS = 3
BATCH_SIZE = 100

victims = list(fmnist["test"])
#shuffle(victims)
# Filter out the examples that have a label equal to 3:
#victims = list(filter(lambda x: x[1] != TARGET_CLASS, victims))
victims = victims[:BATCH_SIZE]

# Run the Carlini-Wagner L2 norm attack
loader = DataLoader(victims, batch_size=BATCH_SIZE, shuffle=False)

attack = AttackCarliniWagnerL2(
        targeted=True,
        constant_c=10,
        max_steps=5000,
        cuda=True if device == torch.device('cuda') else False,
)

results = []

for (input, label) in loader:

    # Clone the input to use it later for plotting
    og_input = input.clone()

    # Create target tensor that contains a target class different than the true label:
    target = torch.tensor([TARGET_CLASS]*BATCH_SIZE)
    
    input_adv, perturbation = attack.run(
        model, 
        input, 
        target
    )
    
    for (og, ex, pert, lab) in zip(og_input, input_adv, perturbation, label):
        input_adv_ex = ex.flatten().reshape(1, 1, 28, 28)
        input_adv_ex = torch.from_numpy(input_adv_ex).float()

        perturb = pert.flatten().reshape(1, 1, 28, 28)
        perturb = torch.from_numpy(perturb).float()
        
        out = model(input_adv_ex)
        pred = torch.argmax(out, dim=1)
        # Add to results
        results.append((og, input_adv_ex, perturb, lab, pred))

# Save to a file
torch.save(results, 'saved_data_c10.pt')