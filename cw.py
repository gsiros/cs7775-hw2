import sys
import torch
import numpy as np
from torch import optim
from torch import autograd
from helpers import *

class AttackCarliniWagnerL2:

    def __init__(self, targeted=True, constant_c=0.1, max_steps=5000, cuda=True):
        self.targeted = targeted 
        self.num_classes = 10 # FMNIST has 10 classes
        self.confidence = 0  # as set by Carlini in his code (we can change it to something different)
        self.initial_const = constant_c  # changed from default of .01 in Carlini's code
        self.max_steps = max_steps # number of iterations to perform gradient descent on the perturbation
        self.abort_early = True # abort binary search step early when losses stop decreasing
        self.clip_min = -1. # auxilliary value to convert to tanh-space
        self.clip_max = 1. # same^
        self.cuda = cuda # GPU (True) or CPU (False)
        self.init_rand = False  # experimental value to perform experiments based on random init
        self.inf = 1e10  # define infinity for the attack

    # Comparison function to boost the confidence of the adversarial examples
    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # Adversarial Loss function:
        # \\|\\delta\\|_2^2 + c \\cdot f(x + \\delta)
        # where  delta = x' - x
        # and f(x') = \\max\\{0, (\\max_{i \\ne t}{Z(x')_i} - Z(x')_t) \\cdot \\tau + \\kappa\\}
        
        # Compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        
        # Difference between target and non-target attacks:
        if self.targeted:
            # If the attack is targeted, optimize for making the other class most likely
            f6 = torch.clamp(other - real, min=0.)  # same as max(..., 0.)
        else:
            # Else, if the attack is non-targeted, optimize for making this class least likely.
            f6 = torch.clamp(real - other, min=0.)  # same as max(..., 0.)
        
        # Apply the scale constant that we computed with the binary steps and sum the batch:
        f6 = torch.sum(scale_const * f6)

        # Sum the L2 distance between the original and the adversarial images in the batch:
        l2_dist_loss = dist.sum()

        # Return the sum of the two losses:
        loss = l2_dist_loss + f6
        return loss

    def _optimize(self, optimizer, model, input_var, perturbation_var, target_var, scale_const_var, input_orig=None):
        # Apply perturbation and clamp resulting image to keep it bounded:
        input_adv = tanh_rescale(perturbation_var + input_var, self.clip_min, self.clip_max)

        output = model(input_adv)

        # distance to the original input data
        if input_orig is None:
            dist = l2_dist(input_adv, input_var, keepdim=False)
        else:
            dist = l2_dist(input_adv, input_orig, keepdim=False)

        loss = self._loss(output, target_var, dist, scale_const_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Return numpy arrays instead of torch Variables/Tensors:
        loss_np = loss.detach().numpy()
        dist_np = dist.detach().numpy()
        output_np = output.detach().numpy()
        input_adv_np = input_adv.permute(0, 2, 3, 1).detach().numpy()  # back to BHWC for numpy consumption
        perturb_np = perturbation_var.permute(0, 2, 3, 1).detach().numpy()  # back to BHWC for numpy consumption
        return loss_np, dist_np, output_np, input_adv_np, perturb_np

    def run(self, model, input, target):
        batch_size = input.size(0)

        # Set the lower and upper bounds accordingly for the constant c to optimize:
        scale_const = np.ones(batch_size) * self.initial_const

        # python/numpy placeholders for:
        #   overall best l2 distance, 
        #   overall best label (if targeted this is the target label, if untargeted this is the label that yields the best results),
        #   overall best adversarial image
        o_best_l2 = [self.inf] * batch_size
        o_best_score = [-1] * batch_size
        o_best_adv_ex = input.permute(0, 2, 3, 1).detach().numpy()
        o_best_perturb = np.zeros_like(o_best_adv_ex)

        
        # Convert inputs to tanh-space:
        input_var = autograd.Variable(torch_arctanh(input), requires_grad=False)
        # Keep the original input variable in tanh-space for calculating the l2 distance later in optimization:
        input_orig = tanh_rescale(input_var, self.clip_min, self.clip_max)

        # Convert target-class indexes to one-hot vectors, we need it for the loss function:
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.cuda:
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad=False)

        # Setup the perturbation variable.
        # THIS IS THE MAIN VARIABLE TO OPTIMIZE, THE ADVERSARIAL ATTACK.
        perturbation = torch.zeros(input_var.size()).float()
        if self.init_rand:
            # Experiment with a non-zero starting point...
            perturbation = torch.normal(means=perturbation, std=0.001)
        if self.cuda:
            perturbation = perturbation.cuda()
        # Convert to torch Variable:
        perturbation_var = autograd.Variable(perturbation, requires_grad=True)

        # Setup the Adam optimizer:
        optimizer = optim.Adam([perturbation_var], lr=0.0005)

        # Convert scale constant c from numpy array to torch tensor:
        scale_const_tensor = torch.from_numpy(scale_const).float()
        if self.cuda:
            scale_const_tensor = scale_const_tensor.cuda()
        # Convert to torch tensor:
        scale_const_var = autograd.Variable(scale_const_tensor, requires_grad=False)

        prev_loss = 1e6 # Initialize the previous loss to a large value, could be self.inf:
        # Begin GD optimization of the perturbation:
        for step in range(self.max_steps):
            # Optimize:
            loss, dist, output, adv_img, perturb = self._optimize(
                optimizer,
                model,
                input_var,
                perturbation_var,
                target_var,
                scale_const_var,
                input_orig
            )

            # Print stats every 100 steps and in the last step:
            if step % 100 == 0 or step == self.max_steps - 1:
                print('Step: {0:>4}, loss: {1:6.4f}, dist: {2:8.5f}, perturbation mean: {3:.5e}'.format(step, loss, dist.mean(), perturbation_var.data.mean()))


            # Abort early if the loss stops decreasing:
            # Carlini's suggestion...
            if self.abort_early and step % (self.max_steps // 10) == 0:
                if loss > prev_loss * .9999:
                    print('Aborting early...')
                    break
                prev_loss = loss

            # Update best results:
            for i in range(batch_size):
                target_label = target[i]
                output_logits = output[i]
                output_label = np.argmax(output_logits)
                di = dist[i]
            
                # If we find a successful adversarial example and the l2 distance is smaller than the
                # best distance found so far, update the best distance and the label:        
                if di < o_best_l2[i] and self._compare(output_logits, target_label):
                    o_best_l2[i] = di
                    o_best_score[i] = output_label
                    o_best_adv_ex[i] = adv_img[i]
                    o_best_perturb[i] = perturb[i]

            sys.stdout.flush()

        # Count the number of successful adversarial examples in each batch:
        batch_failure = 0
        batch_success = 0
        for i in range(batch_size):
            if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                batch_success += 1
            else:
                batch_failure += 1

        print('Adversarial Examples\n\tFailures: {0:2d}\n\tSuccesses: {1:2d}\n'.format(batch_failure, batch_success))

        sys.stdout.flush()
        return (o_best_adv_ex, o_best_perturb)