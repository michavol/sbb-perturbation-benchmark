import math
import torch
import torch.nn as nn

from PerturbationExtrapolation.pdae.model import Decoder


def get_random_perturbation_matrix(dim_z, n_perturbations):
    # each row can be viewed as a multi-node mean shift intervention)
    return torch.randn((n_perturbations, dim_z))

 
def get_perturbation_indices(n_perturbations):
    '''Creates matrix  of perturbation indices in which each row corresponds to one perturbation condition:
    either no perturbation, a single one, or two simultaneous ones.'''
    n_environments = 1 + n_perturbations + math.comb(n_perturbations, 2)  # obs + single + double
    indices = torch.zeros((n_environments, n_perturbations))
    
    # single perturbations
    for i in range(1, n_perturbations + 1):
        indices[i, i - 1] = 1.

    # double perturbations
    row = n_perturbations + 1
    for i in range(n_perturbations):
        for j in range(i+1, n_perturbations):
            indices[row, i] = 1.
            indices[row, j] = 1.
            row +=1

    return indices


# Set up latent distributions as mean-shifted isotropic Gaussians
def get_latent_sample(perturbation_indices, perturbation_matrix, n_samples, sigma, base_distribution='Gaussian'):
    '''Returns a perturbed latent sample, where perturbation_indices is a (binary)
    array indicating which perturbations have been performed, and perturbation
    matrix contains the corresponding mean shifts as rows.
    Args:
        perturbation_indices: (n_perturbations,)
        perturbation_matrix (n_perturbations, dim_z)
    '''    
    dim_z = perturbation_matrix.shape[1]
    mean_shift = torch.einsum('ij,i->j', perturbation_matrix, perturbation_indices).reshape(1, dim_z)
    if base_distribution == 'Gaussian':
        z_basal = torch.randn((n_samples, dim_z)) * sigma
        z = z_basal + mean_shift
    elif base_distribution == 'Uniform':
        z = torch.rand((n_samples, dim_z)) * sigma + mean_shift
    elif base_distribution == 'Laplace':
        z = torch.distributions.Laplace(0, 1).sample((n_samples, dim_z)) * sigma + mean_shift
    else:
        raise NotImplementedError('Unknown base distribution')
    return z, z_basal


def get_data(dim_z, dim_noise, sigma_noise, dim_x, n_perturbations, n_samples, mixing_layer_shapes, sigma_z, perturbation_matrix=None, perturbation_labels=None, fixed_mixing=None, base_distribution='Gaussian'):
    with torch.no_grad():
        if perturbation_matrix is None:
            perturbation_matrix = get_random_perturbation_matrix(dim_z, n_perturbations)
        
        if perturbation_labels is None:
            perturbation_labels = get_perturbation_indices(n_perturbations)  
        
        # perturbation_labels = torch.tensor(perturbation_labels, dtype=torch.float32)

        if fixed_mixing is not None:
            true_mixing = fixed_mixing
        else:
            true_mixing = Decoder(dim_z, dim_noise, sigma_noise, dim_x, mixing_layer_shapes)
            true_mixing.apply(lambda m: nn.init.xavier_uniform_(m.weight) if type(m) == nn.Linear else None)
        
        n_environments = perturbation_labels.shape[0]
        true_latents = torch.zeros((n_environments, n_samples, dim_z))
        true_basal_latents = torch.zeros((n_environments, n_samples, dim_z))
        observations = torch.zeros((n_environments, n_samples, dim_x))

        for i in range(n_environments):
            # perform the latent perturbation for the current environment
            true_latents[i], true_basal_latents[i] = get_latent_sample(perturbation_labels[i], perturbation_matrix, n_samples, sigma_z, base_distribution)
   
            # decode latents to observations
            observations[i] = true_mixing.forward(true_latents[i])

        # copy the perturbation_indices for each sample to get shape (n_environments, n_samples, n_perturbations)
        # perturbation_labels = perturbation_index_matrix.unsqueeze(1).repeat(1, n_samples, 1)
    
    return observations, perturbation_labels, perturbation_matrix, true_basal_latents, true_latents, true_mixing