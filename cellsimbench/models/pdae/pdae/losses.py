import torch

device = None
def set_losses_device(_device):
     global device 
     device = _device


def get_reconstruction_loss(x, x_rec, x_perturbed, beta):
        # see DPA paper, Algorithm 1, page 11
        # extract second set of reconstructions from the diagonal of the perturbed observations:
        # for stochastic decoder:       x_rec_prime is based on the same z, but different noise and thus different from x_rec
        # for deterministic decoder:    x_rec_prime is equal to x_rec
        x_rec_prime = x_perturbed.diagonal().permute(1,0) # batch_size x batch_size x dim_x -> batch_size x dim_x
        
        r1 = torch.linalg.vector_norm(x - x_rec, dim=1).pow(beta).mean()
        r2 = torch.linalg.vector_norm(x - x_rec_prime, dim=1).pow(beta).mean()
        r3 = torch.linalg.vector_norm(x_rec - x_rec_prime, dim=1).pow(beta).mean()
        # for stochastic decoder:       rec_loss is the DPA loss
        # for deterministic decoder:    we have r1 = r2 and r3 = 0, so rec_loss reduces to pointwise reconstruction loss
        reconstruction_loss = 0.5 * (r1 + r2 - r3)
        return reconstruction_loss


def get_perturbation_energy_loss(x, x_perturbed, beta, return_components=False, remove_marginal=True):
    '''Compute the loss for the perturbation autoencoder.
    Args:
        x: (batch_size, dim_x) observations
        x_perturbed: (batch_size, batch_size, dim_x) synthetic perturbed observations where x_perturbed[i,j] is x[i->j]
    '''
    batch_size, dim_x = x.shape[0], x.shape[1]
    
    x_perturbed = x_perturbed.transpose(0,1) # we need x_perturbed[i,j] to be x_perturbed[j->i]

    # "similarities" = pairwise distances between X_i and X_{j->i}
    similarities = torch.cdist(x_perturbed, x.unsqueeze(1)).pow(beta) # (batch_size, batch_size, batch_size)

    # "diversities" = pairwise distances between X_{j->i} and X'_{j->i} 
    diversities = torch.cdist(x_perturbed, x_perturbed).pow(beta) # (batch_size, batch_size, batch_size)

    # perturbation energy loss = distribution matching across perturbations
    if remove_marginal:
        # zero out unwanted entries in similarities and diversities
        idxs = torch.arange(batch_size)
        similarities[idxs, idxs] = 0.0  # set diagonal to zero
         
        i_idx = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1)   # shape [batch_size,1,1]
        j_idx = torch.arange(batch_size, device=x.device).view(1, batch_size, 1)   # shape [1,batch_size,1]
        k_idx = torch.arange(batch_size, device=x.device).view(1, 1, batch_size)   # shape [1,1,batch_size]
        # make the mask: keep only triples where i != j  AND  i != k  AND  j != k
        mask = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx) # mask.shape == [batch_size, batch_size, batch_size]
        diversities = diversities * mask.float()  # set unwanted entries to zero

        p1 = similarities.sum() / (batch_size * (batch_size - 1))  # similarity term
        p2 = diversities.sum() / (batch_size * (batch_size - 1) * (batch_size - 2))  # diversity term

    else:
        p1 = similarities.sum() / (batch_size**2)
        p2 = diversities.sum() / (batch_size**3)

    perturbation_energy_loss = p1 - 0.5 * p2
    
    if return_components: # used in tests
         return perturbation_energy_loss, similarities, diversities
    else: 
        return perturbation_energy_loss


def get_marginal_prior_energy_loss(prior_noise, z_base, beta, return_components=False):
    batch_size, dim_z = z_base.shape[0], z_base.shape[1]
    similarities = torch.cdist(z_base, prior_noise).pow(beta)
    diversities = torch.cdist(z_base, z_base).pow(beta)

    s = similarities.sum() / (batch_size**2)
    d = diversities.sum() / (batch_size * (batch_size - 1))
    
    marginal_prior_energy_loss = s - 0.5 * d
    
    if return_components: # used in tests
         return marginal_prior_energy_loss, similarities, diversities
    else: 
        return marginal_prior_energy_loss


def get_l21_loss(perturbation_matrix):
    return torch.linalg.vector_norm(perturbation_matrix, ord=2, dim=1).sum()