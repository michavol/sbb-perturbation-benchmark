import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, dim_x, dim_z, layer_shapes):
        super(Encoder, self).__init__()
        layers = []
        input_dim = dim_x
        for output_dim in layer_shapes:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ELU())
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, dim_z))
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        return self.network(input)
    

class Decoder(nn.Module):
    
    def __init__(self, dim_z, dim_noise, sigma_noise, dim_x, layer_shapes, use_softplus):
        super(Decoder, self).__init__()
        self.dim_noise = dim_noise
        self.use_softplus = use_softplus  # applies log(1+exp(x)) to decoder output to model non-negative log1p transformed count data
        if dim_noise > 0:
            assert sigma_noise is not None and sigma_noise > 0
            self.stochastic = True
            self.sigma_noise = sigma_noise
        else: 
            self.stochastic = False

        layers = []
        input_dim = dim_z + dim_noise
        for output_dim in layer_shapes:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ELU()) 
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, dim_x))
        if self.use_softplus:
            layers.append(nn.Softplus())
            
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        if self.dim_noise > 0:
            # add noise for stochastic case
            noise = torch.randn(input.shape[:-1] + (self.dim_noise,)).to(input.device) * self.sigma_noise
            input = torch.concatenate((input, noise), dim=-1)

        return self.network(input)


class PerturbationAutoencoder(nn.Module):
    
    def __init__(
        self,
        dim_z,
        dim_noise,
        sigma_noise,
        dim_x,
        n_perturbations,
        encoder_layer_shapes,
        decoder_layer_shapes,
        use_softplus_after_decoder,
        update_encoder_on_reconstruction_loss=False,
        use_bias_in_perturbation_matrix=False,
    ):
        super().__init__()
        
        # model encoder mapping from observations to latents
        self.encoder = Encoder(dim_x, dim_z, encoder_layer_shapes)
        self.update_encoder_on_reconstruction_loss = update_encoder_on_reconstruction_loss

        # model decoder mapping from latents to observations; same class as true_mixing
        self.decoder = Decoder(dim_z, dim_noise, sigma_noise, dim_x, decoder_layer_shapes, use_softplus=use_softplus_after_decoder)

        # perturbation_matrix
        self.perturbation_matrix = nn.Linear(n_perturbations, dim_z, bias=use_bias_in_perturbation_matrix)
        # self.perturbation_matrix.weight.data = torch.tensor([[1., 0.], [0., 1.], [1., 1.]]).T
        # self.perturbation_matrix.requires_grad = False  # disable gradients for the perturbation matrix
        # print("Initial perturbation matrix weights:\n", self.perturbation_matrix.weight.data)


    def update_perturbation_matrix_via_OLS(self, z, perturbation_labels):
        ''' 
        Regression approach to learning the perturbation matrix.
        Args:
            z: (batch_size, dim_z) latents
            perturbation_labels: (batch_size, n_perturbations) binary per-environment perturbation labels
        '''
        z_means = []
        ix_pert_tr_ctrl = None
        perturbation_labels_unique = perturbation_labels.unique(dim=0) 
        # the order of the perturbations doesn't matter for estimating the perturbation matrix as long as the order of perturbation_labels_unique is consistent with the order of z_means
        for i, p in enumerate(perturbation_labels_unique):
            if p.eq(0).all():
                # this is the control condition, store its index
                ix_pert_tr_ctrl = i
            z_means.append(z[perturbation_labels.eq(p).all(dim=1)].mean(dim=0))
        z_means = torch.vstack(z_means)  # (n_perturbations, dim_z)
        # center means at first environment with label [0, 0, 0] for the OLS regression
        z_means_centered = z_means - z_means[ix_pert_tr_ctrl]  # center around the control condition
        self.perturbation_matrix.weight.data = torch.linalg.lstsq(perturbation_labels_unique, z_means_centered).solution.T
        if self.perturbation_matrix.bias is not None:
            self.perturbation_matrix.bias.data.zero_()


    def forward(self, x, perturbation_labels, return_components=False):
        '''
        Args:
            x: observations (batch_size, dim_x)
            perturbation_labels: (batch_size, n_perturbations)
        Returns:
            x_rec: reconstructed observations (batch_size, dim_x)
            x_perturbed: perturbed observations (batch_size, batch_size, dim_x)
            z_base: base latents (batch_size, dim_z)
        '''
        z = self.encoder(x)
        batch_size, dim_z = z.shape[0], z.shape[1]

        # compute reconstructions
        if self.update_encoder_on_reconstruction_loss:
            # allow for gradients to propagate back through the encoder
            x_rec = self.decoder(z)
        else:
            # detach gradients, so that the encoder is not updated based on the reconstruction loss
            x_rec = self.decoder(z.clone().detach())

        perturbation_shifts = self.perturbation_matrix(perturbation_labels) # (batch_size, dim_z)    

        # base
        z_base = z - perturbation_shifts  # (batch_size, dim_z)
        # Create a batch of z_perturbed, initialized with the original z
        # z_perturbed[i,j] == z[i], i.e. z is repeated along the second dimension
        z_perturbed = z.unsqueeze(1).expand(batch_size, batch_size, dim_z).clone() # (batch_size, batch_size, dim_z)
        # Create a difference matrix perturbation_diff (batch_size, batch_size, dim_z) 
        # perturbation_diff[i,j] == -perturbation_shifts[i] + perturbation_shifts[j]
        perturbation_diff = -perturbation_shifts.unsqueeze(1) + perturbation_shifts.unsqueeze(0)
        # Apply the perturbation shifts to all off-diagonal elements
        # z_perturbed[i,j] == z[i] + perturbation_diff[i,j] == z[i] - perturbation_shifts[i] + perturbation_shifts[j] == z_base[i] + perturbation_shifts[j]
        z_perturbed = z_perturbed + perturbation_diff

        # decode
        x_perturbed = self.decoder(z_perturbed)

        if return_components:
            return x_rec, x_perturbed, z_base, z, z_perturbed, perturbation_shifts, perturbation_diff
        return x_rec, x_perturbed, z_base


    @torch.no_grad()
    def predict(self, x_src, perturbation_labels_src, perturbation_labels_trg, return_components=False):
        '''
        Predicts an observation for every target environment given every source environment.
        Args:
            x_src:                      (batch_size, dim_x) source observations
            perturbation_labels_src:    (batch_size, n_perturbations) binary source perturbation labels
            perturbation_labels_trg:    (batch_size, n_perturbations) binary target perturbation labels
        Returns:
            x_trg:                      (batch_size * batch_size, dim_x) target observations
            z_trg:                      (batch_size * batch_size, dim_z) target latents
        '''
        z_src = self.encoder(x_src)  # (batch_size, dim_z)
        batch_size, dim_z = z_src.shape[0], z_src.shape[1]
        
        shifts_src = self.perturbation_matrix(perturbation_labels_src)  # (batch_size, dim_z)
        shifts_trg = self.perturbation_matrix(perturbation_labels_trg)  # (dim_z,)
        
        z_base = (z_src - shifts_src).unsqueeze(1) # (batch_size, 1, dim_z)
        shifts_trg = shifts_trg.unsqueeze(0) # (1, batch_size, dim_z)
        # z_trg[i,j] = z_base[i] + shifts_trg[j]
        z_trg = z_base + shifts_trg # (batch_size, batch_size, dim_z)
        z_trg = z_trg.view(-1, dim_z) # (batch_size * batch_size, dim_z)
        x_trg = self.decoder(z_trg)  

        if return_components:
            return x_trg, z_trg, z_base, shifts_src, shifts_trg
        return x_trg, z_trg
