import torch
import numpy as np
from torch import linalg as LA

class VAE_Loss(torch.nn.Module):
  def __init__(self, feats):
    super(VAE_Loss, self).__init__()
    self.mse = torch.nn.MSELoss()
    self.n_feats = feats
  
  def KL_Loss(self, z_mean, z_log_variance, z):
    return -0.5 * torch.sum(1 + z_log_variance - z_mean.pow(2) - z_log_variance.exp(), dim=0)
  
  def negative_reconstruction_loss(self, x_mean, x):
    return -0.5 * torch.sum((x-x_mean).pow(2), dim=0)

    #D out of sum or inside of it
  def forward(self, loss_mats):
    elbo = 0
    for z_mean, z_log_var, z, x_mean, x in loss_mats:
      kl_loss = self.KL_Loss(z_mean, z_log_var, z)
      nrecon_loss = self.negative_reconstruction_loss(x_mean, x)
      elbo +=  kl_loss - nrecon_loss
    elbo /= len(loss_mats)
    print("elbo:", elbo)
    return elbo
  
  def get_loss(self, loss_mats):
    scores = []
    for _, _, _, x_mean, x in loss_mats:
      recon_loss = (x-x_mean).pow(2)
      scores.append(recon_loss.detach().numpy())
    return scores