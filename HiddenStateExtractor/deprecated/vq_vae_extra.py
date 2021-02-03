# 32 * 32 * 128, strong decoder
#class VQ_VAE(nn.Module):
#  def __init__(self,
#               num_inputs=3,
#               num_hiddens=128,
#               num_residual_hiddens=64,
#               num_residual_layers=2,
#               num_embeddings=128,
#               commitment_cost=0.25,
#               channel_var=CHANNEL_VAR,
#               alpha=0.1,
#               **kwargs):
#    super(VQ_VAE, self).__init__(**kwargs)
#    self.num_inputs = num_inputs
#    self.num_hiddens = num_hiddens
#    self.num_residual_layers = num_residual_layers
#    self.num_residual_hiddens = num_residual_hiddens
#    self.num_embeddings = num_embeddings
#    self.commitment_cost = commitment_cost
#    self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, 3, 1, 1)), requires_grad=False)
#    self.alpha = alpha
#    self.enc = nn.Sequential(
#        nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
#    self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost)
#    self.dec = nn.Sequential(
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers),
#        nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//4),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))

# 16*16*16, strong decoder
#class VQ_VAE(nn.Module):
#  def __init__(self,
#               num_inputs=3,
#               num_hiddens=16,
#               num_residual_hiddens=64,
#               num_residual_layers=2,
#               num_embeddings=64,
#               commitment_cost=0.25,
#               channel_var=CHANNEL_VAR,
#               alpha=0.1,
#               **kwargs):
#    super(VQ_VAE, self).__init__(**kwargs)
#    self.num_inputs = num_inputs
#    self.num_hiddens = num_hiddens
#    self.num_residual_layers = num_residual_layers
#    self.num_residual_hiddens = num_residual_hiddens
#    self.num_embeddings = num_embeddings
#    self.commitment_cost = commitment_cost
#    self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, 3, 1, 1)), requires_grad=False)
#    self.alpha = alpha
#    self.enc = nn.Sequential(
#        nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
#    self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost)
#    self.dec = nn.Sequential(
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers),
#        nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//4),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//4),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))

# 32*32*128, weak decoder
#class VQ_VAE(nn.Module):
#  def __init__(self,
#               num_inputs=3,
#               num_hiddens=128,
#               num_residual_hiddens=64,
#               num_residual_layers=2,
#               num_embeddings=32,
#               commitment_cost=0.25,
#               channel_var=CHANNEL_VAR,
#               alpha=0.1,
#               **kwargs):
#    super(VQ_VAE, self).__init__(**kwargs)
#    self.num_inputs = num_inputs
#    self.num_hiddens = num_hiddens
#    self.num_residual_layers = num_residual_layers
#    self.num_residual_hiddens = num_residual_hiddens
#    self.num_embeddings = num_embeddings
#    self.commitment_cost = commitment_cost
#    self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, 3, 1, 1)), requires_grad=False)
#    self.alpha = alpha
#    self.enc = nn.Sequential(
#        nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
#    self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost)
#    self.dec = nn.Sequential(
#        nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))