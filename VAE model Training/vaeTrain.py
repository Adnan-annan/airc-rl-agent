import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from IPython.display import Image
from IPython.core.display import Image, display



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


bs = 64
dataset = datasets.ImageFolder(root='C:\\Users\\adnan\\Desktop\\VAE\\dataset\\', transform=transforms.Compose([
    transforms.ToTensor(),
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=False)
len(dataset.imgs), len(dataloader)


fixed_x, _ = next(iter(dataloader))
save_image(fixed_x, 'real_image.png')
Image('real_image.png')




class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 3, 8)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=6144, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), F.softplus(self.fc2(h))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_fn(self, images, reconst, mean, logvar):
        KL = -0.5 * torch.sum((1 + logvar - mean.pow(2) - logvar.exp()), dim=0)
        KL = torch.mean(KL)
        reconstruction = F.binary_cross_entropy(reconst.view(-1,38400), images.view(-1, 38400), reduction='sum') #size_average=False)
        return reconstruction + 5.0 * KL



from torchsummary import summary
VARIANTS_SIZE = 32
image_channels = fixed_x.size(1)
vae = VAE(image_channels=image_channels, z_dim=VARIANTS_SIZE ).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
summary(vae, (3, 80, 160))

#vae.load_state_dict(torch.load("C:\\Users\\adnan\\vae-jetbot-legocity.torch"))
#print("loading model done")

from torch.utils.tensorboard import SummaryWriter
import numpy as np
epochs = 200
writer = SummaryWriter()

vae.train()
for epoch in range(epochs):
    losses = []
    gridR = None
    gridI = None
    for idx, (images, _) in enumerate(dataloader):
       
        images = images.to(device)
        optimizer.zero_grad()
        recon_images, mu, logvar = vae(images)
        loss = vae.loss_fn(images, recon_images, mu, logvar)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
        gridI = torchvision.utils.make_grid(images)
        gridR = torchvision.utils.make_grid(recon_images)

    writer.add_image('reconst', gridR, epoch)
    writer.add_image('Image', gridI, epoch)

    writer.add_scalar('Loss/train',np.average(losses), epoch)
    print("EPOCH: {} loss: {}".format(epoch+1, np.average(losses)))


    if epoch%10==0:
        torch.save(vae.state_dict(),"Feb22%d.torch"%(epoch+1), _use_new_zipfile_serialization=False)

    