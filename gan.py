import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import pytorch_lightning as pl

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=16)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=16)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=16)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        x = x.view(-1, 320)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return torch.sigmoid(x)

# generate fake data of shape [1, 28, 28]
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.lin1 = nn.Linear(latent_dim, 7*7*64) # [n, 64, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7) # [n, 1, 28, 28]
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7) 
        
        x = self.ct1(x)
        x = F.relu(x)
        
        x = self.ct2(x)
        x = F.relu(x)
        
        return self.conv(x)

class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
        
        self.validation_z = torch.randn(6, self.hparams.latent_dim)
        
        self.automatic_optimization = False
    
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        opt_g, opt_d = self.optimizers()
        
        opt_g.zero_grad()
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim).to(self.device)
        fake_imgs = self(z)
        y_hat = self.discriminator(fake_imgs)
        y = torch.ones(real_imgs.size(0), 1).to(self.device)
        g_loss = self.adversarial_loss(y_hat, y)
        self.manual_backward(g_loss)
        opt_g.step()
        
        opt_d.zero_grad()
        y_hat_real = self.discriminator(real_imgs)
        y_real = torch.ones(real_imgs.size(0), 1).to(self.device)
        real_loss = self.adversarial_loss(y_hat_real, y_real)
        
        y_hat_fake = self.discriminator(fake_imgs.detach())
        y_fake = torch.zeros(real_imgs.size(0), 1).to(self.device)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
        
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Logging
        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []
    
    def plot_imgs(self):
        z = self.validation_z.to(self.device)
        sample_imgs = self(z)
        
        print('epoch: ', self.current_epoch)
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach().cpu()[i, 0, :, :], cmap='gray_r', interpolation=None)
            plt.title('generated data')
            plt.xticks([])
            plt.yticks([])
            plt.axis(False)
        plt.show()
    
    def on_train_epoch_end(self):
        self.plot_imgs()
        

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dm = MNISTDataModule()
    model = GAN().to(device)    
    
    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(model, dm)  