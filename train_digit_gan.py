#!/usr/bin/env python
# coding: utf-8

# In[9]:


# train_digit_gan.py

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Data Preparation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

# --- 2. Generator ---
class Generator(nn.Module):
    def __init__(self, z_dim=100, label_dim=10, img_dim=784):
        super().__init__()
        self.label_emb = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(z_dim + label_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        img = self.model(x)
        return img

# --- 3. Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, label_dim=10, img_dim=784):
        super().__init__()
        self.label_emb = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(img_dim + label_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([img, label_embedding], dim=1)
        validity = self.model(x)
        return validity

# --- 4. Initialize Models and Optimizers ---
z_dim = 100
G = Generator(z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

# --- 5. Training Loop ---
epochs = 30
for epoch in range(epochs):
    for real_imgs, labels in dataloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.view(batch_size, -1).to(device)
        labels = labels.to(device)

        # Real and fake labels
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # --- Train Discriminator ---
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z, labels)
        D_real = D(real_imgs, labels)
        D_fake = D(fake_imgs.detach(), labels)
        loss_D = criterion(D_real, real) + criterion(D_fake, fake)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # --- Train Generator ---
        z = torch.randn(batch_size, z_dim).to(device)
        gen_imgs = G(z, labels)
        D_gen = D(gen_imgs, labels)
        loss_G = criterion(D_gen, real)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")

# Save generator
os.makedirs("models", exist_ok=True)
torch.save(G.state_dict(), "models/generator.pth")

