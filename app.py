#!/usr/bin/env python
# coding: utf-8

# In[9]:


# app.py

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms
import matplotlib.pyplot as plt

# --- Generator Model ---
class Generator(torch.nn.Module):
    def __init__(self, z_dim=100, label_dim=10, img_dim=784):
        super().__init__()
        self.label_emb = torch.nn.Embedding(10, label_dim)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(z_dim + label_dim, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, img_dim),
            torch.nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        img = self.model(x)
        return img

# --- Load Model ---
device = torch.device("cpu")
z_dim = 100
generator = Generator(z_dim)
generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
generator.eval()

# --- Streamlit App ---
st.title("🖊️ Handwritten Digit Generator")
digit = st.number_input("Choose a digit (0-9):", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    z = torch.randn(5, z_dim)
    labels = torch.tensor([digit]*5)
    with torch.no_grad():
        fake_imgs = generator(z, labels).reshape(-1, 1, 28, 28)

    # Display images
    grid = make_grid(fake_imgs, nrow=5, normalize=True)
    ndarr = grid.mul(255).byte().permute(1, 2, 0).numpy()
    st.image(ndarr, caption=f"Generated Digit: {digit}", use_column_width=True)

