from flask import Flask, request, render_template,jsonify
app = Flask(__name__)
from flask import redirect, url_for
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import time
import datetime
import sys
import numpy as np
import argparse
import math
import itertools
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import io

##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


# Parameters
n_residual_blocks=9
img_height=256 #size of image height
img_width=256 #size of image width
channels = 3 #number of channels
input_shape = (channels, img_height, img_width)
lr=0.0002
b1=0.5 #adam: decay of first order momentum of gradient
b2=0.999 #adam: decay of first order momentum of gradient

# Image transformations
transforms_ = transforms.Compose([
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)
#D_A = Discriminator(input_shape)
#D_B = Discriminator(input_shape)


# Specify a path
#PATH_DA = "models/D_A_199.pth"
#PATH_DB = "models/D_B_199.pth"
PATH_GAB = "models/G_AB_199.pth"
PATH_GBA = "models/G_BA_199.pth"


# Load
G_AB.load_state_dict(torch.load(PATH_GAB, map_location=torch.device('cpu')))
G_BA.load_state_dict(torch.load(PATH_GBA, map_location=torch.device('cpu')))
#D_A = torch.load(PATH_DA, map_location=torch.device('cpu'))
#D_B = torch.load(PATH_DB, map_location=torch.device('cpu'))

# Optimizers
#optimizer_G = torch.optim.Adam(
    #itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
#)
#optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
#optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

G_AB.eval()
G_BA.eval()
#D_A.eval()
#D_B.eval()

# Test
predicted_image = G_AB(torch.randn(1, 3, img_height, img_width))
import matplotlib.pyplot as plt

# Save predicted image using pyplot
#plt.imshow(transforms.ToPILImage()(predicted_image.cpu()[0]))
#plt.savefig("predicted_image.png")


@app.route('/')
def index():
    return render_template('index.html')

#@app.route('/predict_painting')
#def go_to_prediction():
#	return render_template('/predict_painting.html')

#@app.route('/predict_image')
#def go_to_prediction_image():
#	return render_template('predict_image.html')


@app.route('/predict_painting', methods=['POST', 'GET'])
def predict_painting():
    # Get the image from post request
    if request.method == 'POST':
        img = request.files['image'].read()
        img = Image.open(io.BytesIO(img))
        img = img.convert('RGB')
        img = img.resize((img_width, img_height))
        # Convert image to tensor
        img = transforms.ToTensor()(img)
        # Add the batch dimension
        img = img.unsqueeze(0)
        # Create prediction
        with torch.no_grad():
            prediction = G_AB(img).cpu()
        autocontraster = transforms.RandomAutocontrast(0.5)
        prediction = autocontraster(prediction)
        #equalizer = transforms.RandomEqualize()
        #prediction = equalizer(prediction)
        # Save the image
        save_image(img, "static/images/actual.png")
        save_image(prediction, 'static/images/prediction.png')
        return redirect(url_for('predict_painting'))
    else :
        return render_template('predict_painting.html', image_path="prediction.png")

@app.route('/predict_image', methods=['POST', 'GET'])
def predict_image():
    # Get the image from post request
    if request.method == 'POST':
        img = request.files['painting'].read()
        img = Image.open(io.BytesIO(img))
        img = img.convert('RGB')
        img = img.resize((img_width, img_height))
        # Convert image to tensor
        img = transforms.ToTensor()(img)
        # Add the batch dimension
        img = img.unsqueeze(0)
        # Create prediction
        with torch.no_grad():
            prediction = G_BA(img).cpu()
        autocontraster = transforms.RandomAutocontrast(0.5)
        prediction = autocontraster(prediction)
        #equalizer = transforms.RandomEqualize()
        #prediction = equalizer(prediction)
        # Save the image
        save_image(img, "static/images/actual_image.png")
        save_image(prediction, 'static/images/prediction_image.png')
        return redirect(url_for('predict_image'))
    else:
        return render_template('predict_image.html', image_path="prediction.png")

if __name__ == '__main__':
    app.run(debug=True)
    