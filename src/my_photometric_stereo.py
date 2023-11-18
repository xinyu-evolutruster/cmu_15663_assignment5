#!/usr/bin/env python
# coding: utf-8

# %%
# Import the necessary packages
import os
import cv2
import skimage
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter

import cp_hw5

# %%


def pseudo_normal_to_albedo_and_normal(B, h, w, eps=1e-9):
    A = np.sqrt(np.sum(B ** 2, axis=0))

    N = B / (A + eps)
    A = A.reshape(h, w)
    N = N.reshape(3, h, w).transpose(1, 2, 0)

    return A, N


def display_albedo_and_normal(A, N, display_A=True):
    N_new = (N + 1) / 2

    if display_A:
        plt.imshow(A * 10, cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.show()

    plt.imshow(N_new)
    plt.axis('off')
    plt.show()


# %%
# Methods used to compute x and y gradient
def gradient_x(img_channel):
    return np.gradient(img_channel, axis=1, edge_order=2)


def gradient_y(img_channel):
    return np.gradient(img_channel, axis=0, edge_order=2)


def build_matrix_A(B, B_grad_x, B_grad_y):
    A_1 = B[0] * B_grad_x[1] - B[1] * B_grad_x[0]
    A_2 = B[0] * B_grad_x[2] - B[2] * B_grad_x[0]
    A_3 = B[1] * B_grad_x[2] - B[2] * B_grad_x[1]
    A_4 = -B[0] * B_grad_y[1] + B[1] * B_grad_y[0]
    A_5 = -B[0] * B_grad_y[2] + B[2] * B_grad_y[0]
    A_6 = -B[1] * B_grad_y[2] + B[2] * B_grad_y[1]

    A = np.array([A_1, A_2, A_3, A_4, A_5, A_6]).T

    return A


# %%
def display_surface(Z):
    # Z is an hxw array of surface depths
    h, w = Z.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    # set 3D figure
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    # add a light and shade to the axis for visual effect
    ls = LightSource()
    color_shade = ls.shade(-Z, plt.cm.gray)

    # display a surface
    surf = ax.plot_surface(
        x, y, -Z, facecolors=color_shade, rstride=4, cstride=4)

    # turn off axis
    plt.axis('off')
    plt.show()


# %%
# Load the images of a lambertian object
data_dir = '../data/toilet_paper'
num_imgs = 7
input_imgs = []
for i in range(1, 1 + num_imgs):
    img = cv2.imread(os.path.join(
        data_dir, 'toilet_paper_{}_cropped.tiff'.format(i)), -1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for c in range(3):
        img_rgb[:, :, c] = cv2.blur(img_rgb[:, :, c], (2, 2))
    img_rgb[img_rgb < 2000] = 0

    img_rgb = cv2.resize(img_rgb, (500, 500))
    img_rgb = img_rgb[120:450, 90:450]
    input_imgs.append((img_rgb * 2).astype(np.int32))

G = np.array([[1, 0, 0], [0, 1, 0], [0.0, 0.0, 1.0]])

plt.imshow(input_imgs[0] / (2**16-1))
plt.axis('off')
plt.show()

# %%
# Load the images of the object with some glossiness
data_dir = '../data/plush'
num_imgs = 8
input_imgs = []

for i in range(1, 1 + num_imgs):
    img = cv2.imread(os.path.join(
        data_dir, 'gira_cropped_{}.tiff'.format(i)), -1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[100:, :]

    for c in range(3):
        img_rgb[:, :, c] = cv2.blur(img_rgb[:, :, c], (4, 4))

    img_rgb = cv2.resize(img_rgb, (400, 500))

    input_imgs.append(img_rgb)

G = np.array([[1, 0, 0], [0, 1, 0], [0.08, 0.0, 10.0]])

plt.imshow(input_imgs[0])
plt.axis('off')
plt.show()

# %%
# Load the images of a lambertian object with more complicated shape
data_dir = '../data/cupboard'
num_imgs = 8
input_imgs = []

for i in range(1, 1 + num_imgs):
    img = cv2.imread(os.path.join(
        data_dir, 'cp_{}_cropped.tiff'.format(i)), -1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for c in range(3):
        img_rgb[:, :, c] = cv2.blur(img_rgb[:, :, c], (2, 2))

    img_rgb = cv2.resize(img_rgb, (480, 400))

    input_imgs.append(img_rgb * 2)

G = np.array([[1., 0., 0.], [0., 1., 0.], [0.1, 0.24, 1.7]])

plt.imshow(input_imgs[0] / (2**16 - 1))
plt.axis('off')
plt.show()

# %%
# Stack the `num_img` luminance channels into a matrix I of size `num_img`xP
h, w, c = input_imgs[0].shape
I = np.zeros((num_imgs, h * w))
for i in range(num_imgs):
    img_xyz = skimage.color.rgb2xyz(input_imgs[i])
    img_y = img_xyz[:, :, 1].reshape((img_xyz.shape[0]*img_xyz.shape[1]))
    I[i] = img_y

# %%
# Uncalibrated photometric stereo
# SVD decomposition

# L: 3x`num_img``
# B: 3XP
# I = L^T x B

U, S, Vh = np.linalg.svd(I, full_matrices=False)

LT = U[:, :3]
S = S[:3]
B = Vh[:3, :]

S_sqrt = np.sqrt(np.diag(S))
L_e = (LT @ S_sqrt).T
B_e = S_sqrt @ B

# %%
# Break B_e into albedoes A_e and normals N_e
A_e, N_e = pseudo_normal_to_albedo_and_normal(B_e, h, w)
display_albedo_and_normal(A_e, N_e)

# %%
# Test with different sigmas
for sigma in range(1, 6):

    B_blurred = B_e.copy()
    B_blurred = B_blurred.reshape(3, h, w).transpose(1, 2, 0)
    B_grad_x = np.zeros_like(B_blurred)
    B_grad_y = np.zeros_like(B_blurred)

    for c in range(3):
        B_blurred[:, :, c] = gaussian_filter(B_blurred[:, :, c], sigma)
        B_grad_x[:, :, c] = gradient_x(B_blurred[:, :, c])
        B_grad_y[:, :, c] = gradient_y(B_blurred[:, :, c])

    B_grad_x = B_grad_x.reshape(-1, 3).transpose(1, 0)
    B_grad_y = B_grad_y.reshape(-1, 3).transpose(1, 0)

    A = build_matrix_A(B_e, B_grad_x, B_grad_y)

    # Use SVD to solve the homogeneous equation
    # If Ax = 0, then x is the last column of V where A = UDV^T
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    x = Vh[-1]

    Delta = np.array([
        [-x[2], x[5], 1.0],
        [x[1], -x[4], 0],
        [-x[0], x[3], 0]
    ])

    Delta_inv = np.linalg.inv(Delta)
    B_int = Delta_inv @ B_e

    G_F = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])
    B_int = G @ G_F @ B_int

    A_int, N_int = pseudo_normal_to_albedo_and_normal(B_int, h, w)
    display_albedo_and_normal(A_int, N_int, display_A=False)

    print("sigma =", sigma)

# %%

sigma = 1

B_blurred = B_e.copy()
B_blurred = B_blurred.reshape(3, h, w).transpose(1, 2, 0)
B_grad_x = np.zeros_like(B_blurred)
B_grad_y = np.zeros_like(B_blurred)

for c in range(3):
    B_blurred[:, :, c] = gaussian_filter(B_blurred[:, :, c], sigma)
    B_grad_x[:, :, c] = gradient_x(B_blurred[:, :, c])
    B_grad_y[:, :, c] = gradient_y(B_blurred[:, :, c])

B_grad_x = B_grad_x.reshape(-1, 3).transpose(1, 0)
B_grad_y = B_grad_y.reshape(-1, 3).transpose(1, 0)

A = build_matrix_A(B_e, B_grad_x, B_grad_y)

# Use SVD to solve the homogeneous equation
# If Ax = 0, then x is the last column of V where A = UDV^T
U, S, Vh = np.linalg.svd(A, full_matrices=False)
x = Vh[-1]

Delta = np.array([
    [-x[2], x[5], 1.0],
    [x[1], -x[4], 0],
    [-x[0], x[3], 0]
])

Delta_inv = np.linalg.inv(Delta)
B_int = Delta_inv @ B_e

G_F = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0]
])
B_int = G @ G_F @ B_int

A_int, N_int = pseudo_normal_to_albedo_and_normal(B_int, h, w)
display_albedo_and_normal(A_int, N_int, display_A=True)

N_int = N_int[30:, 30:]

# The normal to the surface is in the direction (-dz/dx, -dz/dy, 1)
zx = N_int[:, :, 0] / (N_int[:, :, 2] + 1e-10)
zy = N_int[:, :, 1] / (N_int[:, :, 2] + 1e-10)
Z = cp_hw5.integrate_poisson(zx, zy).astype(np.float32)

plt.imshow(Z)
plt.axis('off')
plt.show()

display_surface(Z)

# In[13]:
# New rendering under different lighting condition
custom_ray = np.array([-10, 10, 20])
custom_ray = custom_ray / np.linalg.norm(custom_ray)

color = A_int * np.dot(N_int, custom_ray)
color[color < 0] = 0

plt.imshow(color * 3, cmap='gray', vmin=0, vmax=1.0)
plt.axis('off')
plt.show()

# %%
