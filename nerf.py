from functools import lru_cache
import sys, os, argparse, json, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools

from common import *

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="dataset/10TK")
parser.add_argument('--output', type=str, default="results/10TK/nerf")
args = parser.parse_args()



class NeRFDataset(torch.utils.data.Dataset):
	def __init__(self, frameList, scale=1):
		self.frameList = frameList
		self.scale = scale


	def __len__(self):
		return len(self.frameList)


	@property
	def shape(self):
		return (int(self.frameList[0].height * self.scale), int(self.frameList[0].width * self.scale), 3)


	def __getitem__(self, idx):
		frame = self.frameList[idx]
		img = cv2.imread(frame.file_path) / 255
		img = cv2.resize(img, None, fx=self.scale, fy=self.scale)
		x = np.linspace(0, frame.width, img.shape[1])
		y = np.linspace(0, frame.height, img.shape[0])
		pos = np.dstack(np.meshgrid(x, y)).reshape(-1,2)
		pos = np.hstack((pos, np.ones_like(pos[:,0:1])))
		vec = np.linalg.inv(frame.intrinsic) @ pos.T
		vec = (frame.rmat @ vec).T
		vec /= np.linalg.norm(vec, axis=-1)[:,np.newaxis]
		vec = np.concatenate((np.repeat(frame.tvec[np.newaxis,:], vec.shape[0], axis=0), vec), axis=-1)
		colors = img.reshape(-1,3)

		return vec.astype(np.float32), colors.astype(np.float32)



class DivisionDataset(torch.utils.data.Dataset):
	def __init__(self, dataset, division):
		self.dataset = dataset
		self.division = division


	def __len__(self):
		return len(self.dataset) * self.division


	def __getitem__(self, idx):
		inputs, targets = self.dataset[idx // self.division]

		div_ind = idx % self.division
		size = inputs.shape[0] // self.division
		inputs = inputs[div_ind * size:(div_ind + 1) * size]
		targets = targets[div_ind * size:(div_ind + 1) * size]

		return inputs, targets



class NeRFModule(nn.Module):
	def __init__(self, 
		num_first_layers=4, num_second_layers=3, latent_dim=256, 
		encoding_length_pos=10, encoding_length_dir=4):
		super().__init__()

		self.encoding_length_pos = encoding_length_pos
		self.encoding_length_dir = encoding_length_dir
		self.latent_dim = latent_dim

		layers = [nn.Linear(self.encoding_length_pos * 6, self.latent_dim)]
		for i in range(num_first_layers):
			layers.append(nn.ReLU())
			layers.append(nn.Linear(self.latent_dim, self.latent_dim))
		layers.append(nn.ReLU())
		self.first_module = nn.Sequential(*layers)

		layers = [nn.Linear(self.encoding_length_pos * 6 + self.latent_dim, self.latent_dim)]
		for i in range(num_second_layers):
			layers.append(nn.ReLU())
			layers.append(nn.Linear(self.latent_dim, self.latent_dim))
		layers.append(nn.Linear(self.latent_dim, self.latent_dim + 1))
		self.second_module = nn.Sequential(*layers)

		self.final_module = nn.Sequential(
			nn.ReLU(),
			nn.Linear(self.encoding_length_dir * 6 + self.latent_dim, self.latent_dim // 2),
			nn.ReLU(),
			nn.Linear(self.latent_dim // 2, 3),
			nn.Sigmoid(),
		)


	def forward(self, inputs):
		x, d = inputs[:,:3], inputs[:,3:]
		x_enc = self.positional_encoding(x, self.encoding_length_pos)
		d_enc = self.positional_encoding(d, self.encoding_length_dir)

		v = self.first_module(x_enc)
		v = self.second_module(torch.cat([x_enc, v], dim=-1))
		sigma = v[:,0]
		v = v[:,1:]
		rgb = self.final_module(torch.cat([d_enc, v], dim=-1))

		return sigma, rgb


	def positional_encoding(self, x, length):
		ret = []
		for l in range(length):
			ret.append(torch.sin((2 ** l) * np.pi * x))
			ret.append(torch.cos((2 ** l) * np.pi * x))
		return torch.cat(ret, dim=1)



class NeRFModel(nn.Module):
	def __init__(self, nerf_module, max_depth, depth_resolution=64):
		super().__init__()
		self.nerf_module = nerf_module
		self.max_depth = max_depth
		self.depth_resolution = depth_resolution
		self.loss = nn.MSELoss()


	def forward(self, inputs, targets=None):
		pos, vec = inputs[:,:,:3], inputs[:,:,3:]
		mass = []
		rgbs = []
		for d in range(self.depth_resolution):
			pos_ = pos + vec * d / self.depth_resolution * self.max_depth
			x = torch.cat((pos_, vec), dim=-1).view(-1,6)
			sigma, rgb = self.nerf_module(x)
			mass.append(sigma * self.max_depth / self.depth_resolution)
			rgbs.append(rgb)

		mass = torch.stack(mass, dim=1)
		mass = F.pad(mass, (1, 0), mode='constant', value=0.0)
		alpha = 1.0 - torch.exp(-mass[:, 1:])
		rgbs = torch.stack(rgbs, dim=1)

		T = torch.exp(-torch.cumsum(mass[:, :-1], dim=1))
		w = T * alpha
		bg = 0.0
		colors = torch.sum(w[..., None] * rgbs, dim=1)
		colors += (1.0 - torch.sum(w, dim=1, keepdims=True)) * bg
		colors = colors.view(*vec.shape)

		if targets is not None:
			return self.loss(colors, targets)
		else:
			return colors



if __name__ == '__main__':
	data = json.load(open(os.path.join(args.dataset, "transforms_test.json")))
	frameList = read_transforms(args.dataset, data)

	nerf_module = NeRFModule()
	nerf_model = NeRFModel(nerf_module, max_depth=20).cuda()
	raw_dataset = NeRFDataset(frameList, scale=0.25)
	train_dataset = DivisionDataset(raw_dataset, division=64)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
	test_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=1, shuffle=True)

	optimizer = torch.optim.Adam(nerf_model.parameters(), lr=5e-4)

	for epoch in range(10):
		for i, (inputs, targets) in enumerate(train_loader):
			inputs = inputs.cuda()
			targets = targets.cuda()
			loss = nerf_model(inputs, targets)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print("\r[Epoch %d/%d, Iter: %d/%d] loss: %.5f" % (epoch, 10, i, len(train_loader), loss.item()), end='')

			if i % 100 == 0:
				with torch.no_grad():
					nerf_model.eval()
					for j, (inputs, targets) in enumerate(test_loader):
						inputs = inputs.cuda()
						outputs = nerf_model(inputs)
						for k in range(len(outputs)):
							img = outputs[k].cpu().numpy().reshape(raw_dataset.shape)
							gt = targets[k].numpy().reshape(raw_dataset.shape)
							cv2.imwrite(os.path.join(args.output, "%d_%d_predict.png" % (epoch, i)), img * 255)
							cv2.imwrite(os.path.join(args.output, "%d_%d_gt.png" % (epoch, i)), gt * 255)

						break
					nerf_model.train()

		print()

