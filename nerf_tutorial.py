import glob, os, argparse, json
from functools import lru_cache

import numpy as np
import ray
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import *


torch.backends.cudnn.benchmark = True
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="dataset/10TK")
parser.add_argument('--output', type=str, default="results/10TK/nerf")
args = parser.parse_args()



def _line2floats(line):
	return map(float, line.strip().split())


def read_frame(frame, bg=1.0, crop=None):
	rgb = cv2.imread(frame.file_path, cv2.IMREAD_UNCHANGED) / 255
	if rgb.shape[2] == 4:
		rgb = np.clip(rgb[...,:3] + bg * (1 - rgb[...,3:]), 0.0, 1.0)

	if crop is not None:
		margin = ((rgb.shape[1] - crop[0]) // 2, (rgb.shape[0] - crop[1]) // 2)
		rgb = rgb[margin[1]:margin[1]+crop[1],margin[0]:margin[0]+crop[0]]

	rgb = cv2.resize(rgb, (frame.width, frame.height))[...,::-1]
	return rgb.astype(np.float32)


def split_ray(t_n, t_f, N, batch_size):
	"""Split the ray into N partitions.

	partition: [t_n, t_n + (1 / N) * (t_f - t_n), ..., t_f]

	Args:
		t_n (float): t_near. Start point of split.
		t_f (float): t_far. End point of split.
		N (int): Num of partitions.
		batch_size (int): Batch size.

	Returns:
		ndarray, [batch_size, N]: A partition.

	"""
	partitions = np.linspace(t_n, t_f, N+1, dtype=np.float32)
	return np.repeat(partitions[None], repeats=batch_size, axis=0)


def sample_coarse(partitions):
	"""Sample ``t_i`` from partitions for ``coarse`` network.

	t_i ~ U[t_n + ((i - 1) / N) * (t_f - t_n), t_n + (i / N) * (t_f - t_n)]

	Args:
		partitions (ndarray, [batch_size, N+1]): Outputs of ``split_ray``.

	Return:
		ndarray, [batch_size, N]: Sampled t.

	"""
	low = partitions[:, :-1]
	high = partitions[:, 1:]
	t = torch.rand(low.shape, dtype=torch.float32, device=partitions.device) * (high - low) + low
	return t


def _pcpdf(partitions, weights, N_s, det=False):
	# Get pdf
	weights = weights + 1e-5 # prevent nans
	pdf = weights / torch.sum(weights, -1, keepdim=True)
	cdf = torch.cumsum(pdf, -1)
	cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

	# Take uniform samples
	if det:
		u = torch.linspace(0., 1., steps=N_s)
		u = u.expand(list(cdf.shape[:-1]) + [N_s])
	else:
		u = torch.rand(list(cdf.shape[:-1]) + [N_s])

	# Invert CDF
	u = u.contiguous().to(weights)
	inds = torch.searchsorted(cdf, u, right=True)
	below = torch.max(torch.zeros_like(inds-1), inds-1)
	above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
	inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

	# cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
	# bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
	matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
	cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
	partitions_g = torch.gather(partitions.unsqueeze(1).expand(matched_shape), 2, inds_g)

	denom = (cdf_g[...,1]-cdf_g[...,0])
	denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
	t = (u-cdf_g[...,0])/denom
	samples = partitions_g[...,0] + t * (partitions_g[...,1]-partitions_g[...,0])

	return samples


def sample_fine(partitions, weights, t_c, N_f):
	"""Sample ``t_i`` from partitions for ``fine`` network.

	Sampling from each partition according to given weights.

	Args:
		partitions (ndarray, [batch_size, N_c+1]): Outputs of ``split_ray``.
		weights (ndarray, [batch_size, N_c]):
			T_i * (1 - exp(- sigma_i * delta_i)).
		t_c (ndarray, [batch_size, N_c]): ``t`` of coarse rendering.
		N_f (int): num of sampling.

	Return:
		ndarray, [batch_size, N_c+N_f]: Sampled t.

	"""
	t_f = _pcpdf(partitions, weights, N_f)
	t = torch.cat([t_c, t_f], axis=1)
	t = torch.sort(t, axis=1)[0]
	return t


def _rgb_and_weight(func, ray, t, N):
	batch_size = ray.shape[0]

	o = ray[:, :3]
	d = ray[:, 3:]
	x = o[:, None] + t[..., None] * d[:, None]
	x = x.view(batch_size, N, -1)
	d = d[:, None].repeat(1, N, 1)

	x = x.view(batch_size * N, -1)
	d = d.view(batch_size * N, -1)

	# forward.
	rgb, sigma = func(x, d)

	rgb = rgb.view(batch_size, N, -1)
	sigma = sigma.view(batch_size, N, -1)

	delta = F.pad(t[:, 1:] - t[:, :-1], (0, 1), mode='constant', value=1e8)
	mass = sigma[..., 0] * delta
	mass = F.pad(mass, (1, 0), mode='constant', value=0.)

	alpha = 1. - torch.exp(- mass[:, 1:])
	T = torch.exp(- torch.cumsum(mass[:, :-1], dim=1))
	w = T * alpha
	return rgb, w


def volume_rendering_with_radiance_field(func_c, func_f, ray, t_n, t_f,
										 N_c, N_f, bg):
	"""Rendering with Neural Radiance Field.

	Args:
		func_c: NN for coarse rendering.
		func_f: NN for fine rendering.
		o (ndarray, [batch_size, 3]): Start points of the ray.
		d (ndarray, [batch_size, 3]): Directions of the ray.
		t_n (float): Start point of split.
		t_f (float): End point of split.
		N_c (int): num of coarse sampling.
		N_f (int): num of fine sampling.
		c_bg (tuple, [3,]): Background color.

	Returns:
		C_c (tensor, [batch_size, 3]): Result of coarse rendering.
		C_f (tensor, [batch_size, 3]): Result of fine rendering.

	"""
	batch_size = ray.shape[0]
	device = ray.device

	partitions = split_ray(t_n, t_f, N_c, batch_size)
	partitions = torch.tensor(partitions).to(ray)

	# coarse rendering:
	with torch.no_grad():
		t_c = sample_coarse(partitions)

	rgb_c, w_c = _rgb_and_weight(func_c, ray, t_c, N_c)
	C_c = torch.sum(w_c[..., None] * rgb_c, axis=1)
	C_c = C_c + (1. - torch.sum(w_c, axis=1, keepdims=True)) * bg

	# fine rendering.
	with torch.no_grad():
		t_f = sample_fine(partitions, w_c.detach(), t_c, N_f)

	rgb_f, w_f = _rgb_and_weight(func_f, ray, t_f.detach(), N_f+N_c)
	C_f = torch.sum(w_f[..., None] * rgb_f, axis=1)
	C_f = C_f + (1. - torch.sum(w_f, axis=1, keepdims=True)) * bg

	return C_c, C_f


def camera_params_to_rays(intrinsic, extrinsic, width, height):
	"""Make rays (o, d) from camera parameters.

	Args:
		f (float): A focal length.
		cx, xy (float): A center of the image.
		pose (ndarray, [4, 4]): camera extrinsic matrix.
		width(int): The height of the rendered image.
		height(int): The width of the rendered image.

	Returns:
		o (ndarray, [height, width, 3]): The origin of the camera coordinate.
		d (ndarray, [height, width, 3]): The direction of each ray.

	"""
	_o = np.zeros((height, width, 3), dtype=np.float32)

	v, u = np.mgrid[:height, :width].astype(np.float32)
	_d = np.dstack((v, u, np.ones_like(v))).reshape(-1,3)
	_d = (np.linalg.inv(intrinsic) @ _d.T).T.reshape(v.shape[0], v.shape[1], 3)

	R = extrinsic[:3,:3]
	t = extrinsic[:3,3]

	o = _o + t
	_d = (R @ _d.reshape(-1,3).T).T.reshape(_d.shape[0], _d.shape[1], 3) + t
	d = _d - o
	d /= np.linalg.norm(d, axis=2, keepdims=True)
	ray = np.concatenate((o, d), axis=-1)
	return ray


def _init_weights(m):
	if type(m) == nn.Linear:
		nn.init.kaiming_normal_(m.weight)
		nn.init.zeros_(m.bias)


class RadianceField(nn.Module):
	def __init__(self, 
		num_first_layers=4, num_second_layers=3, num_final_layers=3, latent_dim=256, 
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
		self.second_module = nn.Sequential(*layers)
		self.sigma_layer = nn.Sequential(
			nn.Linear(self.latent_dim, 1),
			nn.ReLU(),
		)
		self.third_module = nn.Sequential(
			nn.Linear(self.latent_dim, self.latent_dim),
			nn.ReLU(),
		)

		layers = [nn.Linear(self.encoding_length_dir * 6 + self.latent_dim, self.latent_dim // 2)]
		for i in range(num_final_layers):
			layers.append(nn.ReLU())
			layers.append(nn.Linear(self.latent_dim // 2, self.latent_dim // 2))
		layers.append(nn.ReLU())
		layers.append(nn.Linear(self.latent_dim // 2, 3))
		layers.append(nn.Sigmoid())

		self.final_module = nn.Sequential(*layers)

		self.apply(_init_weights)


	def forward(self, x, d):
#		x, d = inputs[:,:3], inputs[:,3:]
		x_enc = self.positional_encoding(x, self.encoding_length_pos)
		d_enc = self.positional_encoding(d, self.encoding_length_dir)

		v = self.first_module(x_enc)
		v = self.second_module(torch.cat([x_enc, v], dim=-1))
		sigma = self.sigma_layer(v)
		v = self.third_module(v)
		rgb = self.final_module(torch.cat([d_enc, v], dim=-1))

		return rgb, sigma


	def positional_encoding(self, p, L):
		p = torch.tanh(p)

		batch_size = p.shape[0]
		i = torch.arange(L, dtype=torch.float32, device=p.device)
		a = (2. ** i[None, None]) * np.pi * p[:, :, None]
		s = torch.sin(a)
		c = torch.cos(a)
		e = torch.cat([s, c], axis=2).view(batch_size, -1)
		return e



class NeRF(nn.Module):
	def __init__(self, N_c=32, N_f=64, t_n=0.0, t_f=2.5, L_x=10, L_d=4, c_bg=(1, 1, 1)):
		super().__init__()
		self.N_c = N_c
		self.N_f = N_f
		self.t_n = t_n
		self.t_f = t_f
		self.bg = nn.Parameter(torch.tensor(c_bg, dtype=torch.float32).view(1, 3), requires_grad=False)

		self.rf_c = RadianceField(
			encoding_length_pos=L_x, encoding_length_dir=L_d)
		self.rf_f = RadianceField(
			encoding_length_pos=L_x, encoding_length_dir=L_d)
		self.loss = nn.MSELoss()


	def device(self):
		return next(self.parameters()).device


	def sample(self, view, batch_size):
		ray = camera_params_to_rays(view['intrinsic'], view['extrinsic'], view['width'], view['height'])
		ray = ray.reshape(-1, 6)

		device = self.device()
		ray = torch.tensor(ray, device=device)

		_C_c = []
		_C_f = []
		with torch.no_grad():
			for i in range(0, ray.shape[0], batch_size):
				ray_i = ray[i:i+batch_size]
				C_c_i, C_f_i = self.forward(ray_i)
				_C_c.append(C_c_i.cpu().numpy())
				_C_f.append(C_f_i.cpu().numpy())

		C_c = np.concatenate(_C_c, axis=0)
		C_f = np.concatenate(_C_f, axis=0)
		C_c = np.clip(0., 1., C_c.reshape(view['height'], view['width'], 3))
		C_f = np.clip(0., 1., C_f.reshape(view['height'], view['width'], 3))

		return C_c, C_f


	def forward(self, inputs, targets=None):
		outputs_coarse, outputs_fine = volume_rendering_with_radiance_field(
			self.rf_c, self.rf_f, inputs, self.t_n, self.t_f,
			N_c=self.N_c, N_f=self.N_f, bg=self.bg)

		if targets is not None:
			loss = self.loss(outputs_coarse, targets) + self.loss(outputs_fine, targets)
			return loss
		else:
			return outputs_coarse, outputs_fine



class NeRFDataset:
	def __init__(self, frameList, batch_size, crop=None):
		self.frameList = frameList
		self.batch_size = batch_size
		self.crop = crop

		self.inputs = []
		self.targets = []

		for i, frame in enumerate(self.frameList):
			print("\rReading frames: %d/%d" % (i+1, len(self.frameList)), end="")
			rgb = read_frame(frame, crop=self.crop)
			ray = camera_params_to_rays(frame.intrinsic, frame.trans_inv, frame.width, frame.height)
			C = rgb[:, :, :3]

			ray = ray.reshape(-1, 6)
			C = C.reshape(-1, 3)

			self.inputs.append(ray)
			self.targets.append(C)
		print()

		self.inputs = np.concatenate(self.inputs, axis=0)
		self.targets = np.concatenate(self.targets, axis=0)
		perm = np.random.permutation(len(self.inputs))

		self.inputs = torch.FloatTensor(self.inputs[perm])
		self.targets = torch.FloatTensor(self.targets[perm])


	@property
	def shape(self):
		return (self.frameList[0].height, self.frameList[0].width, 3)


	def __len__(self):
		return len(self.inputs) // self.batch_size - 1


	def __getitem__(self, idx):
		return self.inputs[idx*self.batch_size:(idx+1)*self.batch_size], \
			self.targets[idx*self.batch_size:(idx+1)*self.batch_size]


@profile
def train(nerf, train_data, n_epoch):
	optimizer = torch.optim.AdamW(nerf.parameters(), lr=3e-4)
	scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=n_epoch * len(train_data))

	lossList = []
	fig_loss = plt.figure()
	ax_loss = fig_loss.add_subplot()
	fig_vis = plt.figure(figsize=(15, 5))
	ax_coarse = fig_vis.add_subplot(1, 3, 1)
	ax_fine = fig_vis.add_subplot(1, 3, 2)
	ax_original = fig_vis.add_subplot(1, 3, 3)

	for e in range(1, n_epoch+1):
		cum_loss = 0.0

		for i, (inputs, targets) in enumerate(train_data):
			inputs = inputs.cuda()
			targets = targets.cuda()

			loss = nerf(inputs, targets)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			cum_loss += loss.detach()
			if (i + 1) % 100 == 0:
				loss_value = cum_loss.item() / 100
				cum_loss = 0.0
				print("\r[Epoch %d/%d, Iter: %d/%d] loss: %.5f" % (e, n_epoch, i + 1, len(train_data), loss_value), end='')
				lossList.append(loss_value)
				ax_loss.clear()
				ax_loss.plot(np.arange(0, len(lossList)), np.exp(-np.log10(np.array(lossList))))
				fig_loss.savefig("loss.png")

			if (i + 1) % 1000 == 0:
				ind = (i // 1000) % len(frameList)
				frame = frameList[ind]
				rgb = read_frame(frame, crop=(crop_width, crop_height))

				# 512 * 512 はやや時間がかかるので半分のサイズでレンダリング
				half_intrinsic = frame.intrinsic / 2
				half_intrinsic[2,2] = 1.0
				view = {
					'intrinsic': half_intrinsic,
					'extrinsic': frame.trans_inv,
					'height': frame.height // 2,
					'width': frame.width // 2,
				}

				C_c, C_f = nerf.sample(view, batch_size)

				ax_coarse.clear()
				ax_coarse.set_title("Coarse")
				ax_coarse.imshow(C_c)
				ax_fine.clear()
				ax_fine.set_title("Fine")
				ax_fine.imshow(C_f)
				ax_original.clear()
				ax_original.set_title("Original")
				ax_original.imshow(rgb)
				fig_vis.savefig("nerf.png")


if __name__ == '__main__':
#	ray.init(num_cpus=16)
	"""
	dataset_path = 'dataset/greek/'

	with open(os.path.join(dataset_path, 'intrinsics.txt'), 'r') as file:
		f, cx, cy, _ = _line2floats(file.readline())
		_, _, _ = _line2floats(file.readline())
		_, = _line2floats(file.readline())
		_, = _line2floats(file.readline())
		img_height, img_width = _line2floats(file.readline())

	print('focal length: {}'.format(f))
	print('image center: ({}, {})'.format(cx, cy))
	print('image size: ({}, {})'.format(img_width, img_height))

	# データセットの画像サイズ．
	width = 512
	height = 512

	fx = f * width / img_width
	fy = f * height / img_height
	cx = cx * width / img_width
	cy = cy * height / img_height

	print('focal length: {}'.format(f))
	print('image center: ({}, {})'.format(cx, cy))
	print('image size: ({}, {})'.format(width, height))

	intrinsic = np.float32([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1]
	])


	pose_paths = sorted(glob.glob(dataset_path + 'pose/*.txt'))
	rgb_paths = sorted(glob.glob(dataset_path + 'rgb/*.png'))

	frameList = []

	for pose_path, rgb_path in zip(pose_paths, rgb_paths):
		trans_mat = np.genfromtxt(pose_path, dtype=np.float32).reshape(4, 4)
		trans_mat_inv = np.linalg.inv(trans_mat)

		frame = Frame(
			file_path=rgb_path,
			img=None,
			intrinsic=intrinsic,
			dist=np.zeros((4,), dtype=np.float32),
			trans=trans_mat_inv,
			trans_inv=trans_mat,
			rmat=trans_mat_inv[:3,:3],
			rmat_inv=trans_mat[:3,:3],
			rvec=cv2.Rodrigues(trans_mat_inv[:3,:3])[0],
			rvec_inv=cv2.Rodrigues(trans_mat[:3,:3])[0],
			tvec=trans_mat_inv[:3,3],
			tvec_inv=trans_mat[:3,3],
			width=width,
			height=height,
			mask=None
		)

		frameList.append(frame)
	"""

	data = json.load(open(os.path.join(args.dataset, "transforms_train.json")))
	frameList = read_transforms(args.dataset, data)

	fx = frameList[0].intrinsic[0,0]
	fy = frameList[0].intrinsic[1,1]
	cx = frameList[0].intrinsic[0,2]
	cy = frameList[0].intrinsic[1,2]
	img_width = frameList[0].width
	img_height = frameList[0].height

	crop_width = 1024
	crop_height = 1024
	cx -= (img_width - crop_width) / 2
	cy -= (img_height - crop_height) / 2

	width = 512
	height = 512

	fx = fx * width / crop_width
	fy = fy * height / crop_height
	cx = cx * width / crop_width
	cy = cy * height / crop_height

	print('focal length: ({}, {})'.format(fx, fy))
	print('image center: ({}, {})'.format(cx, cy))
	print('image size: ({}, {})'.format(width, height))

	intrinsic = np.float32([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1]
	])

	for fi in range(len(frameList)):
		frameList[fi].intrinsic = intrinsic
		frameList[fi].width = width
		frameList[fi].height = height
		frameList[fi].trans_inv[:3, 3] *= 0.1


	"""
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')
	# ax.set_xlim(-2, 2)
	# ax.set_ylim(-2, 2)
	# ax.set_zlim(-2, 2)
	ax.view_init(elev=90, azim=90)

	for frame in frameList[::16]:
		ray = camera_params_to_rays(frame.intrinsic, frame.trans_inv, frame.width, frame.height)
		o = ray[..., :3]
		d = ray[..., 3:]

		# 焦点（赤）
		o_x, o_y, o_z = o[0, :1].T
		ax.scatter(o_x, o_y, o_z, c='red')

		# レンダリング下限（青）
		t_n = .5
		x_n, y_n, z_n = (o + d * t_n)[::16, ::16].reshape(-1, 3).T
		ax.scatter(x_n, y_n, z_n, c='blue', s=0.1)

		# レンダリング上限（緑）
		t_f = 2.5
		x_f, y_f, z_f = (o + d * t_f)[::16, ::16].reshape(-1, 3).T
		ax.scatter(x_f, y_f, z_f, c='green', s=0.1)

	plt.show()
	"""

	n_epoch = 10
	batch_size = 512
	train_data = NeRFDataset(frameList, batch_size, crop=(crop_width, crop_height))

	nerf = NeRF(t_n=0., t_f=2.5, c_bg=(1, 1, 1)).cuda()

	train(nerf, train_data, n_epoch)
