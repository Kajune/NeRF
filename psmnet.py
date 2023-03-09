import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import math
from PSMNet.models import *
import cv2
from PIL import Image


normal_mean_var = {'mean': [0.485, 0.456, 0.406],
					'std': [0.229, 0.224, 0.225]}
infer_transform = transforms.Compose([transforms.ToTensor(),
									  transforms.Normalize(**normal_mean_var)])


class PSMNet:
	def __init__(self, model_path, maxdisp):
		self.model = stackhourglass(maxdisp)
		self.model = nn.DataParallel(self.model, device_ids=[0])
		self.model.cuda()

		state_dict = torch.load(model_path)
		self.model.load_state_dict(state_dict['state_dict'])
		self.model.eval()


	def predict(self, imgL, imgR):
		imgL = infer_transform(Image.fromarray(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)))
		imgR = infer_transform(Image.fromarray(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)))

		if imgL.shape[1] % 16 != 0:
			times = imgL.shape[1]//16
			top_pad = (times+1)*16 -imgL.shape[1]
		else:
			top_pad = 0

		if imgL.shape[2] % 16 != 0:
			times = imgL.shape[2]//16
			right_pad = (times+1)*16-imgL.shape[2]
		else:
			right_pad = 0

		imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
		imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

		imgL = imgL.cuda()
		imgR = imgR.cuda()

		with torch.no_grad():
			disp = self.model(imgL, imgR)

		disp = torch.squeeze(disp)
		pred_disp = disp.data.cpu().numpy()

		if top_pad !=0 and right_pad != 0:
			pred_disp = pred_disp[top_pad:,:-right_pad]
		elif top_pad ==0 and right_pad != 0:
			pred_disp = pred_disp[:,:-right_pad]
		elif top_pad !=0 and right_pad == 0:
			pred_disp = pred_disp[top_pad:,:]

		return pred_disp



if __name__ == '__main__':
	psmnet = PSMNet("PSMNet/pretrained_sceneflow_new.tar", 96)

	left = cv2.imread("PSMNet/left.png")
	right = cv2.imread("PSMNet/right.png")
	disp = psmnet.predict(left, right)

	cv2.imwrite("PSMNet/result.png", disp * 255 / np.max(disp))






