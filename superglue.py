import cv2
import torch
import numpy as np

from SuperGlue.matching import Matching

torch.set_grad_enabled(False)


default_config = {
	'superpoint': {
		'nms_radius': 4,
		'keypoint_threshold': 0.005,
		'max_keypoints': -1
	},
	'superglue': {
		'weights': "outdoor",
		'sinkhorn_iterations': 20,
		'match_threshold': 0.2,
	}
}


def to_tensor(img):
	return torch.FloatTensor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[np.newaxis,np.newaxis,:,:] / 255).cuda()


class SuperGlue:
	def __init__(self, config=default_config):
		self.matching = Matching(config).eval().cuda()


	def detectAndCompute(self, img, mask=None):
		pred = self.matching.superpoint({'image': to_tensor(img)})
		keypoints = pred['keypoints'][0].cpu().numpy()
		kpList = [cv2.KeyPoint(x=kp[0].item(), y=kp[1].item(), size=1) for kp in keypoints]
		return kpList, pred


	def compute_matches(self, frame1, frame2):
		data = {
			"image0": to_tensor(frame1.img),
			"image1": to_tensor(frame2.img),
			**{k + '0': v for k, v in frame1.descriptors.items()},
			**{k + '1': v for k, v in frame2.descriptors.items()},
		}

		pred = self.matching(data)
		results = []

		matches = pred["matches0"][0].cpu().numpy()
		scores = pred["matching_scores0"][0].cpu().numpy()
		results = [[cv2.DMatch(i, index, score)] for i, (index, score) in enumerate(zip(matches, scores)) if index > 0]

		return results
