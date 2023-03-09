from functools import lru_cache, partial
import sys, os, argparse, json, random
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import itertools

from common import *
import superglue as sg
import scipy.optimize
from pylsd.lsd import lsd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="dataset/10TK")
parser.add_argument('--feature', type=str, choices=['sift', 'orb', 'superglue'], default='superglue')
parser.add_argument('--scale', type=float, default=0.75)
args = parser.parse_args()


NUM_FEATURES = 1000
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6

if args.feature == 'sift':
	feature_detector = cv2.SIFT_create()
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = {"checks": 50}
	matcher = cv2.FlannBasedMatcher(index_params, search_params)

elif args.feature == 'orb':
	feature_detector = cv2.ORB_create(NUM_FEATURES)
	index_params= dict(algorithm = FLANN_INDEX_LSH,
						table_number = 6,
						key_size = 12,
						multi_probe_level = 1)
	search_params = {"checks": 50}
	matcher = cv2.FlannBasedMatcher(index_params, search_params)

elif args.feature == 'superglue':
	feature_detector = sg.SuperGlue()
	matcher = feature_detector

else:
	raise NotImplementedError()



def detect_feature_impl(frame, feature, rootsift):
	kp, des = feature_detector.detectAndCompute(frame.img, None)
	if rootsift and feature == 'sift':
		des /= (des.sum(axis=1, keepdims=True) + 1e-7)
		des = np.sqrt(des)
	return kp_to_tuple(kp), des


def detect_features(frameList, rootsift=True):
	for i, frame in enumerate(frameList):
		print("\rDetecting features: %d/%d" % (i + 1, len(frameList)), end="")
		kp, des = detect_feature_impl(frame, args.feature, rootsift)
		frameList[i].keypoints = tuple_to_kp(kp)
		frameList[i].descriptors = des

	return frameList


def detect_features_if_not_yet(frame, rootsift=True):
	if not hasattr(frame, 'kp') or hasattr(frame, 'des'):
		kp, des = detect_feature_impl(frame, args.feature, rootsift)
		frame.keypoints = tuple_to_kp(kp)
		frame.descriptors = des
	return frame


def compute_matches(frame1, frame2, ratio=0.7):
	if args.feature == 'superglue':
		return matcher.compute_matches(frame1, frame2)

	else:
		matches = matcher.knnMatch(frame1.descriptors , frame2.descriptors, k=2)
		matches = [m for m in matches if len(m) == 2]

		good_matches = []
		for m, n in matches:
			if m.distance < ratio * n.distance:
				good_matches.append([m])

		return good_matches


clicked_pos = {}


def click(event, x, y, flags, params, name, img):
	global clicked_pos

	vis_img = img.copy()
	if name not in clicked_pos:
		clicked_pos[name] = {}
	if event == cv2.EVENT_LBUTTONDOWN:
		clicked_pos[name]['begin'] = [x, y]
	elif event == cv2.EVENT_RBUTTONDOWN:
		clicked_pos[name]['end'] = [x, y]

	if 'begin' in clicked_pos[name] and 'end' in clicked_pos[name]:
		cv2.line(vis_img, clicked_pos[name]['begin'], clicked_pos[name]['end'], (0,255,0), 1)
	if 'begin' in clicked_pos[name]:
		cv2.circle(vis_img, clicked_pos[name]['begin'], 3, (255,0,0), -1)
	if 'end' in clicked_pos[name]:
		cv2.circle(vis_img, clicked_pos[name]['end'], 3, (0,0,255), -1)

	cv2.imshow(name, vis_img)


def select_lines(frame1, frame2):
	global clicked_pos

	cmap = plt.get_cmap()

#	edge1 = detect_edge(frame1.img)
#	edge2 = detect_edge(frame2.img)
#	lines1 = lsd(edge1)
#	lines2 = lsd(edge2)

	img1_resize = cv2.resize(frame1.img, None, fx=args.scale, fy=args.scale)
	img2_resize = cv2.resize(frame2.img, None, fx=args.scale, fy=args.scale)
	cmap = plt.get_cmap('tab20')
	lines_frame1 = []
	lines_frame2 = []

	while True:
		clicked_pos = {}
		cv2.imshow('img1', img1_resize)
		cv2.imshow('img2', img2_resize)
		cv2.setMouseCallback('img1', partial(click, name='img1', img=img1_resize))
		cv2.setMouseCallback('img2', partial(click, name='img2', img=img2_resize))
		key = cv2.waitKey(0) & 0xff
		cv2.destroyAllWindows()

		if 'img1' in clicked_pos and 'img2' in clicked_pos and \
			'begin' in clicked_pos['img1'] and 'end' in clicked_pos['img1'] and \
			'begin' in clicked_pos['img2'] and 'end' in clicked_pos['img2']:
			lines_frame1.append(np.float32([clicked_pos['img1']['begin'], clicked_pos['img1']['end']]))
			lines_frame2.append(np.float32([clicked_pos['img2']['begin'], clicked_pos['img2']['end']]))

			color = (np.array(cmap(len(lines_frame1) - 1))[:3] * 255).astype(np.uint8)[::-1]
			cv2.line(img1_resize, lines_frame1[-1][0].astype(np.int32), lines_frame1[-1][1].astype(np.int32), color.tolist(), 1)
			cv2.line(img2_resize, lines_frame2[-1][0].astype(np.int32), lines_frame2[-1][1].astype(np.int32), color.tolist(), 1)

		if key == ord('n') or key == ord('q'):
			break

	return key, np.array(lines_frame1) / args.scale, np.array(lines_frame2) / args.scale


def compute_line_correspondence(frame1, frame2, lines1, lines2, F, num_pts_in_line=10):
	assert len(lines1) == len(lines2), "Different number of lines were passed in compute_line_correspondence."

	lines1_ret = []
	lines2_ret = []
	for line1, line2 in zip(lines1, lines2):
		t = np.linspace(0, 1, num_pts_in_line)
		line1_pts = (line1[1] - line1[0])[np.newaxis,:] * t[:,np.newaxis] + line1[0]
		epilines = cv2.computeCorrespondEpilines(line1_pts.reshape(-1,1,2), 1, F).reshape(-1, 3)

		a, b, c = epilines[:,0], epilines[:,1], epilines[:,2]
		d = (line2[1,1] - line2[0,1]) / (line2[1,0] - line2[0,0])
		e = line2[0,1] - d * line2[0,0]
		x = - (c + b * e) / (a + b * d)
		y = -(x * a + c) / b
		line2_pts = np.dstack((x, y))[0]

		"""
		img1 = frame1.img.copy()
		for pt in line1_pts:
			cv2.circle(img1, (int(pt[0]), int(pt[1])), 3, (0,0,255), -1)	
		img2 = drawlines(frame2.img.copy(), epilines)
		for pt in line2_pts:
			cv2.circle(img2, (int(pt[0]), int(pt[1])), 3, (0,0,255), -1)	

		cv2.imshow("img1", cv2.resize(img1, None, fx=args.scale, fy=args.scale))
		cv2.imshow("img2", cv2.resize(img2, None, fx=args.scale, fy=args.scale))
		cv2.waitKey()
		"""

		lines1_ret.append(line1_pts)
		lines2_ret.append(line2_pts)

	return np.array(lines1_ret), np.array(lines2_ret)


if __name__ == '__main__':
	data = json.load(open(os.path.join(args.dataset, "transforms_test.json")))
	frameList = read_transforms(args.dataset, data)

	comb_list = list(itertools.combinations(range(len(frameList)), 2))
	random.seed(0)
	random.shuffle(comb_list)

	FList = []
	linesList1 = []
	linesList2 = []

	while True:
		index1, index2 = comb_list.pop(0)
		frame1 = frameList[index1]
		frame2 = frameList[index2]
		frame1 = detect_features_if_not_yet(frameList[index1])
		frame2 = detect_features_if_not_yet(frameList[index2])
		matches = compute_matches(frame1, frame2, ratio=0.7)
		print(index1, index2)
		if len(matches) < 100:
			continue

#		visualize_matches(frame1, frame2, matches)
		F, mask = compute_fundamental(frame1, frame2, matches)

		key, lines1, lines2 = select_lines(frame1, frame2)
		lines1, lines2 = compute_line_correspondence(frame1, frame2, lines1, lines2, F, num_pts_in_line=2)
		FList.append(F)
		linesList1.append(lines1)
		linesList2.append(lines2)


		def cost_fun(params, FList, linesList1, linesList2, cx, cy):
			K = np.float32([
				[params[0], 0, cx],
				[0, params[0], cy],
				[0, 0, 1]
			])

			errorList = []
			for F, lines1, lines2 in zip(FList, linesList1, linesList2):
				E = K.T @ F @ K
				n_points, R, t, mask = cv2.recoverPose(E, lines1.reshape(-1,2), lines2.reshape(-1,2), K)
				extrinsic1 = to_extrinsic(np.eye(3, 3), np.zeros((3, 1)))
				extrinsic2 = to_extrinsic(R, t)
				proj1 = K @ extrinsic1[:3]
				proj2 = K @ extrinsic2[:3]
				point_4d_hom = cv2.triangulatePoints(proj1, proj2, lines1.reshape(-1,1,2), lines2.reshape(-1,1,2))
				euclid_points = cv2.convertPointsFromHomogeneous(point_4d_hom.T).reshape(lines1.shape[0], lines1.shape[1], 3)

				line_vecs = euclid_points[:,-1] - euclid_points[:,0]
				line_vecs /= np.linalg.norm(line_vecs, axis=1)[:,np.newaxis]

				for vec1, vec2 in itertools.combinations(line_vecs, 2):
					angle = np.degrees(np.arccos(np.dot(vec1, vec2)))
					error = min(np.abs(angle), np.abs(angle - 90), np.abs(angle - 180))
					errorList.append(error)

			return np.sum(errorList)


		x0 = np.float32([frame1.width])
		angle_minmax = np.float32([179, 1])
		f_minmax = frame1.width / (2 * np.tan(np.radians(angle_minmax) / 2))
		res = scipy.optimize.minimize(cost_fun, x0, method='Nelder-Mead', 
			options=dict(maxiter=int(1e4), disp=True),
			bounds=[f_minmax],
			args=(FList, linesList1, linesList2, frame1.width / 2, frame1.height / 2))
		print(res)

		if key == ord('q'):
			break
