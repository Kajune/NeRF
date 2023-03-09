from functools import lru_cache
import sys, os, argparse, json, random
import numpy as np
import cv2
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import ray

from common import *
from bundle_adjustment import *
import superglue as sg


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="dataset/10TK")
parser.add_argument('--feature', type=str, choices=['sift', 'orb', 'superglue'], default='superglue')
parser.add_argument('--output', type=str, default="results/10TK/sfm.ply")
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


if __name__ == '__main__':
	data = json.load(open(os.path.join(args.dataset, "transforms_test.json")))
	frameList = read_transforms(args.dataset, data)


	ba = BundleAdjustment()
	for i, frame in enumerate(frameList):
		ba.add_frame(i, frame)


	comb_not_ready = list(itertools.combinations(range(len(frameList)), 2))
	comb_ready = []

	if os.path.exists("bad_list.txt"):
		bad_list = [tuple([int(a) for a in line.replace("\n", "").split(" ")]) for line in open("bad_list.txt", "r")]
		comb_not_ready = [comb for comb in comb_not_ready if comb not in bad_list]

	f_bad_list = open("bad_list.txt", "a")

	count = 0
	while len(comb_not_ready) > 0 or len(comb_ready) > 0:
		if len(comb_ready) == 0:
			if not ba.initialized_first_camera_pose:
				comb_ready.append(comb_not_ready.pop(0))
			else:
				print("Number of ready to process combinations is zero. Terminating")
				break

		index1, index2 = comb_ready.pop(0)
		print("Number of combinations: (%d / %d)" % (len(comb_ready), len(comb_not_ready)))
		print("Processing image pair:", index1, index2)

		frame1 = detect_features_if_not_yet(frameList[index1])
		frame2 = detect_features_if_not_yet(frameList[index2])
		matches = compute_matches(frame1, frame2, ratio=0.7)
		if len(matches) < 100:
			f_bad_list.write("%d %d\n" % (index1, index2))
			continue

		"""
		R, tvec, mask = compute_pose_5points(frame1, frame2, matches)
		points3d = triangulate_points(frame1, frame2, 
			to_extrinsic(np.eye(3, 3), np.zeros((3, 1))), to_extrinsic(R, tvec), matches)
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points3d)
		o3d.io.write_point_cloud("tmp.ply", pcd)
		"""

#		visualize_matches(frame1, frame2, matches)

		print("Adding matches")
		success = ba.add_matches(index1, index2, matches)
		if success:
			comb_to_move = []
			for comb in comb_not_ready:
				if index1 in comb or index2 in comb:
					comb_to_move.append(comb)
			comb_ready += comb_to_move
			comb_not_ready = [comb for comb in comb_not_ready if comb not in comb_ready]

		if (count + 1) % 10 == 0:
			cost = ba.optimize()
		count += 1

#		ba.visualize()

		o3d.io.write_point_cloud(args.output, ba.get_pcd())
