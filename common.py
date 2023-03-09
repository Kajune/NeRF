from dataclasses import dataclass
from functools import lru_cache
import os, copy
import numpy as np
import networkx as nx
import itertools
import cv2



@dataclass
class Voxel:
	pos : np.ndarray
	size : float
	level : int



@dataclass
class Frame:
	file_path : str
	img : np.ndarray
	intrinsic : np.ndarray
	dist : np.ndarray
	trans : np.ndarray
	trans_inv : np.ndarray
	rmat : np.ndarray
	rmat_inv : np.ndarray
	rvec : np.ndarray
	rvec_inv : np.ndarray
	tvec : np.ndarray
	tvec_inv : np.ndarray
	width : int
	height : int
	mask : np.ndarray



def read_transforms(root, transforms):
	cam_mat = np.float32([
		[transforms["fl_x"], 	0, 					transforms["cx"]],
		[0, 					transforms["fl_y"], transforms["cy"]],
		[0,						0,					1],
	])

	cam_dist = np.float32([
		transforms["k1"], transforms["k2"], transforms["p1"], transforms["p2"]
	])

	frameList = []
	for i, frame in enumerate(transforms["frames"]):
		print("\rReading frame: %d/%d" % (i+1, len(transforms["frames"])), end="")
		trans_mat = np.float32(frame["transform_matrix"])

		bl2cv = np.diag([1,-1,-1])
		R_bl = trans_mat[:3,:3].T
		T_bl = -1.0 * R_bl @ trans_mat[:3,3]
		R = bl2cv @ R_bl
		T = bl2cv @ T_bl
		trans_mat[:3,:3] = R
		trans_mat[:3,3] = T

		trans_mat_inv = np.linalg.inv(trans_mat)

		img = cv2.imread(os.path.join(root, frame["file_path"]))
		mask = cv2.imread(os.path.join(root, frame["file_path"]), -1)[...,3] > 0

		frameList.append(Frame(
			file_path=os.path.join(root, frame["file_path"]),
			img=img,
			intrinsic=cam_mat,
			dist=cam_dist,
			trans=trans_mat,
			trans_inv=trans_mat_inv,
			rmat=trans_mat[:3,:3],
			rmat_inv=trans_mat_inv[:3,:3],
			rvec=cv2.Rodrigues(trans_mat[:3,:3])[0],
			rvec_inv=cv2.Rodrigues(trans_mat_inv[:3,:3])[0],
			tvec=trans_mat[:3,3],
			tvec_inv=trans_mat_inv[:3,3],
			width=int(transforms["w"]),
			height=int(transforms["h"]),
			mask=mask
		))

	print()

	return frameList


def is_in_img(pt, frame):
	return 0 <= pt[0] < frame.width and 0 <= pt[1] < frame.height


def min_max(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))


def kp_to_tuple(kps):
	return [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kps]


def tuple_to_kp(kps):
	return [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=p[1], angle=p[2],
						response=p[3], octave=p[4], class_id=p[5]) for p in kps]


def to_extrinsic(R, tvec):
	extrinsic = np.zeros((4,4), dtype=np.float32)
	extrinsic[:3,:3] = R
	extrinsic[:3,3] = tvec.squeeze()
	return extrinsic


def visualize_matches(frame1, frame2, matches):
	img1 = cv2.imread(frame1.file_path)
	img2 = cv2.imread(frame2.file_path)

	match_img = cv2.drawMatchesKnn(img1, frame1.keypoints, img2, frame2.keypoints, matches, None, flags=2)
	cv2.imshow("", cv2.resize(match_img, None, fx=0.4, fy=0.4))
	cv2.waitKey()
	cv2.destroyAllWindows()


def compute_homography(frame1, frame2, matches):
	src_pts = np.float32([frame1.keypoints[m[0].queryIdx].pt for m in matches]).reshape(-1,1,2)
	dst_pts = np.float32([frame2.keypoints[m[0].trainIdx].pt for m in matches]).reshape(-1,1,2)
	H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

	dst_pts_reproj = cv2.perspectiveTransform(src_pts, H)
	rmse = np.sqrt(np.mean(np.sum((dst_pts_reproj - dst_pts)[mask] ** 2, axis=-1)))

	return H, mask, rmse


def get_paired_2D_points(frame1, frame2, matches):
	src_pts = np.float32([frame1.keypoints[m[0].queryIdx].pt for m in matches]).reshape(-1,1,2)
	dst_pts = np.float32([frame2.keypoints[m[0].trainIdx].pt for m in matches]).reshape(-1,1,2)
	src_pts = cv2.undistortPoints(src_pts, cameraMatrix=np.eye(3), distCoeffs=frame1.dist)
	dst_pts = cv2.undistortPoints(dst_pts, cameraMatrix=np.eye(3), distCoeffs=frame2.dist)

	return src_pts, dst_pts


def compute_pose_5points(frame1, frame2, matches):
	assert np.all(frame1.intrinsic == frame2.intrinsic) and np.all(frame1.dist == frame2.dist), \
		"Two cameras must have same intrinsic parameters"

	src_pts, dst_pts = get_paired_2D_points(frame1, frame2, matches)
	E, mask = cv2.findEssentialMat(src_pts, dst_pts, frame1.intrinsic)
	n_points, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, frame1.intrinsic)

	return R, t, mask


def compute_fundamental(frame1, frame2, matches):
	src_pts, dst_pts = get_paired_2D_points(frame1, frame2, matches)
	F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)
	return F, mask


def drawlines(img, lines):
	c = img.shape[1]
	for r in lines:
		color = tuple(np.random.randint(0, 255, 3).tolist())
		x0, y0 = map(int, [0, -r[2]/r[1] ])
		x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
	return img


def detect_edge(img, th1=2, th2=8):
	img = cv2.GaussianBlur(img, (1, 1), 3)
	return cv2.Canny(img, threshold1=th1, threshold2=th2)


def triangulate_points(frame1, frame2, extrinsic1, extrinsic2, matches):
	src_pts, dst_pts = get_paired_2D_points(frame1, frame2, matches)

	proj1 = frame1.intrinsic @ extrinsic1[:3]
	proj2 = frame2.intrinsic @ extrinsic2[:3]

	point_4d_hom = cv2.triangulatePoints(proj1, proj2, src_pts, dst_pts)
	euclid_points = cv2.convertPointsFromHomogeneous(point_4d_hom.T).squeeze()

	return euclid_points


def minimum_multi_edges_cut(G, nodes_to_partition, weight="weight"):
	min_connectivity = 1
	# min_connectivityは実際のところほとんど1だった
#	for node_pair in nodes_to_partition:
#		min_connectivity = max(min_connectivity, nx.node_connectivity(G, *node_pair))

	edge_list = list(G.edges())
	edge_list.sort(key=lambda x: G.edges[x][weight])

	best_cut = None
	G_tmp = G.copy()

	for k in range(min_connectivity, len(edge_list)):
		for cut_edges in itertools.combinations(edge_list, k):
			edges_with_weight = [(x[0], x[1], G.edges[x][weight]) for x in cut_edges]
			G_tmp.remove_edges_from(cut_edges)

			isOK = True
			for node_pair in nodes_to_partition:
				if nx.has_path(G_tmp, *node_pair):
					isOK = False
					break

			G_tmp.add_weighted_edges_from(edges_with_weight)

			if isOK:
				best_cut = cut_edges
				break

		if best_cut is not None:
			break

	return best_cut
