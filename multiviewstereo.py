from functools import lru_cache
import sys, os, argparse, json, random
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

from common import *
from psmnet import PSMNet
from raft import RAFT



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="dataset/10TK")
parser.add_argument('--stereo', type=str, choices=["sgbm", "psmnet", "raft"], default="sgbm")
parser.add_argument('--max_disp', type=int, default=128)
parser.add_argument('--output', type=str, default="results/10TK/multiviewstereo.ply")
args = parser.parse_args()


if args.stereo == "psmnet":
	psmnet = PSMNet("PSMNet/pretrained_sceneflow_new.tar", args.max_disp)
	raft = None

elif args.stereo == "raft":
	psmnet = None
	raft = RAFT()

else:
	psmnet = None
	raft = None


def enumerate_frame_pairs(frameList, max_angle=30, min_baseline=3.0):
	frame_pair_list = []

	for i in range(len(frameList)):
		for j in range(i+1, len(frameList)):
			frame1 = frameList[i]
			frame2 = frameList[j]

			unit_vec = np.float64([0, 0, 1])
			frame1_vec = np.dot(frame1.rmat, unit_vec)
			frame2_vec = np.dot(frame2.rmat, unit_vec)
			frame1_vec /= np.linalg.norm(frame1_vec)
			frame2_vec /= np.linalg.norm(frame2_vec)
			angle = np.degrees(np.arccos(np.dot(frame1_vec, frame2_vec)))
			baseline = np.linalg.norm(frame1.tvec_inv - frame2.tvec_inv)

			if angle <= max_angle and baseline >= min_baseline:
				frame_pair_list.append([i, j])

	return frame_pair_list


def rectify_frames(frame1, frame2, img1, img2):
	image_size = (frame1.width, frame1.height)
	trans_mat = np.dot(frame2.trans, frame1.trans_inv)
	R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(frame1.intrinsic, frame1.dist, frame2.intrinsic, frame2.dist, 
		image_size, cv2.Rodrigues(trans_mat[:3,:3])[0], trans_mat[:3,3], flags=0)

	if np.abs(P2[1,3]) > 0:
		# Skip vertical stereo
		return img1, img2, frame1.mask, frame2.mask, None, None, None, None, None

	map1_x, map1_y = cv2.initUndistortRectifyMap(frame1.intrinsic, frame1.dist, R1, P1, image_size, m1type=cv2.CV_32FC1)
	map2_x, map2_y = cv2.initUndistortRectifyMap(frame2.intrinsic, frame2.dist, R2, P2, image_size, m1type=cv2.CV_32FC1)

	img1_rect = cv2.remap(img1, map1_x, map1_y, interpolation=cv2.INTER_LINEAR)
	img2_rect = cv2.remap(img2, map2_x, map2_y, interpolation=cv2.INTER_LINEAR)
	mask1_rect = cv2.remap(frame1.mask.astype(np.float32), map1_x, map1_y, interpolation=cv2.INTER_LINEAR)
	mask2_rect = cv2.remap(frame2.mask.astype(np.float32), map2_x, map2_y, interpolation=cv2.INTER_LINEAR)

	return img1_rect, img2_rect, mask1_rect, mask2_rect, Q, R1, R2, P1, P2


def compute_disparity_and_points(img1_rect, img2_rect, Q, image_scale=0.5):
	if image_scale != 1.0:
		img1_rect = cv2.resize(img1_rect, None, fx=image_scale, fy=image_scale)
		img2_rect = cv2.resize(img2_rect, None, fx=image_scale, fy=image_scale)

	if args.stereo == "sgbm":
		window_size = 3
		max_disp = img1_rect.shape[1] // 8
		stereo = cv2.StereoSGBM_create(
			minDisparity = -max_disp,
			numDisparities = max_disp * 2,
			blockSize = 16,
			P1 = 8*3*window_size**2,
			P2 = 32*3*window_size**2,
			disp12MaxDiff = 1,
			uniquenessRatio = 10,
			speckleWindowSize = 100,
			speckleRange = 32
		)
		disparity = stereo.compute(img1_rect, img2_rect).astype(np.float32) / 16.0
		valid_disparity = disparity >= -max_disp

	elif args.stereo == "psmnet":
		division = 2
		h = img1_rect.shape[0] // division

		"""
		offset = 100
		img2_rect_ = np.zeros_like(img2_rect)
		img2_rect_[:,:-offset] = img2_rect[:,offset:]
		img2_rect = img2_rect_
		"""

		disparity = np.vstack([psmnet.predict(img1_rect[i*h:i*h+h], img2_rect[i*h:i*h+h]) for i in range(division)])
		disparity -= offset
		valid_disparity = np.ones_like(disparity, dtype=np.bool_)

	elif args.stereo == "raft":
		flow = raft.predict(img1_rect, img2_rect)
		disparity = flow[0]
		valid_disparity = np.ones_like(disparity, dtype=np.bool_)

	else:
		raise NotImplementedError()

	if image_scale != 1.0:
		disparity = cv2.resize(disparity, None, fx=1/image_scale, fy=1/image_scale, interpolation=cv2.INTER_LINEAR) / image_scale
		valid_disparity = cv2.resize(valid_disparity.astype(np.uint8), None, fx=1/image_scale, fy=1/image_scale, interpolation=cv2.INTER_NEAREST).astype(np.bool_)

	return disparity, valid_disparity, cv2.reprojectImageTo3D(disparity, Q)


def min_max(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))


def is_in_other_frames(points, frameList):
	mask = np.ones((points.shape[0],1,), dtype=np.bool_)

	for frame in random.sample(frameList, 10):
		imgpts, _ = cv2.projectPoints(points, frame.rvec, frame.tvec, frame.intrinsic, frame.dist)

		imgpts = imgpts.astype(np.int32)
		is_in_img = (0 <= imgpts[...,0]) & (imgpts[...,0] < frame.width) & (0 <= imgpts[...,1]) & (imgpts[...,1] < frame.height)
		imgpts[~is_in_img] = 0
		is_in_mask = frame.mask[imgpts[...,1], imgpts[...,0]]

		mask = mask & is_in_img & is_in_mask

	return mask.squeeze()


if __name__ == '__main__':
	data = json.load(open(os.path.join(args.dataset, "transforms_test.json")))
	frameList = read_transforms(args.dataset, data)

	frame_pair_list = enumerate_frame_pairs(frameList)

	pcd = o3d.geometry.PointCloud()
	volume = o3d.pipelines.integration.ScalableTSDFVolume(
		voxel_length=1/64,
		sdf_trunc=0.04,
		color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
	mesh = o3d.geometry.TriangleMesh()

#	vis = o3d.visualization.Visualizer()
#	vis.create_window()

	for index, (i, j) in enumerate(frame_pair_list):
		print("\rStereo Reconstruction: %d/%d" % (index+1, len(frame_pair_list)), end="")
		frame1 = frameList[i]
		frame2 = frameList[j]

		img1 = cv2.imread(frame1.file_path)
		img2 = cv2.imread(frame2.file_path)

		img1_rect, img2_rect, mask1_rect, mask2_rect, Q, R1, R2, P1, P2 = rectify_frames(frame1, frame2, img1, img2)
		if Q is None:
			continue

		disparity, valid_disparity, points = compute_disparity_and_points(img1_rect, img2_rect, Q)
		valid_points = (mask1_rect > 0) & valid_disparity
		points[...,2][~valid_points] = 0

		points_ = (frame1.rmat_inv @ np.linalg.inv(R1) @ points.reshape(-1,3).T).T + frame1.tvec_inv
		if len(points_) == 0:
			continue

#		silhouette_mask = is_in_other_frames(points_.reshape(-1,3), frameList).reshape(points.shape[0], points.shape[1])
#		points[...,2][~silhouette_mask] = 0

		"""
		disp_vis = cv2.cvtColor(min_max(disparity) * 255 * mask1_rect, cv2.COLOR_GRAY2BGR).astype(np.uint8)
		depth_vis = cv2.cvtColor(min_max(points[...,2]) * 255 * mask1_rect, cv2.COLOR_GRAY2BGR).astype(np.uint8)
		vis = np.vstack((np.hstack((img1, img2)), np.hstack((img1_rect, img2_rect)), np.hstack((disp_vis, depth_vis))))
		cv2.imshow("", cv2.resize(vis, None, fx=0.25, fy=0.25))
		cv2.waitKey()
		"""

		#
		# TSDF Integration
		#
		extrinsic = np.zeros((4,4), dtype=np.float64)
		extrinsic[:3,:3] = frame1.rmat_inv @ np.linalg.inv(R1)
		extrinsic[:3,3] = frame1.tvec_inv
		extrinsic[3,3] = 1

		color = o3d.geometry.Image(np.ascontiguousarray(img1_rect[...,::-1]))
		max_depth = 20
		depth = o3d.geometry.Image(points[...,2] / max_depth)
		rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
			color, depth, depth_scale=1/max_depth, depth_trunc=1e10, convert_rgb_to_intensity=False)

		"""
		pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
			rgbd,
			o3d.camera.PinholeCameraIntrinsic(frame1.width, frame1.height, P1[:3,:3]),
			np.linalg.inv(extrinsic)
		)
		o3d.io.write_point_cloud(args.output, pcd)
		"""

		volume.integrate(
			rgbd,
			o3d.camera.PinholeCameraIntrinsic(frame1.width, frame1.height, P1[:3,:3]),
			np.linalg.inv(extrinsic)
		)

		cur_mesh = volume.extract_triangle_mesh()
		cur_mesh.compute_vertex_normals()
#		o3d.visualization.draw_geometries([mesh])
		o3d.io.write_triangle_mesh(args.output, cur_mesh)

		"""
		vis.remove_geometry(mesh, False)
		vis.add_geometry(cur_mesh, index == 0)
		mesh = cur_mesh

		vis.poll_events()
		vis.update_renderer()
		"""

		#
		# Raw point extraction
		#
		"""
		points = points[valid_points]
		points = (frame1.rmat_inv @ np.linalg.inv(R1) @ points.T).T + frame1.tvec_inv
		colors = img1_rect[valid_points].reshape(-1,3)[...,::-1] / 255

		if len(points) == 0:
			continue

		silhouette_mask = is_in_other_frames(points, frameList)
		points = points[silhouette_mask]
		colors = colors[silhouette_mask]

		pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.points), points]))
		pcd.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.colors), colors]))

		o3d.io.write_point_cloud(args.output, pcd)
		"""

		# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
		# o3d.visualization.draw_geometries([pcd, mesh_frame])
