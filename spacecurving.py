from functools import lru_cache
import sys, os, argparse, json
import numpy as np
import cv2
import open3d as o3d

from common import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="dataset/10TK")
parser.add_argument('--num_division', type=int, default=4)
parser.add_argument('--min_voxel_size', type=float, default=0.1)
parser.add_argument('--num_frames', type=int)
parser.add_argument('--output', type=str, default="results/10TK/spacecurving.ply")
args = parser.parse_args()



@lru_cache
def get_pos_list(size, division):
	x = np.linspace(-size / 2, size / 2, division)
	y = np.linspace(-size / 2, size / 2, division)
	z = np.linspace(-size / 2, size / 2, division)
	return np.array(np.meshgrid(x, y, z)).reshape(3,-1).T


def is_in_all_frames(voxel, frameList):
	r = voxel.size / 2
	pts = np.array([
		voxel.pos + np.float32([-r, -r, -r]),
		voxel.pos + np.float32([-r, -r, r]),
		voxel.pos + np.float32([-r, r, -r]),
		voxel.pos + np.float32([-r, r, r]),
		voxel.pos + np.float32([r, -r, r]),
		voxel.pos + np.float32([r, r, -r]),
		voxel.pos + np.float32([r, -r, -r]),
		voxel.pos + np.float32([r, r, r]),
	]).reshape(-1,1,3)

	for frame in frameList:
		imgpts, _ = cv2.projectPoints(pts, frame.rvec_inv, frame.tvec_inv, frame.intrinsic, frame.dist)
		if len(imgpts) == 0:
			return False

		hull = cv2.convexHull(imgpts.astype(np.int32))
		proj_mask = np.zeros((frame.height, frame.width), dtype=np.uint8)
		proj_mask = cv2.drawContours(proj_mask, [hull], 0, 255, -1) > 0

		if (not np.any(proj_mask)) or (not np.any(proj_mask & frame.mask)):
			return False

	return True


def curve_voxel(voxelList, frameList):
	voxelList_ret = []

	while len(voxelList) > 0:
		voxel = voxelList.pop(0)

		if is_in_all_frames(voxel, frameList):
			if voxel.size <= args.min_voxel_size:
				voxelList_ret.append(voxel)
			else:
				pos3DList = get_pos_list(voxel.size, args.num_division) + voxel.pos
				voxelList += [Voxel(pos=pt, size=voxel.size / args.num_division, level=voxel.level+1) for pt in pos3DList]

		print("\rNumber of unprocessed voxels: %d" % len(voxelList), end="")
	print()

	return voxelList_ret


if __name__ == '__main__':
	data = json.load(open(os.path.join(args.dataset, "transforms_test.json")))
	frameList = read_transforms(args.dataset, data)
	if args.num_frames is not None:
		frameList = frameList[:args.num_frames]

	tvecList = np.array([frame.tvec_inv for frame in frameList])

	max_size = np.median(np.linalg.norm(tvecList, axis=1))
	pos3DList = get_pos_list(max_size * 2, 10)

	voxelList = []
	for pt in pos3DList:
		voxelList.append(Voxel(pos=pt, size=max_size * 2 / 10, level=1))

	voxelList = curve_voxel(voxelList, frameList)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector([voxel.pos for voxel in voxelList])
	voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxelList[0].size)

#	o3d.io.write_point_cloud(args.output, pcd)
	o3d.io.write_voxel_grid(args.output, voxel)
	o3d.visualization.draw_geometries([voxel])
