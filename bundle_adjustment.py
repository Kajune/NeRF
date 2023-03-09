from functools import lru_cache
import sys, os, argparse, json, random
import numpy as np
import cv2
import open3d as o3d
import networkx as nx
import itertools

from common import *
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


class BundleAdjustment:
	def __init__(self):
		self.camera_graph = nx.Graph()
		self.point_graph = nx.Graph()
		self.initialized_first_camera_pose = False


	def add_frame(self, index, frame):
		self.camera_graph.add_node(index)
		self.camera_graph.nodes[index]['frame'] = frame


	def add_matches(self, index1, index2, matches):
		self.camera_graph.add_edge(index1, index2)
		self.camera_graph.edges[index1, index2]['matches'] = matches

		frame1 = self.camera_graph.nodes[index1]['frame']
		frame2 = self.camera_graph.nodes[index2]['frame']

		success = True
		if not hasattr(frame1, 'extrinsic') and not hasattr(frame2, 'extrinsic'):
			success = self._compute_pose_5points(frame1, frame2, matches)

		elif hasattr(frame1, 'extrinsic') and not hasattr(frame2, 'extrinsic'):
			success = self._compute_pose_pnp(frame1, frame2, matches, index1, index2, inverse=False)

		elif not hasattr(frame1, 'extrinsic') and hasattr(frame2, 'extrinsic'):
			success = self._compute_pose_pnp(frame1, frame2, matches, index1, index2, inverse=True)

		for i, m in enumerate(matches):
			pt_index1 = (index1, m[0].queryIdx)
			pt_index2 = (index2, m[0].trainIdx)
			self.point_graph.add_edge(pt_index1, pt_index2, weight=m[0].distance)
			self.point_graph.nodes[pt_index1]["pos2D"] = frame1.keypoints[m[0].queryIdx].pt
			self.point_graph.nodes[pt_index2]["pos2D"] = frame2.keypoints[m[0].trainIdx].pt

			self.remove_misconnected_point_edge_iterative(pt_index1)

		self._retrieve_colors(frame1, frame2, matches, index1, index2)
		if success:
			self._triangulate_points(frame1, frame2, matches, index1, index2)

		return success


	def optimize(self):
		def cost_fun(params, n_points, camera_indices, point_indices, points_2d, intrinsics, dists):
			n_cameras = len(intrinsics)
			extrinsics = params[:n_cameras * 6].reshape((n_cameras, 6))
			points_3d = params[n_cameras * 6:].reshape((n_points, 3))
			points_proj = np.zeros_like(points2d)
			for ci in np.unique(camera_indices):
				rvec = extrinsics[ci,:3].reshape((3,1))
				tvec = extrinsics[ci,3:].reshape((3,1))
				objpoints = np.array([points3d[point_indices[camera_indices == ci]]])
				imgpoints_reproj, _ = cv2.projectPoints(objpoints, rvec, tvec, intrinsics[ci], dists[ci])
				points_proj[camera_indices == ci] = imgpoints_reproj.squeeze()

			return (points_proj - points_2d).ravel() / len(points2d)

		def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
			m = camera_indices.size * 2
			n = n_cameras * 6 + n_points * 3
			A = lil_matrix((m, n), dtype=int)

			i = np.arange(camera_indices.size)
			for s in range(6):
				A[2 * i, camera_indices * 6 + s] = 1
				A[2 * i + 1, camera_indices * 6 + s] = 1

			for s in range(3):
				A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
				A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

			return A

		intrinsics = []
		extrinsics = []
		dists = []
		frame_indices = []
		for frame_index, data in self.camera_graph.nodes(data=True):
			if not hasattr(data["frame"], "extrinsic"):
				continue
			extrinsic = data["frame"].extrinsic
			R = extrinsic[:3,:3]
			tvec = extrinsic[:3,3]
			rvec, _ = cv2.Rodrigues(R)
			frame_indices.append(frame_index)
			intrinsics.append(data["frame"].intrinsic)
			extrinsics.append(np.concatenate([rvec[:,0], tvec]))
			dists.append(data["frame"].dist)
		intrinsics = np.array(intrinsics)
		extrinsics = np.array(extrinsics)
		dists = np.array(dists)

		points3d = []
		points2d = []
		camera_indices = []
		point_indices = []
		for i, group in enumerate(nx.connected_components(self.point_graph)):
			group = list(group)
			G = self.point_graph.subgraph(group)

			pos3d = np.mean([G.nodes[pt_index]['pos3D'] for pt_index in group], axis=0)
			points3d.append(pos3d)
			for pt_index in group:
				points2d.append(G.nodes[pt_index]['pos2D'])
				camera_indices.append(frame_indices.index(pt_index[0]))
				point_indices.append(i)
		points3d = np.array(points3d)
		points2d = np.array(points2d)
		camera_indices = np.array(camera_indices)
		point_indices = np.array(point_indices)

		x0 = np.hstack((extrinsics.ravel(), points3d.ravel()))
		A = bundle_adjustment_sparsity(len(extrinsics), len(points3d), camera_indices, point_indices)
		res = least_squares(cost_fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', loss='huber',
							args=(len(points3d), camera_indices, point_indices, points2d, intrinsics, dists))

		extrinsics = res.x[:len(extrinsics) * 6].reshape((len(extrinsics), 6))
		points3d = res.x[len(extrinsics) * 6:].reshape((len(points3d), 3))

		for frame_index in self.camera_graph.nodes():
			if not hasattr(data["frame"], "extrinsic") or frame_index not in frame_indices:
				continue
			extrinsic = extrinsics[frame_indices.index(frame_index)]
			rvec = extrinsic[:3]
			tvec = extrinsic[3:]
			R, _ = cv2.Rodrigues(rvec)
			self.camera_graph.nodes[frame_index]["frame"].extrinsic[:3,:3] = R
			self.camera_graph.nodes[frame_index]["frame"].extrinsic[:3,3] = tvec

		for i, group in enumerate(nx.connected_components(self.point_graph)):
			group = list(group)
			G = self.point_graph.subgraph(group)

			for pt_index in group:
				self.point_graph.nodes[pt_index]['pos3D'] = points3d[i]

		return res.cost


	def remove_misconnected_point_edge_iterative(self, pt_index):
		group = list(nx.node_connected_component(self.point_graph, pt_index))
		G = self.point_graph.subgraph(group)
		frame_indices = np.array([frame_index for frame_index, _ in group])

		for index in np.unique(frame_indices):
			duplicate_indices = np.where(frame_indices == index)[0]
			if len(duplicate_indices) == 2:
				source = group[duplicate_indices[0]]
				target = group[duplicate_indices[1]]
				if not nx.has_path(G, source, target):
					# Already removed in previsou loop
					continue

				"""
				path_list = list(nx.all_simple_paths(G, source, target))
				edge_list = []
				for path in path_list:
					for pi in range(len(path) - 1):
						edge_list.append(tuple(path[pi:pi+2]))
				edge_list = list(set(edge_list))
				"""

				min_weight = np.inf
				best_edge = None
				G_tmp = G.copy()
				for edge_to_remove in G.edges():
					weight = G.edges[edge_to_remove]["weight"]
					G_tmp.remove_edge(*edge_to_remove)
					if not nx.has_path(G_tmp, source, target):
						if weight < min_weight:
							best_edge = edge_to_remove
							min_weight = weight

					G_tmp.add_edge(*edge_to_remove, weight=weight)

				if best_edge is None:
					print("Bug!!! Edge to remove not found")

				else:
					"""
					pos = nx.spring_layout(G, seed=7)
					plt.clf()
					nx.draw_networkx(G, pos)
					labels = nx.get_edge_attributes(G,'weight')
					nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
					plt.show()
					"""

					self.point_graph.remove_edge(*best_edge)

					"""
					plt.clf()
					nx.draw_networkx(G, pos)
					labels = nx.get_edge_attributes(G,'weight')
					nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
					plt.show()
					"""

			elif len(duplicate_indices) > 2:
				print("Bug!!! There are more than 3 duplicates")
				print([group[index] for index in duplicate_indices])

				pos = nx.spring_layout(G, seed=7)
				plt.clf()
				nx.draw_networkx(G, pos)
				labels = nx.get_edge_attributes(G,'weight')
				nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
				plt.show()


	def remove_misconnected_point_edges(self):
		for group in nx.connected_components(self.point_graph):
			group = list(group)
			G = self.point_graph.subgraph(group)

			nodes_to_partition = []
			frame_indices = np.array([frame_index for frame_index, pt_index in group])
			for index in np.unique(frame_indices):
				duplicate_indices = np.where(frame_indices == index)[0]
				if len(duplicate_indices) >= 2:
					for i in range(len(duplicate_indices)):
						for j in range(i+1, len(duplicate_indices)):
							nodes_to_partition.append((group[duplicate_indices[i]], group[duplicate_indices[j]]))

			if len(nodes_to_partition) > 0:
				best_cut = minimum_multi_edges_cut(G, nodes_to_partition)
				self.point_graph.remove_edges_from(best_cut)


	def get_pcd(self):
		points3d = []
		colors = []

		for group in nx.connected_components(self.point_graph):
			posList = []
			colorList = []
			for pt_index in group:
				pos = self.point_graph.nodes[pt_index]['pos3D']
				color = self.point_graph.nodes[pt_index]['color']
				posList.append(pos)
				colorList.append(color)

			points3d.append(np.mean(posList, axis=0))
			colors.append(np.mean(colorList, axis=0))

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points3d)
		pcd.colors = o3d.utility.Vector3dVector(colors)
		return pcd


	def visualize(self):
		vis_graph = self.camera_graph.copy()
		vis_graph.remove_nodes_from(list(nx.isolates(vis_graph)))

		plt.clf()
		nx.draw_networkx(vis_graph)
		plt.draw()
		plt.pause(0.01)
#		plt.show()
#		nx.draw_networkx(self.point_graph)
#		plt.show()


	def _compute_pose_5points(self, frame1, frame2, matches):
		R, tvec, mask = compute_pose_5points(frame1, frame2, matches)
		frame1.extrinsic = to_extrinsic(np.eye(3, 3), np.zeros((3, 1)))
		frame2.extrinsic = to_extrinsic(R, tvec)
		if self.initialized_first_camera_pose:
			print("There seems to be multiple isolated sub view-graphs in BundleAdjustment network.")
			return False

		self.initialized_first_camera_pose = True
		return True


	def _compute_pose_pnp(self, frame1, frame2, matches, index1, index2, inverse=False):
		objpoints = []
		imgpoints = []
		for i, m in enumerate(matches):
			pt_index1 = (index1, m[0].queryIdx)
			pt_index2 = (index2, m[0].trainIdx)

			if not inverse:
				if pt_index1 in self.point_graph.nodes and 'pos3D' in self.point_graph.nodes[pt_index1]:
					objpoints.append(self.point_graph.nodes[pt_index1]['pos3D'])
					imgpoints.append(frame2.keypoints[m[0].trainIdx].pt)

			else:
				if pt_index2 in self.point_graph.nodes and 'pos3D' in self.point_graph.nodes[pt_index2]:
					objpoints.append(self.point_graph.nodes[pt_index2]['pos3D'])
					imgpoints.append(frame1.keypoints[m[0].queryIdx].pt)
		
		objpoints = np.float32(objpoints).reshape(-1,1,3)
		imgpoints = np.float32(imgpoints).reshape(-1,1,2)
		success, rvec, tvec, inliers = cv2.solvePnPRansac(objpoints, imgpoints, 
			frame1.intrinsic if inverse else frame2.intrinsic, frame1.dist if inverse else frame2.dist)
		if not success:
			print("PnP failed")
			return False

		"""
		imgpoints_reproj, _ = cv2.projectPoints(objpoints, rvec, tvec,
			frame1.intrinsic if inverse else frame1.intrinsic, frame1.dist if inverse else frame2.dist)
		rmse = np.sqrt(np.mean((imgpoints_reproj - imgpoints)[inliers] ** 2))
		print(rmse, len(inliers))
		"""

		R, _ = cv2.Rodrigues(rvec)

		if not inverse:
			frame2.extrinsic = to_extrinsic(R, tvec)
		else:
			frame1.extrinsic = to_extrinsic(R, tvec)

		return True


	def _retrieve_colors(self, frame1, frame2, matches, index1, index2):
		img1 = frame1.img
		img2 = frame2.img
		src_pts = np.int32([frame1.keypoints[m[0].queryIdx].pt for m in matches])
		dst_pts = np.int32([frame2.keypoints[m[0].trainIdx].pt for m in matches])
		colors1 = img1[src_pts[:,1], src_pts[:,0], ::-1] / 255
		colors2 = img2[dst_pts[:,1], dst_pts[:,0], ::-1] / 255

		for i, m in enumerate(matches):
			pt_index1 = (index1, m[0].queryIdx)
			pt_index2 = (index2, m[0].trainIdx)
			self.point_graph.nodes[pt_index1]['color'] = colors1[i]
			self.point_graph.nodes[pt_index2]['color'] = colors2[i]


	def _triangulate_points(self, frame1, frame2, matches, index1, index2):
		points3d = triangulate_points(frame1, frame2, frame1.extrinsic, frame2.extrinsic, matches)

		for i, m in enumerate(matches):
			pt_index1 = (index1, m[0].queryIdx)
			pt_index2 = (index2, m[0].trainIdx)

			if not 'pos' in self.point_graph.nodes[pt_index1]:
				self.point_graph.nodes[pt_index1]['pos3D'] = points3d[i]

			if not 'pos' in self.point_graph.nodes[pt_index2]:
				self.point_graph.nodes[pt_index2]['pos3D'] = points3d[i]
