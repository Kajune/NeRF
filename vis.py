import open3d as o3d
import sys

if __name__ == '__main__':
	pcd = o3d.io.read_point_cloud(sys.argv[1])
	o3d.visualization.draw_geometries([pcd])
