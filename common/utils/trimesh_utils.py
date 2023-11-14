from copy import deepcopy
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as sciR


def get_tri_pcl(
    np_points: np.ndarray, np_colors: np.ndarray
) -> trimesh.points.PointCloud:
    """
    np_points: (N,3)
    np_colors: (N, 4) np.uint8 values ranging from [0, 255]
    """
    return trimesh.points.PointCloud(np_points, np_colors)


def get_tri_pcl_uniform_color(
    np_points: np.ndarray, color: np.ndarray
) -> trimesh.points.PointCloud:
    """
    np_points: (N, 3)
    color: (3,) np.uint8 values randing from [0, 255]
    """
    np_colors = np.empty((len(np_points), 4), dtype=np.uint8)
    np_colors = color
    return get_tri_pcl(np_points, np_colors)


def get_trimesh_frame_vis(transform, arrow_radius=1e-3, arrow_height=0.1):
    cyl = trimesh.primitives.Cylinder(
        radius=arrow_radius, height=arrow_height, sections=12
    )

    x_transform = np.eye(4)
    x_transform[:3, :3] = sciR.from_euler("xyz", [0, 90, 0], degrees=True).as_matrix()
    x_transform[0, 3] = arrow_height / 2
    x_transform = transform @ x_transform
    x_axis = deepcopy(cyl).apply_transform(x_transform)
    x_axis.visual.vertex_colors = [255, 0, 0, 200]

    y_transform = np.eye(4)
    y_transform[:3, :3] = sciR.from_euler("xyz", [-90, 0, 0], degrees=True).as_matrix()
    y_transform[1, 3] = arrow_height / 2
    y_transform = transform @ y_transform
    y_axis = deepcopy(cyl).apply_transform(y_transform)
    y_axis.visual.vertex_colors = [0, 255, 0, 200]

    z_transform = np.eye(4)
    z_transform[2, 3] = arrow_height / 2
    z_transform = transform @ z_transform
    z_axis = deepcopy(cyl).apply_transform(z_transform)
    z_axis.visual.vertex_colors = [0, 0, 255, 200]

    return [x_axis, y_axis, z_axis]


class OpenGLCameraPose:
    @staticmethod
    def from_eye_position(
        eye_position: np.ndarray, look_at: np.ndarray, up_direction: np.ndarray
    ):
        normalized = lambda x: x / np.linalg.norm(x)
        camera_z = normalized(eye_position - look_at)
        camera_x = normalized(np.cross(up_direction, camera_z))
        if np.any(np.isnan(camera_x)):
            print(
                f"Warning: up_direction vector {up_direction} is parallel to camera z vector {camera_z}"
            )
            up_direction = np.copy(camera_z) + np.array([1, 1, 1])
            if np.all(up_direction == 0):
                up_direction = np.array([1, 1, 1])
            print(f"Using random non-parallel up_direction vector {up_direction}")
            camera_x = normalized(np.cross(up_direction, camera_z))
        camera_y = normalized(np.cross(camera_z, camera_x))
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = eye_position
        camera_pose[:3, :3] = np.column_stack((camera_x, camera_y, camera_z))
        return camera_pose


def get_revolving_cams(
    look_at=np.zeros((3,)), radius=1.5, theta=np.pi / 4, num_phis=30
):
    phis = np.linspace(0, 2 * np.pi, num_phis)
    cameras = []
    for phi in phis:
        eye_position = look_at + np.array(
            [
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ]
        )
        # Choose an up direction. That's it.
        up_direction = np.array([0, 0, 1])
        camera = OpenGLCameraPose.from_eye_position(eye_position, look_at, up_direction)
        cameras.append(camera)
    return cameras
