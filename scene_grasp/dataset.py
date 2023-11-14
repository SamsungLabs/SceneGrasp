from common.utils.misc_utils import transform_pcl


class NOCSDataPoint:
    def __init__(
        self,
        rgb,
        depth,
        camera_k,
        seg_masks,
        class_ids,
        class_confidences,
        obj_canonical_pcls,
        scale_matrices,
        pose_matrices,
        endpoints,
        metadata,
    ):
        """
        pose @ scale @ obj_canonical_pcls -> obj pcl in camera frame
        """
        self.rgb = rgb
        self.depth = depth
        self.camera_k = camera_k
        self.seg_masks = seg_masks
        self.class_ids = class_ids
        self.class_confidences = class_confidences
        self.obj_canonical_pcls = obj_canonical_pcls
        self.scale_matrices = scale_matrices
        self.pose_matrices = pose_matrices
        self.endpoints = endpoints
        self.__camera_frame_pcls = None
        self.metadata = metadata

    def get_camera_frame_pcls(self):
        if self.__camera_frame_pcls is None:
            self.__camera_frame_pcls = []
            for can_idx, can_pcl in enumerate(self.obj_canonical_pcls):
                scale_matrix = self.scale_matrices[can_idx]
                pose_matrix = self.pose_matrices[can_idx]
                camera_pcl = transform_pcl(can_pcl, pose_matrix, scale_matrix)
                self.__camera_frame_pcls.append(camera_pcl)
        return self.__camera_frame_pcls

    def get_len(self):
        return len(self.obj_canonical_pcls)
