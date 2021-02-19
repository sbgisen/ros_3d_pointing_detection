#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ros_3d_pointing_detection.calc_3d_dist import point_3d_line_distance, point_plane_distance
from tfpose_ros.msg import Persons
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, CameraInfo
from sensor_msgs import point_cloud2
import numpy as np

from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
import message_filters
import rospy


class PointingDetector3D(object):
    def __init__(self):
        rospy.init_node('3d_pointing_detector')

        self._persons_sub = message_filters.Subscriber("~persons", Persons)
        self._darknet_sub = message_filters.Subscriber("~darknet", BoundingBoxes)
        self._points_sub = message_filters.Subscriber("~points", PointCloud2)
        self._camera_info_sub = message_filters.Subscriber('~camera_info', CameraInfo)
        self._sub = message_filters.ApproximateTimeSynchronizer(
            [self._persons_sub, self._darknet_sub, self._points_sub, self._camera_info_sub], 10, 1)
        self._sub.registerCallback(self._callback)

        self.__pub = rospy.Publisher('~detect_object', String, queue_size=10)

    def _callback(self, persons_msg, darknet_msg, points_msg, camera_info_msg):
        if not persons_msg.persons:
            return

        points_list = list(point_cloud2.read_points(points_msg, field_names=("x", "y", "z")))
        points = np.array(points_list)
        points_mat = points.reshape([points_msg.height, points_msg.width, 3])
        right_arm_joints = self.right_arm_joints(persons_msg.persons[0], points_mat)

        if right_arm_joints is None:
            return

        if not self.is_arm_stretched(right_arm_joints):
            return

        is_hit, hit_point = self.get_3d_ray_hit_point(right_arm_joints, points)

        if not is_hit:
            return

        hit_point_2d = self.cam2pixel(hit_point, np.array(camera_info_msg.K).reshape([3, 3]))

        for bbox in darknet_msg.bounding_boxes:
            xmin = bbox.x
            ymin = bbox.y
            xmax = xmin + bbox.w
            ymax = ymin + bbox.h
            if hit_point_2d[0] >= xmin and hit_point_2d[0] <= xmax and hit_point_2d[1] >= ymin and hit_point_2d[1] <= ymax:
                self.__pub.publish(String(data=bbox.Class))

    def right_arm_joints(self, person, points):
        p0 = p1 = p2 = None
        image_h, image_w = points.shape[:2]
        for part in person.body_part:
            if part.part_id == 2:
                p0 = points[int(part.y * image_h + 0.5), int(part.x * image_w + 0.5)]
            elif part.part_id == 3:
                p1 = points[int(part.y * image_h + 0.5), int(part.x * image_w + 0.5)]
            elif part.part_id == 4:
                p2 = points[int(part.y * image_h + 0.5), int(part.x * image_w + 0.5)]
        if p0 is None or p1 is None or p2 is None:
            return None
        return (p0, p1, p2)

    def is_arm_stretched(self, right_arm_joints, angle_thresh=30.0):
        vec1 = np.array(right_arm_joints[1] - right_arm_joints[0])
        vec2 = np.array(right_arm_joints[2] - right_arm_joints[1])
        angle = np.arccos(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        angle = angle / np.pi * 180.0
        return np.abs(angle) <= angle_thresh

    def get_3d_ray_hit_point(self, right_arm_joints, points, thresh_in_front_of_wrist=0.50, thresh_close_to_line=0.1):
        ''' Get the hit point between the pointing ray and the point cloud.

        A point in the point cloud that is
            (1) in front of the wrist for `thresh_in_front_of_wrist`,
            (2) and is close to the ray within `thresh_close_to_line`
        is considered as the hit point.

        Arguments:
            right_arm_joints {np.ndarray}: shape=(3, 3). Three joints' xyz positions.
            points {np.ndarray}: shape=(N, 3). N points of xyz positions.
        Return:
            ret {bool}: Is there a valid hit point.
            xyz {np.ndarray}: shape=(3, ). The hit point's position.
        '''
        p1, p2 = right_arm_joints[0], right_arm_joints[2]

        # Select points that are in front of the wrist.
        dists_plane = point_plane_distance(points, p1, p2 - p1)
        thresh = thresh_in_front_of_wrist + np.linalg.norm(p2 - p1)
        valid_idx = dists_plane >= thresh
        valid_pts = points[valid_idx]
        dists_plane = dists_plane[valid_idx]
        if valid_pts.size == 0:
            return False, None

        # Select points that are close to the pointing direction.
        dists_3d_line = point_3d_line_distance(valid_pts, p1, p2)
        valid_idx = dists_3d_line <= thresh_close_to_line
        valid_pts = valid_pts[valid_idx]
        if valid_pts.size == 0:
            return False, None
        dists_plane = dists_plane[valid_idx]

        # Get hit point.
        closest_point_idx = np.argmin(dists_plane)
        hit_point = valid_pts[closest_point_idx]
        return True, hit_point

    def cam2pixel(self, xyz_in_camera, camera_intrinsics):
        ''' Project a point represented in camera coordinate onto the image plane.
        Arguments:
            xyz_in_camera {np.ndarray}: (3, ).
            camera_intrinsics {np.ndarray}: 3x3.
        Return:
            xy {np.ndarray, np.float32}: (2, ). Column and row index.
        '''
        pt_3d_on_cam_plane = xyz_in_camera / xyz_in_camera[2]  # z=1
        xy = camera_intrinsics.dot(pt_3d_on_cam_plane)[0:2]
        xy = tuple(int(v) for v in xy)
        return xy


def main():
    _ = PointingDetector3D()
    rospy.spin()


if __name__ == '__main__':
    main()
