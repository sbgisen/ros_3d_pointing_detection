#!/usr/bin/env python
# -*- coding: utf-8 -*-

import message_filters
import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import ClassificationResult
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from tfpose_ros.msg import Persons

from ros_3d_pointing_detection.calc_3d_dist import point_3d_line_distance
from ros_3d_pointing_detection.calc_3d_dist import point_plane_distance
from ros_3d_pointing_detection.msg import DetectedObject


class PointingDetector3D(object):
    def __init__(self):
        rospy.init_node('3d_pointing_detector')

        self._persons_sub = message_filters.Subscriber("~persons", Persons)
        self._objects_sub = message_filters.Subscriber("~objects", BoundingBoxArray)
        self._points_sub = message_filters.Subscriber("~points", PointCloud2)
        self._sub = message_filters.ApproximateTimeSynchronizer(
            [self._persons_sub, self._objects_sub, self._points_sub], 10, 1)
        self._sub.registerCallback(self._callback)

        self.__pub = rospy.Publisher('~detect_object', DetectedObject, queue_size=10)
        self.__result_pub = rospy.Publisher('~result', ClassificationResult, queue_size=10)

    def _callback(self, persons_msg, objects_msg, points_msg):
        if not persons_msg.persons:
            return

        points_list = list(point_cloud2.read_points(points_msg, field_names=("x", "y", "z")))
        points = np.array(points_list)
        right_arm_joints = self.right_arm_joints(persons_msg.persons[0])

        if right_arm_joints is None:
            rospy.loginfo("not found right arm")
            return

        if not self.is_arm_stretched(right_arm_joints):
            rospy.loginfo("not stretched")
            return

        is_hit, hit_point = self.get_3d_ray_hit_point(right_arm_joints, points)

        if not is_hit:
            rospy.loginfo("not hit")
            return

        line = right_arm_joints[2] - right_arm_joints[0]
        base = np.array([1, 0, 0])
        axis = np.cross(base, line)
        angle = np.arccos(np.dot(base, line))
        q = tf.transformations.quaternion_about_axis(angle, tuple(axis))
        pointing_pose = PoseStamped()
        pointing_pose.header = persons_msg.header
        pointing_pose.pose.position = Point(right_arm_joints[0][0], right_arm_joints[0][1], right_arm_joints[0][2])
        pointing_pose.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])
        self.__pose_pub.publish(pointing_pose)

        min_dist = 0.5
        min_box = None
        result = ClassificationResult()
        result.header = objects_msg.header
        for i, box in enumerate(objects_msg.boxes):
            origin = np.array([box.pose.position.x, box.pose.position.y, box.pose.position.z])
            dist = np.linalg.norm(origin - hit_point)
            if dist < min_dist:
                min_dist = dist
                min_box = box
                result.label_names.append('pointing_object')
            else:
                result.label_names.append('')
        if min_box is not None:
            self.__pub.publish(
                DetectedObject(
                    header=min_box.header,
                    id="",
                    pose=min_box.pose,
                    dimensions=min_box.dimensions))

        self.__result_pub.publish(result)

    def right_arm_joints(self, person):
        p0 = p1 = p2 = None
        for part in person.body_part:
            if part.part_id == 2:
                p0 = np.array([part.x, part.y, part.z])
            elif part.part_id == 3:
                p1 = np.array([part.x, part.y, part.z])
            elif part.part_id == 4:
                p2 = np.array([part.x, part.y, part.z])
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
