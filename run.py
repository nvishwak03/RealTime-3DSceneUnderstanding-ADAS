#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import torch

from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView

def main():
    depth = "output_videos/depth_out.mp4"

    detector = ObjectDetector('nano', 0.25, 0.45, None, 'cpu')
    depth_estimator = DepthEstimator('small', 'cpu')
    bbox3d_estimator = BBox3DEstimator()
    bev = BirdEyeView(scale=60, size=(300, 300))

    cap = cv2.VideoCapture("output_videos/Lane_Det.mp4")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_videos/output.mp4", fourcc,fps, (w, h))
    dep_Video = cv2.VideoWriter(depth, fourcc,fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame.copy()

        temp, detections = detector.detect(frame, track=True)

        depth_map = depth_estimator.estimate_depth(frame)
        depth_colored = depth_estimator.colorize_depth(depth_map)

        boxes_3d = []
        active_ids = []

        for det in detections:
            bbox, score, class_id, obj_id = det
            Class = detector.get_class_names()[class_id]
            depth_value = depth_estimator.get_depth_in_region(depth_map, bbox)

            box_3d = {
                'bbox_2d': bbox,
                'depth_value': depth_value,
                'class_name': Class,
                'object_id': obj_id,
                'score': score
            }
            boxes_3d.append(box_3d)

            if obj_id is not None:
                active_ids.append(obj_id)

        bbox3d_estimator.cleanup_trackers(active_ids)

        for box in boxes_3d:
            frame = bbox3d_estimator.draw_box_3d(frame, box)
            bev.reset()
            for box in boxes_3d:
                bev.draw_box(box)
            bev_image = bev.get_image()

            bev_h = h // 3
            bev_w = bev_h
            bev_resized = cv2.resize(bev_image, (bev_w, bev_h))
            frame[h - bev_h:h, 0:bev_w] = bev_resized
            cv2.rectangle(frame, (0, h - bev_h), (bev_w, h), (255, 255, 255), 1)
            cv2.putText(frame, "BEV", (10, h - bev_h + 20),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        dep_Video.write(cv2.resize(depth_colored, (w, h)))

        cv2.imshow("3D Object Detection + BEV", frame)
        cv2.imshow("Depth Map", depth_colored)

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    out.release()
    dep_Video.release()
    cv2.destroyAllWindows()
    print("3D Bounding Box Done")

if __name__ == "__main__":
    main()
