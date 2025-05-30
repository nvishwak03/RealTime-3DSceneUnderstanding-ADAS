import os
import sys
import torch
import joblib
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data import DataLoader

from multi_person_tracker import MPT
from models import VIBE_Demo
from img_utils import Inference
from utils import (
    video_to_images,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    download_ckpt
)

MIN_NUM_FRAMES = 25

def run_vibe(video_path):
    video_file = video_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f'Video file "{video_path}" does not exist.')

    output_path = os.path.splitext(video_path)[0] + '_vibe_output'
    os.makedirs(output_path, exist_ok=True)

    image_folder, num_frames, img_shape = video_to_images(video_path, return_info=True)
    orig_height, orig_width = img_shape[:2]
    tracker = MPT(device=device, batch_size=12, detector_type='yolo', output_format='dict', yolo_img_size=416, display=False
    )
    tracking_results = tracker(image_folder)
    tracking_results = {
        pid: track for pid, track in tracking_results.items()
        if track['frames'].shape[0] >= MIN_NUM_FRAMES
    }

    model = VIBE_Demo(seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True ).to(device)

    ckpt_path = download_ckpt()
    ckpt = torch.load(ckpt_path)['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    vibe_results = {}

    for person_id in tqdm(tracking_results.keys()):
        frames = tracking_results[person_id]['frames']
        bboxes = tracking_results[person_id]['bbox']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            scale=1.1,
        )
        dataloader = DataLoader(dataset, batch_size=450, num_workers=4)

        pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d = [], [], [], [], [], []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.unsqueeze(0).to(device)
                output = model(batch)[-1]
                b, s = batch.shape[:2]

                pred_cam.append(output['theta'][:, :, :3].reshape(b * s, -1))
                pred_verts.append(output['verts'].reshape(b * s, -1, 3))
                pred_pose.append(output['theta'][:, :, 3:75].reshape(b * s, -1))
                pred_betas.append(output['theta'][:, :, 75:].reshape(b * s, -1))
                pred_joints3d.append(output['kp_3d'].reshape(b * s, -1, 3))
                smpl_joints2d.append(output['kp_2d'].reshape(b * s, -1, 2))

        pred_cam = torch.cat(pred_cam).cpu().numpy()
        pred_verts = torch.cat(pred_verts).cpu().numpy()
        pred_pose = torch.cat(pred_pose).cpu().numpy()
        pred_betas = torch.cat(pred_betas).cpu().numpy()
        pred_joints3d = torch.cat(pred_joints3d).cpu().numpy()
        smpl_joints2d = torch.cat(smpl_joints2d).cpu().numpy()

        orig_cam = convert_crop_cam_to_orig_img(pred_cam, bboxes, orig_width, orig_height)
        joints2d_img_coord = convert_crop_coords_to_orig_img(bboxes, smpl_joints2d, crop_size=224)

        vibe_results[person_id] = { 'pred_cam': pred_cam, 'orig_cam': orig_cam, 'verts': pred_verts, 'pose': pred_pose, 'betas': pred_betas, 'joints3d': pred_joints3d, 'joints2d_img_coord': joints2d_img_coord, 'bboxes': bboxes,  'frame_ids': frames,
        }

    joblib.dump(vibe_results, os.path.join(output_path, 'vibe_output.pkl'))
    print(f'Inference complete. Results saved to: {os.path.join(output_path, "vibe_output.pkl")}')

    from utils import Renderer
    from utils import images_to_video, prepare_rendering_results
    import colorsys

    renderer = Renderer(resolution=(orig_width, orig_height))

    output_img_folder = f'{image_folder}_output'
    os.makedirs(output_img_folder, exist_ok=True)

    print(f'Rendering output video, writing frames to {output_img_folder}')

    frame_results = prepare_rendering_results(vibe_results, num_frames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    for frame_idx in range(len(image_file_names)):
        img_fname = image_file_names[frame_idx]
        img = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)


        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']
            mc = mesh_color[person_id]

            img = renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                color=mc,
            )

        cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

    vid_name = os.path.basename(video_file)
    save_name = f'{vid_name.replace(".mp4", "")}_vibe_result.mp4'
    save_name = os.path.join(output_path, save_name)
    images_to_video(output_img_folder, save_name)

    return vibe_results

if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit(1)
    run_vibe(sys.argv[1])
