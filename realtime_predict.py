import argparse
import cv2
import numpy as np
import torch
from collections import deque
from PIL import Image

from predict import predict
from utils.general import get_model, HEIGHT, WIDTH



def compute_median(cap, num_frames):
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if len(frames) == 0:
        return None, frames
    median = np.median(np.array(frames), 0).astype('uint8')[..., ::-1]
    return median, frames


def preprocess_frames(frames, bg_mode='', median=None):
    processed = np.array([]).reshape(0, HEIGHT, WIDTH)
    for f in frames:
        img = Image.fromarray(f[..., ::-1])
        if bg_mode == 'subtract':
            diff = Image.fromarray(np.sum(np.absolute(np.array(img) - median), 2).astype('uint8'))
            diff = np.array(diff.resize((WIDTH, HEIGHT)))
            diff = diff.reshape(1, HEIGHT, WIDTH)
            img_arr = diff
        elif bg_mode == 'subtract_concat':
            diff = Image.fromarray(np.sum(np.absolute(np.array(img) - median), 2).astype('uint8'))
            diff = np.array(diff.resize((WIDTH, HEIGHT)))
            diff = diff.reshape(1, HEIGHT, WIDTH)
            img = np.array(img.resize((WIDTH, HEIGHT)))
            img = np.moveaxis(img, -1, 0)
            img_arr = np.concatenate((img, diff), axis=0)
        else:
            img = np.array(img.resize((WIDTH, HEIGHT)))
            img_arr = np.moveaxis(img, -1, 0)
        processed = np.concatenate((processed, img_arr), axis=0)
    if bg_mode == 'concat' and median is not None:
        processed = np.concatenate((median, processed), axis=0)
    processed = processed / 255.
    return processed


def main(args):
    tracknet_ckpt = torch.load(args.tracknet_file)
    seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])
    tracknet.eval()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print('Failed to open video source.')
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)

    median_img = None
    frame_buffer = deque(maxlen=seq_len)
    if bg_mode:
        median_img, first_frames = compute_median(cap, args.median_frames)
        for f in first_frames[-seq_len:]:
            frame_buffer.append(f)
        if bg_mode == 'concat' and median_img is not None:
            img = Image.fromarray(median_img.astype('uint8'))
            img = np.array(img.resize((WIDTH, HEIGHT)))
            median_img = np.moveaxis(img, -1, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
        disp_frame = frame.copy()
        if len(frame_buffer) == seq_len:
            frames = preprocess_frames(list(frame_buffer), bg_mode, median_img)
            x = torch.from_numpy(frames).unsqueeze(0).float().cuda()
            with torch.no_grad():
                y_pred = tracknet(x).cpu()
            indices = torch.tensor([[(0, i) for i in range(seq_len)]])
            pred = predict(indices, y_pred=y_pred, img_scaler=img_scaler)
            x_pred = pred['X'][-1]
            y_pred_c = pred['Y'][-1]
            vis = pred['Visibility'][-1]
            if vis:
                cv2.circle(disp_frame, (x_pred, y_pred_c), 5, (0, 255, 255), -1)
        cv2.imshow('Realtime Prediction', disp_frame)
        if cv2.waitKey(int(1000 / max(fps, 1))) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='video source. 0 for webcam or path to video')
    parser.add_argument('--tracknet_file', type=str, required=True, help='file path of the TrackNet checkpoint')
    parser.add_argument('--median_frames', type=int, default=60, help='number of frames to estimate background')
    args = parser.parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    main(args)
