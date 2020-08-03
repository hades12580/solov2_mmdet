import argparse

import cv2
import torch

from mmdet.apis import inference_detector, init_detector, show_result
import time


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera', default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    camera = cv2.VideoCapture(args.camera)

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        tic = time.time()
        result = inference_detector(model, img)
        cost = time.time() - tic
        print('cost: {}, fps: {}'.format(cost, 1/cost))

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        show_result(
            img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1)


if __name__ == '__main__':
    main()
