import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import pickle

from automatic_mask_and_probability_generator import \
    SamAutomaticMaskAndProbabilityGenerator


def get_args():
    parser = argparse.ArgumentParser(description='Test output')

    # Root Directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # Dataset
    parser.add_argument('--dataset', type=str, help='BSDS500 or NYUDv2')
    parser.add_argument('--data_split', type=str, default='test',
                        help='train, val or test')

    # Top Mask Selection
    parser.add_argument('--t', type=int)
    # Spectral Clustering
    parser.add_argument('--c', type=int)
    #  Boundary Zero Padding
    parser.add_argument('--p', type=int)
    # Kernel Size for Gaussian Blur
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='kernel size')
    # Temperature Hyperparameter for Similarity Matrix
    parser.add_argument('--tau', type=float, default=0.5,
                        help='tau')
    args = parser.parse_args()
    return args


def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)

    return image


def save_edge_img(p_max, kernel_size,
                  output_path, gray_path, root_dir):

    edges = normalize_image(p_max)

    if kernel_size > 0:
        assert kernel_size % 2 == 1
        edges = cv2.GaussianBlur(edges, (kernel_size, kernel_size), 0)

    # We temporarily used OpenCV's Structured Forests model for edge NMS.
    # In the original SAM paper, Canny edge NMS was used for edge NMS.
    # However, in our environment, it did not produce the edges reported
    # in the paper. This part needs further investigation and improvement.
    # Note that even if we used the Structured Forests model for edge NMS, 
    # the results of the SAM original paper could not be achieved.

    # OpenCV's demo code: https://github.com/opencv/opencv_contrib/blob/3.4.0/modules/ximgproc/samples/edgeboxes_demo.py  # noqa
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(
        str(root_dir / 'model/model.yml.gz'))
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edges = (edges * 255).astype(np.uint8)

    # image for evaluation
    cv2.imwrite(str(output_path), edges)
    # image for visualization
    plt.imsave(str(gray_path), edges, cmap='binary')


def main():
    args = get_args()

    root_dir = Path(args.root_dir)

    dataset = args.dataset
    assert dataset in ['BSDS500', 'NYUDv2']
    data_split = args.data_split
    assert data_split in ['train', 'val', 'test']

    T = args.t
    C = args.c
    P = args.p
    ks = args.kernel_size
    tau = args.tau

    exp = ''
    if T is not None:
        assert T > 0
        exp += f't{T}'
    
    if C is not None:
        assert C > 0 and tau is not None
        if len(exp) > 0:
            exp += '_'
        exp += f'c{C}'
    
    if P is not None:
        assert P > 0
        if len(exp) > 0:
            exp += '_'
        exp += f'p{P}'

    if ks > 0:
        assert ks % 2 == 1
        if len(exp) > 0:
            exp += '_'
        exp += f'ks{ks}'
    
    if tau > 0 and C is not None:
        if len(exp) > 0:
            exp += '_'
        exp += f'tau{tau}'

    if exp == '':
        exp = 'noselect'

    device = "cuda"
    sam = sam_model_registry["default"](
        checkpoint=str(root_dir / "model/sam_vit_h_4b8939.pth"))
    sam.to(device=device)
    generator = SamAutomaticMaskAndProbabilityGenerator(
        sam, points_per_batch=1024)

    masks_root_dir = root_dir / f'output/{dataset}/masks/{data_split}'
    output_root_dir = root_dir / f'output/{dataset}/pred/{exp}/{data_split}'
    gray_root_dir = root_dir / f'output/{dataset}/pred_gray/{exp}/{data_split}'

    masks_root_dir.mkdir(parents=True, exist_ok=True)
    output_root_dir.mkdir(parents=True, exist_ok=True)
    gray_root_dir.mkdir(parents=True, exist_ok=True)

    img_dir = root_dir / f'data/{dataset}/images/{data_split}'

    if dataset == 'BSDS500':
        suf = '.jpg'
    elif dataset == 'NYUDv2':
        suf = '.png'

    for img_path in tqdm(sorted(img_dir.glob(f'*{suf}'))):
        name = img_path.stem
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks_path = masks_root_dir / f'{name}.pkl'
        if Path(masks_path).exists():
            print(f'load {masks_path}')
            with open(masks_path, 'rb') as f:
                masks = pickle.load(f)
        else:
            # Automatic Mask Generation (AMG)
            masks = generator.generate(image)
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            with open(masks_path, 'wb') as f:
                pickle.dump(masks, f)

        # Top Mask Selection
        if T is not None:
            masks = masks[:len(masks)//T]

        if C is not None:

            # Points for Spectral Clustering
            points = []
            for m in masks:
                x, y, w, h = m['bbox']
                points.append([(x+w/2, y+h/2)])

            # Preparetion for Adjacency Matrix
            segs = [m['segmentation'] for m in masks]
            n = len(segs)
            dist_mat = np.zeros((n, n))
            count_mat = np.zeros((n, n))
            for i, seg1 in enumerate(segs):
                for j, seg2 in enumerate(segs):

                    if i == j:
                        break

                    seg1 = seg1.astype(np.uint8)
                    seg2 = seg2.astype(np.uint8)

                    min_area = min(np.sum(seg1), np.sum(seg2))
                    count = np.sum(seg1 * seg2)
                    assert count <= min_area
                    if min_area == 0:
                        count_mat[i, j] = 0
                        count_mat[j, i] = 0
                    else:
                        count_mat[i, j] = count / min_area
                        count_mat[j, i] = count / min_area

                    dist = np.linalg.norm(
                        np.array(points[i][0]) - np.array(points[j][0]))
                    dist_mat[i, j] = dist
                    dist_mat[j, i] = dist

            # Calculate Adjacency Matrix
            # K = 7 is the hyperparameter used in the following paper:
            # Zelnik-Manor and Perona.
            # Self-Tuning Spectral Clustering. NeurIPS 2004.
            K = 7
            min_sz = dist_mat.shape[1]
            sz = min(min_sz-1, K)
            scale = np.partition(dist_mat, sz, axis=1)[:, sz]
            scale_mat = scale.reshape(-1, 1) * scale.reshape(1, -1)
            scale_mat = np.maximum(scale_mat, 1e-6)
            adj_mat = np.exp(-dist_mat**2 / scale_mat) * np.exp(count_mat/tau)

            # Spectral Clustering
            k = max(len(masks)//C, 2)
            sc = SpectralClustering(k, affinity='precomputed', n_init=100)
            sc.fit(adj_mat)
            labels = sc.labels_
            idx2label = {i: l for i, l in enumerate(labels)}

            # Edge Detection
            sc_masks = [None for _ in range(k)]
            sc_probs = [None for _ in range(k)]
            for i, mask in enumerate(masks):
                seg = mask['segmentation']
                prob = mask['prob']

                label = idx2label[i]
                if sc_masks[label] is None:
                    sc_masks[label] = seg
                    sc_probs[label] = prob
                else:
                    sc_masks[label] = np.maximum(sc_masks[label], seg)
                    sc_probs[label] = np.maximum(sc_probs[label], prob)

            for label in range(k):
                if sc_probs[label] is None:
                    continue

                mask = torch.from_numpy(
                    sc_masks[label]).to(device).float()
                sobel_filter_x = torch.tensor(
                    [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                    dtype=torch.float32
                ).to(mask.device).unsqueeze(0)
                sobel_filter_y = torch.tensor(
                    [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                    dtype=torch.float32
                ).to(mask.device).unsqueeze(0)
                G_x = F.conv2d(mask[None, None], sobel_filter_x, padding=1)
                G_y = F.conv2d(mask[None, None], sobel_filter_y, padding=1)
                edge = torch.sqrt(G_x ** 2 + G_y ** 2)
                outer_boundary = (edge > 0).float()

                sc_probs[label] = torch.from_numpy(
                    sc_probs[label]).to(device).unsqueeze(0)
                sc_probs[label][0] = sc_probs[label][0] * outer_boundary

                sc_probs[label] = sc_probs[label].squeeze(0).cpu().numpy()

            p_max = None
            for label in range(k):
                if sc_probs[label] is None:
                    continue
                p = sc_probs[label]
                if p_max is None:
                    p_max = p
                else:
                    p_max = np.maximum(p_max, p)
        else:
            p_max = None
            for mask in masks:
                p = mask["prob"]
                if p_max is None:
                    p_max = p
                else:
                    p_max = np.maximum(p_max, p)

        # Boundary Zero Padding
        if P is not None:
            p_max[:P, :] = 0
            p_max[-P:, :] = 0
            p_max[:, :P] = 0
            p_max[:, -P:] = 0

        output_path = output_root_dir / f'{name}.png'
        gray_path = gray_root_dir / f'{name}.png'

        save_edge_img(p_max, ks, output_path, gray_path, root_dir)


if __name__ == "__main__":
    main()
