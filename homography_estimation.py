import os
import time
import cv2 as cv
import numpy as np
from adalam import AdalamFilter

from example import extract_keypoints


def calculate_average_corner_error(img1_h, img1_w, H_estimated, H_true):
    corners = [[0, 0], [0, img1_w], [img1_h, 0], [img1_h, img1_w]]

    pts_estimated = cv.perspectiveTransform(np.float32([corners]), H_estimated)[0]
    pts_true = cv.perspectiveTransform(np.float32([corners]), H_true)[0]
    total_corner_error = np.sum(np.linalg.norm(pts_estimated - pts_true, axis=1))

    return total_corner_error / 4


def run_on_img_pair(img_pair_path, matcher=AdalamFilter(), robust_estimator=cv.RANSAC):
    img0_path = img_pair_path + '0.png'
    img1_path = img_pair_path + '1.png'
    H_path = img_pair_path + 'H.txt'
    pts1, ors1, scs1, desc1, im1 = extract_keypoints(img0_path, nfeatures=8000, rootsift=False)
    pts2, ors2, scs2, desc2, im2 = extract_keypoints(img1_path, nfeatures=8000, rootsift=False)
    
    time_start = time.time()
    matches = matcher.match_and_filter(k1=pts1, k2=pts2, 
                                       d1=desc1, d2=desc2, 
                                       im1shape=im1.shape[:2], im2shape=im2.shape[:2], 
                                       o1=ors1, o2=ors2, 
                                       s1=scs1, s2=scs2)
    time_end = time.time()
    time_elapsed = time_end - time_start

    matches = matches.cpu().numpy()

    if len(matches) < 4:
        return float('inf'), time_elapsed

    H_true = np.loadtxt(H_path)
    H_estimated, _ = cv.findHomography(pts1[matches[:, 0]], pts2[matches[:, 1]], robust_estimator, 3, None, 2000, 0.995)

    ace = calculate_average_corner_error(im1.shape[0], im1.shape[1], H_estimated, H_true)
    diag_dist = np.linalg.norm(im1.shape[:2])
    ace_over_diag_dist = ace / diag_dist
    return ace_over_diag_dist, time_elapsed


def run_on_dataset(dataset='oxford', matcher=AdalamFilter(), robust_estimator=cv.RANSAC):
    dataset_path = 'homography/datasets/' + dataset + '/'
    img_pair_paths = [dataset_path + img_pair_name + '/' for img_pair_name in os.listdir(dataset_path)]
    ace_over_diag_dist_list = []
    runtime_list = []
    for img_pair_path in img_pair_paths:
        ace_over_diag_dist, elapsed_time = run_on_img_pair(img_pair_path, matcher, robust_estimator)
        ace_over_diag_dist_list.append(ace_over_diag_dist)
        runtime_list.append(elapsed_time)
    return ace_over_diag_dist_list, runtime_list


def main(dataset='oxford', matcher=AdalamFilter(), robust_estimator=cv.RANSAC, verbose=True):
    ace_over_diag_dist_list, runtime_list = run_on_dataset(dataset=dataset, matcher=matcher, robust_estimator=robust_estimator)
    runtime_avg = np.mean(runtime_list)
    if verbose:
        print('Average runtime: ', runtime_avg)
    error_tolerance = 0.05
    success_count = 0
    for ace_over_diag_dist in ace_over_diag_dist_list:
        if ace_over_diag_dist < error_tolerance:
            success_count += 1
    success_rate = success_count / len(ace_over_diag_dist_list)
    if verbose:
        print('Success rate: ', success_rate)
    return success_rate, runtime_avg


if __name__ == '__main__':
    if not os.path.exists('homography'):
        print('Clone https://github.com/ersincine/homography')
    elif not os.path.exists('homography/datasets'):
        print('Run main.py in homography')
    else:
        main()
