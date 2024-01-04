import os
import time
import numpy as np
import cv2 as cv
import random
import torch
from kornia.geometry import Quaternion
from kornia.geometry.epipolar import relative_camera_motion

from adalam import AdalamFilter
from example import extract_keypoints

def drawlines(img1,img2,lines,pts1,pts2, k=None):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    if k is not None:
        rnd = random.Random(0)
        lines = rnd.choices(lines, k=k)
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color,1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def calculate_fundamental_matrix_from_essential_matrix(E, K1, K2):
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

def visualize_fundamental_matrix(img1, img2, pts1, pts2, F, num_epilines=None):
    if img1.ndim == 3:
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2, k=num_epilines)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1,3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1, k=num_epilines)
    
    img_with_lines_and_points = np.hstack((img3, img5))
    img_with_points = np.hstack((img4, img6))
    return img_with_lines_and_points, img_with_points

def visualize_essential_matrix(img1, img2, pts1, pts2, E, K1, K2, num_epilines=None):
    F = calculate_fundamental_matrix_from_essential_matrix(E, K1, K2)
    return visualize_fundamental_matrix(img1, img2, pts1, pts2, F, num_epilines=num_epilines)

def calculate_q_t_error(q_true, t_true, q_estimated, t_estimated):
    assert isinstance(q_true, np.ndarray)
    assert isinstance(t_true, np.ndarray)
    assert isinstance(q_estimated, np.ndarray)
    assert isinstance(t_estimated, np.ndarray)
    
    t_estimated = t_estimated.flatten()
    t_true = t_true.flatten()
    
    assert len(q_true.shape) == 1 and q_true.shape[0] == 4
    assert len(t_true.shape) == 1 and t_true.shape[0] == 3
    assert len(q_estimated.shape) == 1 and q_estimated.shape[0] == 4
    assert len(t_estimated.shape) == 1 and t_estimated.shape[0] == 3
    
    eps = 1e-15

    q_estimated = q_estimated / (np.linalg.norm(q_estimated) + eps)
    q_true = q_true / (np.linalg.norm(q_true) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q_estimated * q_true)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t_estimated = t_estimated / (np.linalg.norm(t_estimated) + eps)
    t_true = t_true / (np.linalg.norm(t_true) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t_estimated * t_true)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    assert not (np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t))), 'This should never happen! Debug here'

    return err_q, err_t

def calculate_AUC_5_20(err_qt):
    if len(err_qt) > 0:
        err_qt = np.asarray(err_qt)
        # Take the maximum among q and t errors
        err_qt = np.max(err_qt, axis=1)
        # Convert to degree
        err_qt = err_qt * 180.0 / np.pi
        # Make infs to a large value so that np.histogram can be used.
        err_qt[err_qt == np.inf] = 1e6

        # Create histogram
        bars = np.arange(21)
        qt_hist, _ = np.histogram(err_qt, bars)
        # Normalize histogram with all possible pairs
        num_pair = float(len(err_qt))
        qt_hist = qt_hist.astype(float) / num_pair

        # Make cumulative
        qt_acc = np.cumsum(qt_hist)
    else:
        qt_acc = [0] * 20
        
    return np.mean(qt_acc[:5]), np.mean(qt_acc)

def run_on_img_pair(img_pair_path, matcher=AdalamFilter(), robust_estimator=cv.RANSAC):
    img0_path = img_pair_path + '0.png'
    img1_path = img_pair_path + '1.png'
    K_path = img_pair_path + 'k.txt'
    R_path = img_pair_path + 'r.txt'
    t_path = img_pair_path + 't.txt'
    pts1, ors1, scs1, res1, desc1, img1 = extract_keypoints(img0_path, nfeatures=8000, rootsift=False)
    pts2, ors2, scs2, res2, desc2, img2 = extract_keypoints(img1_path, nfeatures=8000, rootsift=False)
    
    extras = {'r1': res1, 'r2': res2}

    time_start = time.time()
    matches = matcher.match_and_filter(k1=pts1, k2=pts2, 
                                       d1=desc1, d2=desc2, 
                                       im1shape=img1.shape[:2], im2shape=img2.shape[:2], 
                                       o1=ors1, o2=ors2, 
                                       s1=scs1, s2=scs2,
                                       extras=extras)
    time_end = time.time()
    time_elapsed = time_end - time_start

    matches = matches.cpu().numpy()
    print(matches)
    
    pts1 = pts1[matches[:, 0]]
    pts2 = pts2[matches[:, 1]]
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    print(pts1)
    print(pts2)
    
    K = np.loadtxt(K_path)
    R_true = np.loadtxt(R_path)
    t_true = np.loadtxt(t_path)
    
    # TODO: handle when mathes are not enough
    # TODO: what should we return when essential matrix could not calculated?
    E_estimated, mask = cv.findEssentialMat(pts1, pts2, cameraMatrix=K, method=robust_estimator, prob=0.999, threshold=1.0, maxIters=1000)
    assert E_estimated is not None, 'Essential matrix estimation failed'
    
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    _, R_estimated, t_estimated, _ = cv.recoverPose(E_estimated, pts1, pts2, cameraMatrix=K)
    
    q_true = Quaternion.from_matrix(R_true)
    q_estimated = Quaternion.from_matrix(torch.from_numpy(R_estimated))
    
    q_err, t_err = calculate_q_t_error(q_true.data.detach().numpy(), t_true, q_estimated.data.detach().numpy(), t_estimated)
    return q_err, t_err, time_elapsed


def run_on_datasets(datasets, matcher=AdalamFilter(), robust_estimator=cv.RANSAC):
    for dataset in datasets:
        dataset_path = 'datasets/' + dataset + '/'
        img_pair_paths = [dataset_path + img_pair_name + '/' for img_pair_name in os.listdir(dataset_path)]
        err_qt_list = []
        runtime_list = []
        for img_pair_path in img_pair_paths:
            q_err, t_err, elapsed_time = run_on_img_pair(img_pair_path, matcher, robust_estimator)
            err_qt_list.append((q_err, t_err))
            runtime_list.append(elapsed_time)
    
    auc5, auc20 = calculate_AUC_5_20(err_qt_list)
    return auc5, auc20, runtime_list

def main():
    matcher = AdalamFilter({'scoring_method': 'fginn'})
    auc5, auc20, runtime_list = run_on_datasets(os.listdir('datasets/'), matcher=matcher)
    
    print(auc5)
    print(auc20)
    
if __name__ == '__main__':
    # TODO: hyperparameter optimization using TUM dataset
    main()
