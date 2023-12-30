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

def read_extrinsics(filename: str):
    assert filename.endswith('.png'), 'Filename must end with .png'
    
    timestamp = float(filename[:-4])
    
    with open('groundtruth.txt', 'r') as f:
        lines = f.read().splitlines()
        assert all([line.startswith('#') for line in lines[:3]])
        lines = lines[3:]
        
        nearest_distance = float('inf')
        nearest_idx = None
        
        for line_idx, line in enumerate(lines):
            values = line.split(' ')
            
            current_distance = abs(float(values[0]) - timestamp)
            if nearest_distance > current_distance:
                nearest_distance = current_distance
                nearest_idx = line_idx
        
        if nearest_idx is not None:
            values = lines[nearest_idx].split(' ')
            tx = float(values[1])
            ty = float(values[2])
            tz = float(values[3])
            qx = float(values[4])
            qy = float(values[5])
            qz = float(values[6])
            qw = float(values[7])
            return tx, ty, tz, qx, qy, qz, qw
        
    assert False, 'Timestamp not found in groundtruth.txt'

def main():
    matcher = AdalamFilter()
    
    img0_path = '1305032353.993165.png'
    img1_path = '1305032354.025237.png' # 1305032354.025237.png
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
    
    pts1 = pts1[matches[:, 0]]
    pts2 = pts2[matches[:, 1]]
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    
    # F_estimated, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 3, 0.99, None)
    # assert F_estimated is not None, 'Fundamental matrix estimation failed'
    
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    E_estimated, mask = cv.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv.RANSAC, prob=0.999, threshold=1.0, maxIters=1000)
    assert E_estimated is not None, 'Essential matrix estimation failed'
    F_estimated = calculate_fundamental_matrix_from_essential_matrix(E_estimated, K, K)
    
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    img_with_lines_and_points, img_with_points = visualize_fundamental_matrix(img1, img2, pts1, pts2, F_estimated, num_epilines=20)
    cv.imwrite('img_with_lines_and_points.png', img_with_lines_and_points)
    cv.imwrite('img_with_points.png', img_with_points)
    
    tx0, ty0, tz0, qx0, qy0, qz0, qw0 = read_extrinsics(img0_path)
    tx1, ty1, tz1, qx1, qy1, qz1, qw1 = read_extrinsics(img1_path)
    
    img1_quaternion = Quaternion(torch.tensor([qw0, qx0, qy0, qz0]))
    img2_quaternion = Quaternion(torch.tensor([qw1, qx1, qy1, qz1]))
    # img1_euler = img1_quaternion.to_euler()
    # img2_euler = img2_quaternion.to_euler()
    
    t1 = torch.tensor([[tx0], [ty0], [tz0]])
    t2 = torch.tensor([[tx1], [ty1], [tz1]])
    R1 = img1_quaternion.matrix()
    R2 = img2_quaternion.matrix()
    
    R_true, t_true = relative_camera_motion(R1, t1, R2, t2)
    _, R_estimated, t_estimated, _ = cv.recoverPose(E_estimated, pts1, pts2, cameraMatrix=K)
    
    print(R_true)
    print(t_true)
    print(R_estimated)
    print(t_estimated)
    
    # TODO: Calculate error

if __name__ == '__main__':
    main()