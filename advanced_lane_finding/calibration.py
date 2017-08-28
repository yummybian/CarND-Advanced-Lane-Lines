import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


class Calibration:
    def __init__(self, nx, ny):
        # nx represents the number of inside corners in x
        self.nx, self.ny = nx, ny
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane
        self.cal_imgs = []
        self.cal_imgs_with_corners = []
        self.test_imgs = []

    def load_cal_imgs(self, cal_path):
        cal_img_names = glob.glob(cal_path)
        self.cal_imgs = [cv2.imread(name) for name in cal_img_names]

    def load_test_imgs(self, test_path):
        test_img_names = glob.glob(test_path)
        self.test_imgs = [cv2.imread(name) for name in test_img_names]

    def draw_corners(self):
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        copy_imgs = [img.copy() for img in self.cal_imgs]
        for img in copy_imgs:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # If found, draw corners
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw the corners
                cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                self.cal_imgs_with_corners.append(img)

        print(len(self.cal_imgs))
        print(len(self.cal_imgs_with_corners))

        
    def show_plot(self):
        for i, img in enumerate(self.cal_imgs):
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=18)
            ax2.imshow(self.cal_imgs_with_corners[i])
            ax2.set_title('Image With Corners', fontsize=18)

