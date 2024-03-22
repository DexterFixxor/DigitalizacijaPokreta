import cv2
import numpy as np
from utils import ucitaj_fajlove, ucitaj_unutrasnje_parametre, projektuj_tacke

def draw_axis(img, rot_mat, translation, camera_matrix, length):
    points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [
                        0, 0, length]]).reshape(-1, 3).T
    image_points = projektuj_tacke(
        points, camera_matrix, rot_mat, translation).astype(np.int32)
    img = cv2.line(img, tuple(image_points[:, 0].ravel()), tuple(
        image_points[:, 1].ravel()), (0, 0, 255), 5)
    img = cv2.line(img, tuple(image_points[:, 0].ravel()), tuple(
        image_points[:, 2].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, tuple(image_points[:, 0].ravel()), tuple(
        image_points[:, 3].ravel()), (255, 0, 0), 5)
    return img

def camera_position(images_path, calib_file,  file_name, widht, height, cell_size):
    cam_mat, dist_coeffs = ucitaj_unutrasnje_parametre(calib_file)
    all_files = ucitaj_fajlove(images_path, ekstenzije={".png", ".jpg"})
    
    object_points = np.zeros((widht*height, 3), np.float32)
    for i in range(height):
        for j in range(widht):
            object_points[i*widht + j,
                          :] = np.array([j*cell_size, i*cell_size, 0], dtype=np.float32)
    all_rvec = np.zeros((3, 0), dtype=np.float32)
    all_tvec = np.zeros((3, 0), dtype=np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for image in all_files:
        img = cv2.imread(image)
        undistorted = cv2.undistort(img, cam_mat, dist_coeffs, None, cam_mat)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (widht, height), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria)
            ret, rvec, tvec, _ = cv2.solvePnPRansac(
                object_points, corners2, cam_mat, None)
            if ret:
                all_rvec = np.hstack((all_rvec, rvec))
                all_tvec = np.hstack((all_tvec, tvec))
    rvec = np.mean(all_rvec, axis=1).reshape(3, 1)
    tvec = np.mean(all_tvec, axis=1).reshape(3, 1)
    cv2.namedWindow('Undistorted', cv2.WINDOW_KEEPRATIO)
    rot_mat, _ = cv2.Rodrigues(rvec)

    for image in all_files:
        img = cv2.imread(image)
        undistorted = cv2.undistort(img, cam_mat, dist_coeffs, None, cam_mat)
        undistorted = draw_axis(undistorted, rot_mat,
                                tvec, cam_mat, 5*cell_size)
        cv2.imshow('Undistorted', undistorted)
        cv2.waitKey(500)
    file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    file.write("distortion", dist_coeffs)
    file.write("intrinsic", cam_mat)
    file.write("Rot", rot_mat)
    file.write("Trans", tvec)
    file.release()
    
    
if __name__ == "__main__":
    
    images_path = "./output/position/images/950122061707"
    calib_file = "./output/calib_950122061707.yaml"
    file_name = "./output/position_950122061707.yaml"
    
    width = 8
    height = 5
    cell_size = 30
    camera_position(images_path, calib_file,  file_name, width, height, cell_size)