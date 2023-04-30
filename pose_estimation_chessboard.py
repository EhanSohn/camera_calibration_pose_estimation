import numpy as np
import cv2 as cv

# The given video and calibration data
input_file = './chessboard.MOV'
K = np.array([[1.66265819e+03, 0, 5.61843322e+02],
              [0, 1.66282728e+03, 9.61376489e+02],
              [0, 0, 1]])
dist_coeff = np.array([3.27714845e-01, -2.56167630e+00, 9.56921489e-04, 6.33329004e-03, 6.61015931e+00])
board_pattern = (8, 6)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(input_file)
assert video.isOpened(), 'Cannot read the given input, ' + input_file

# Prepare a L shape box for simple AR
box_lower = board_cellsize * np.array([[4, 0, 0], [4, 4, 0], [7, 4, 0], [7, 3, 0], [5, 3, 0], [5, 0, 0]])
box_upper = board_cellsize * np.array([[4, 0, -2], [4, 4, -2], [7, 4, -2], [7, 3, -2], [5, 3, -2], [5, 0, -2]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    complete, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the L shape box on the image
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 7)
        cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 7)
        for b, t in zip(line_lower, line_upper):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 7)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) scipy.spatial.transform.Rotation
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()