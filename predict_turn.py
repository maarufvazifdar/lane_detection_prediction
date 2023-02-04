import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('./dataset/challenge.mp4')

video_name = './predict_turn.mp4'

# Create a VideoWriter object
output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(
    *'MP4V'), 25, (1920, 720))

output_frame = np.zeros((720, 1920, 3), dtype=np.uint8)
frame_count = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # # Flip input frame horizontally
    # frame = cv2.flip(frame, 1)

    if ret:
        # Region of Interest lane_pts
        roi_pts = np.array([[520, 460],
                            [750, 460],
                            [1120, 670],
                            [130, 670]], np.float32)

        roi_img = np.copy(frame)
        # roi_draw for plotting lane_pts
        roi_draw = np.array([[520, 460],
                             [750, 460],
                             [1120, 670],
                             [130, 670]], np.int32)
        roi_draw = roi_draw.reshape((-1, 1, 2))
        cv2.polylines(roi_img, [roi_draw], True, (0, 0, 255), 2)

        roi_warped_pts = np.array([[0, 0],
                                   [1279, 0],
                                   [1279, 719],
                                   [0, 719]], np.float32)

        # Warping region of interest to get lanes in birds-eye-perspective
        P = cv2.getPerspectiveTransform(roi_pts, roi_warped_pts)
        warped = cv2.warpPerspective(frame, P, (1280, 720))

        # BGR to HLS
        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
        lane = np.zeros_like(frame)

        # Yellow line Thresholding
        l_b_yellow = np.array([13, 0, 110])
        u_b_yellow = np.array([24, 200, 255])
        mask_yellow = cv2.inRange(hls, l_b_yellow, u_b_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

        # Indixes of all yellow pixels in yellow mask
        x_yellow, y_yellow = np.where(mask_yellow == [255])
        yellow_curve = np.polyfit(x_yellow, y_yellow, 2)

        # Points of Yellow curve
        ls = np.linspace(0, 720, 100)
        yellow_x = ls
        yellow_y = np.polyval(yellow_curve, yellow_x)
        yellow_pts = (np.asarray([yellow_y, yellow_x]).T).astype(
            np.int32)
        cv2.polylines(lane, [yellow_pts], False, (0, 0, 255), 20)

        # White line Thresholding
        l_b_white = np.array([0, 212, 0])
        u_b_white = np.array([255, 255, 255])
        mask_white = cv2.inRange(hls, l_b_white, u_b_white)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

        # Indixes of all white pixels in white mask
        x_white, y_white = np.where(mask_white == [255])
        white_curve = np.polyfit(x_white, y_white, 2)

        # Points of White curve
        white_x = ls
        white_y = np.polyval(white_curve, white_x)
        white_pts = (np.asarray([white_y, white_x]).T).astype(
            np.int32)
        cv2.polylines(lane, [white_pts], False, (0, 255, 0), 20)

        # Lane Region
        white_pts_flip = np.flip(white_pts, axis=0)
        lane_pts = np.concatenate((yellow_pts, white_pts_flip), axis=0)
        # lane = np.zeros_like(frame)
        cv2.fillPoly(lane, pts=[lane_pts], color=(255, 0, 0))

        # Lane Centre
        lane_centre = np.empty_like(yellow_pts)
        lane_centre[:, 0] = (yellow_pts[:, 0] + white_pts[:, 0]) / 2
        lane_centre[:, 1] = yellow_pts[:, 1]

        # Drawing arrows on centre of lane
        for i in range(0, 100, 20):
            lane = cv2.arrowedLine(lane, (lane_centre[i + 10][0],
                                          lane_centre[i + 10][1]),
                                   (lane_centre[i][0],
                                    lane_centre[i][1]),
                                   (0, 0, 255), 15, tipLength=0.4)

        # Pixels to meters mapping
        xm = 3.7 / 1000    # meters per pixel along x
        ym = 3 / 140    # meters per pixel along y

        lane_centre_m = np.polyfit(
            lane_centre[:, 1] * ym, lane_centre[:, 0] * xm, 2)
        # Radius of curvature of lane in meters
        rad_of_cur = ((1 + (2 * lane_centre_m[0] * 720 * ym + lane_centre_m
                            [1]) ** 2) ** 1.5) / np.absolute(2 *
                                                             lane_centre_m[0])
        # Slope of lane
        lane_slope = ((lane_centre[50][1] - lane_centre[25][1])
                      / (lane_centre[50][0] - lane_centre[25][0]))

        # Inverse warping lanes into camera perspective
        P2 = cv2.getPerspectiveTransform(roi_warped_pts, roi_pts)
        lane_inv_warp = cv2.warpPerspective(lane, P2, (1280, 720))

        # Detected lanes superimposed on input frames
        output_img = cv2.bitwise_or(frame, lane_inv_warp)

        if lane_slope < 0:
            cv2.putText(output_img, 'Right Turn', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                        1, cv2.LINE_AA)
        elif lane_slope > 0:
            cv2.putText(output_img, 'Left Turn', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                        1, cv2.LINE_AA)
        else:
            cv2.putText(output_img, 'Straight Road', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                        1, cv2.LINE_AA)
        cv2.putText(output_img,
                    'Curve Radius: ' + str(round(rad_of_cur, 1)) + 'm',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1, cv2.LINE_AA)

        # Creating Output Frame
        roi_img = cv2.resize(roi_img, (320, 180))

        warped_img = cv2.resize(warped, (320, 180))

        y_w_mask_img = cv2.bitwise_or(mask_yellow, mask_white)
        y_w_mask_img = np.dstack([y_w_mask_img, y_w_mask_img, y_w_mask_img])

        y_w_mask_img2 = np.copy(y_w_mask_img)
        lane_fit_img = cv2.polylines(
            y_w_mask_img2, [yellow_pts], False, (0, 0, 255), 10)
        lane_fit_img = cv2.polylines(
            y_w_mask_img2, [white_pts], False, (0, 255, 0), 10)

        y_w_mask_img = cv2.resize(y_w_mask_img, (320, 180))

        lane_fit_img = cv2.resize(lane_fit_img, (320, 180))
        lane_img = cv2.resize(lane, (640, 360))

        # Final image
        output_frame[0:720, 0:1280] = output_img
        # region of interest image
        output_frame[0:180, 1280:1600] = roi_img
        # warped image
        output_frame[0:180, 1600:1920] = warped_img
        # yellow and white lanes mask
        output_frame[180:360, 1280:1600] = y_w_mask_img
        # line fititng on yellow and white lanes
        output_frame[180:360, 1600:1920] = lane_fit_img
        # warped lane region with arrows
        output_frame[360:720, 1280:1920] = lane_img

        cv2.putText(output_frame, 'Region of Interest',
                    (1290, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1, cv2.LINE_AA)
        cv2.putText(output_frame, 'Warped ROI',
                    (1610, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1, cv2.LINE_AA)
        cv2.putText(output_frame, 'Lane Mask',
                    (1290, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1, cv2.LINE_AA)
        cv2.putText(output_frame, 'Curves fitted on Lanes',
                    (1610, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1, cv2.LINE_AA)
        cv2.putText(output_frame, 'Lane Region Warped',
                    (1290, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1, cv2.LINE_AA)
        cv2.putText(output_frame, 'Frame No: ' + str(frame_count),
                    (10, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1, cv2.LINE_AA)

        cv2.imshow('Lane Detection', output_frame)

        # Write frames to output video
        output_video.write(output_frame)

        print('Frame:', frame_count)
        frame_count += 1
        cv2.waitKey(10)
        # # Press Q on keyboard to  exit
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
output_video.release()
# Closes all the frames
cv2.destroyAllWindows()
