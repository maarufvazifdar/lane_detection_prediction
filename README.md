# Lane Detection and Prediction

## Pipeline:
1. The given video is read frame-by-frame, and a region of interest points determined(Figure 4), perspective matrix is obtained using cv2.getPerspectiveTransform function and cv2.warpPerspective is used to warp the image into birds-eye perspective (Figure 5).
2. Warped image is converted to HSL colorspace , lower and upper HSL values are determined for yellow and white lines and using cv2.inRange function masks for yellow (Figure 6) and white (Figure 7) lanes are created.
3. From the yellow and white masks 2 nd order polynomials are fitted using numpy.polyfit function.
4. cv2.polylines is used to mark the yellow and white lines in red and green colors (Figure 9). Lane region is shaded using cv2.fillPoly and points of yellow and white lines .
5. Center of the lane is calculated from points of yellow and white lines and cv2.arrowedLine is used to plot arrows for displaying turn direction (Figure 10).
6. Observing the number of pixels along x and y that make the lane, pixels to meters factors are determined.
7. With the pixels to meters factors, curve is fitted on the center lane of the lane mapped into meters.
8. Radius of curvature is calculated and the slope of points on the center of the lane is used to predict the turn direction.
9. Finally, the image is warped back to camera perspective and superimposed on the input frame. (Figure 11)

### Conditions for pipeline to work:  
The pipeline should work well for similar videos once ROI is
adjusted, lightning conditions are good and lanes are visible clearly. HSL thresholds are
set for yellow and white colors but might need to be adjusted. The pipeline does not
account for any markings on the road like zebra crossings, bike lanes, HOV lanes or
turn markings so that can affect the lane detection.
