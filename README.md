# Lane-detection-multimedia
## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

A camera lens bends the image slightly, so straight lines (for example the edges of a road) look curved, especially near the edges of the frame. If this is not corrected, the later curvature estimate would also be wrong. For calibration I used the chessboard images from the camera_cal folder. I know what a chessboard should look like (a regular grid of straight lines), and on the images I can see what it actually looks like, so the difference between the two describes the distortion.

The code is in the function calibrate_camera in calibration.py:

I prepare the "ideal" 3D coordinates of the inner corners of the board: (0,0,0), (1,0,0), ... (8,5,0). The board has 9x6 inner corners. The z coordinate is always 0 because the board is flat, and the unit is one square of the board. The real size of a square does not matter for removing distortion.
For every image in camera_cal I convert it to grayscale and call cv2.findChessboardCorners, which returns the corner positions in pixels.
For every image where the corners were found, I append the ideal points to objpoints and the detected points to imgpoints.
I call cv2.calibrateCamera, which computes the camera matrix mtx (focal length and image center) and the distortion coefficients dist from all these pairs.
I save the result to calibration.npz, so the other scripts only need to load it.

The corners were found on 17 out of 20 images. On calibration1, calibration4 and calibration5 the function did not find 9x6 corners, probably because the board is not fully inside the frame there. The mean reprojection error (RMS) is 1.185 pixels, which means the points predicted by the model differ from the detected ones by a little more than one pixel on average. That is good enough for gently undistorting road images.

The correction itself is done by the function undistort (cv2.undistort). Example on a calibration image:

Calibration: original and undistorted

On the original the edges of the board are bent, and on the undistorted image they are straight

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I apply the same undistort function with the same matrix to the road images (main part of thresholding.py). The difference is subtle and is mostly visible near the edges of the image, for example on the hood and the tree tops:

Road image: original and undistorted

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The function binary_threshold in thresholding.py combines three signals:

S channel (saturation) of the HLS color space, threshold 170 to 255. The yellow line has high saturation, and this channel is fairly robust to changes in brightness. OpenCV loads images as BGR, so the conversion is COLOR_BGR2HLS.
L channel (lightness) of the HLS color space, threshold 200 to 255. The white line has low saturation, so the S channel barely detects it, but it is bright. Without the L channel the dashed white line on the right of test1 was barely visible, and with it the line is clear.
Sobel gradient in the x direction, threshold 20 to 100 on a 0 to 255 scale. The lane lines are almost vertical, so the intensity changes sharply along the horizontal direction at their edges. I take the absolute value (what matters is the strength of the change, not its sign) and scale it to 0 to 255, so the thresholds do not depend on the brightness of the image.

I combine the results with a logical OR: a pixel is kept if any of the methods detected it, because each method has weaknesses in different places.

Binary thresholding steps, with and without the L channel

The L channel also picks up things that are not lane lines (the white sky, white cars), but these are outside the lane and are removed by the next step, because after the perspective transform only the trapezoid around the lane is used. I checked the thresholds on several images (test3, test4, test5), and no large white areas appeared on the road surface itself.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The functions get_transforms and warp in perspective.py. In the camera image the lane lines converge towards the horizon, so they are not parallel. In the top-down view they are parallel and vertical (if the road is flat), which makes fitting a curve and computing the curvature easier.

I picked 4 points on an image with a straight road (straight_lines1.jpg) and defined where they should end up:

	Source (SRC)	Destination (DST)
top left	(580, 460)	(320, 0)
top right	(705, 460)	(960, 0)
bottom right	(1127, 720)	(960, 720)
bottom left	(203, 720)	(320, 720)

I started from approximate values and checked them on the straight road: both lines in the top-down view should be vertical and parallel. In the end they match the green lines that I drew at the DST positions. The margins of 320 px on the left and right leave room to see curved lines too.

cv2.getPerspectiveTransform returns the matrix M, and by swapping source and destination I also get the inverse matrix Minv, which I need later to draw the lane back onto the original image. I compute the matrices once, not for every frame. For binary images I use INTER_NEAREST, as in the lecture example, so the image stays strictly 0/1.

Trapezoid on the image and the top-down view

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions find_lane_pixels and fit_polynomial in lane_finding.py:

Histogram. I sum the white pixels along each column in the bottom half of the top-down binary image. Because the lines are almost vertical, the histogram has two peaks at their positions: the highest one in the left half is the start of the left line, and the highest one in the right half is the start of the right line.
Sliding windows. From the bottom to the top I place 9 windows (width ±100 px around the current position). I collect the white pixels that fall inside the window, and if there are more than 50, I re-center the next window on their mean position. This way the windows follow the curve.
Second-order polynomial. np.polyfit fits x = A·y² + B·y + C separately for the left and the right line. I fit x as a function of y and not the other way around, because the lines are almost vertical and for one x there can be many y values.

Windows, detected pixels and the fit

Red pixels belong to the left line, blue pixels to the right line, and the yellow curves are the fit.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function measure_lane in measurements.py.

Converting to meters. The polynomial from the previous step is in pixels, so before fitting I convert the coordinates to meters:

XM_PER_PIX = 3.7 / 640: a lane is about 3.7 m wide, and in the top-down view there are 640 px between the DST lines.
YM_PER_PIX = 30 / 720: I assume that the visible part of the road in the top-down view corresponds to about 30 m. This is an estimate, not a measurement, so the radius values should be taken as approximate.

Radius of curvature. For a curve x = A·y² + B·y + C the radius at a point y is

R = (1 + (2·A·y + B)²)^(3/2) / |2·A|

I compute it at the bottom of the image (the point closest to the vehicle) for both lines and take the average. The smaller A is, the flatter the curve and the larger the radius.

Vehicle position. I assume that the camera is mounted in the middle of the vehicle, so the vehicle is at the image center (x = 640). The lane center is the midpoint between the left and right line at the bottom of the image. The difference, converted to meters, is the offset of the vehicle from the lane center.

Measured values on the test images:

Image	Radius	Offset from center	Lane width
straight_lines1	16508 m	0.03 m right	3.58 m
straight_lines2	21499 m	0.01 m right	3.55 m
test1	1733 m	0.08 m left	3.54 m
test2	616 m	0.23 m left	3.47 m
test3	1066 m	0.09 m left	3.63 m
test4	1036 m	0.36 m left	3.95 m
test5	8573 m	0.01 m left	3.82 m
test6	934 m	0.24 m left	3.77 m

The lane width serves as a sanity check: on the straight images it is about 3.55 m, close to the expected 3.7 m. On straight sections the radius is very large, as it should be. The biggest deviations are on test4 and test5, the images with the most shadows and noise.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The function draw_lane in pipeline.py builds a closed polygon from the left and right curves (the left one goes from top to bottom and the right one from bottom to top), fills it with green in the top-down view and warps it back to the original perspective with the inverse matrix Minv. It is then blended with transparency over the undistorted image (cv2.addWeighted). The function process_image ties the whole pipeline together and writes the radius and the offset on the image. If the radius is larger than 5000 m, "straight" is printed instead. I chose this threshold myself, to make the display easier to read.

Result on test2

The results for all test images are in output_images/ (the *_result.jpg files).

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Output video: project_video03_out.mp4

The video is processed by video.py, class LaneTracker. On single images the lines wobble slightly, so I added three things:

Sanity check. A fit is accepted only if the lane is between 450 and 800 px wide (roughly 2.6 to 4.6 m) and if the width at the top, middle and bottom of the image does not differ by more than 150 px, i.e. the lines are roughly parallel. I chose these thresholds approximately.
Smoothing. I draw the average of the coefficients of the last 10 accepted fits. For the radius I use the median, because on straight sections some values become huge and spoil the average.
Fallback. If a frame does not pass the check, the previous smoothed fit is used, so the lane does not disappear.

The video project_video03.mp4 (1280x720, 1260 frames) was processed without major problems, and the sanity check rejected 59 frames (about 5%).

The video is saved with the mp4v codec, so the browser on GitHub may not play it. In that case it needs to be downloaded and played in a local player.


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Different cameras and resolutions. Part of the test images (960x540) and videos (project_video01, project_video02) have a different resolution than the chessboard images. A calibration is valid only for a specific camera and resolution, so undistorting them with the matrix from another camera would give a wrong result. That is why the code processes only material of resolution 1280x720 and skips the rest. Of the videos I processed only project_video03, and I did not process the challenge videos.

Shadows and noise. The pipeline is weakest on images with shadows and trees (test4, test5). The lane width there is 3.95 m and 3.82 m, which deviates from the other images. On test5 the radius is 8573 m, although the road in the image turns slightly, so the curvature is probably underestimated. My guess is that white noise next to the left line attracted the windows, but I did not check this separately. Because of this, "straight" is printed on that image. The thresholds for S, L and Sobel were tuned on these test images, so under different lighting (night, rain, strong sun) they can fail.

Estimates in the computation. YM_PER_PIX is an estimate and not a measured quantity, so the absolute radius values are approximate. I also assume that the camera is in the middle of the vehicle. The 5000 m threshold for the "straight" label and the sanity-check thresholds in the video are my own decisions, and I tuned them on a single video only.

Fixed perspective points. The SRC points were chosen for a flat road and a camera that does not move. On uphill or downhill sections or when the vehicle pitches strongly, the trapezoid would no longer match the road, and the lines would stop being parallel in the top-down view.

What would make it more robust.

In the video, instead of searching with sliding windows from scratch in every frame, the search could be limited to the area around the previous fit, which is faster and more resistant to noise.
The window position could be set with the median instead of the mean of the pixels, so isolated noise has less influence.
Adaptive thresholds (for example based on the brightness of the frame) would help with shadows.
Averaging over 10 frames reduces wobbling, but it also adds a bit of lag when the road changes suddenly.

