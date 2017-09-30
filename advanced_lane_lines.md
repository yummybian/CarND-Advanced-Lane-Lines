## Advanced Lane Lines

---

**Advanced Lane Finding Project**

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

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/orig_undistorted.png "Undistorted"
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/lane_screenshot.png "Output"
[video1]: ./output_images/project_video_result.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Preprocess class of the IPython notebook located in "./advanced-lane-finding.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1] 


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used the undistort() to calculate camera calibration matrix and distortion coefficients. It can remove distortion of image and output the undistorted image.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In this step i choose L channel of LUV and b channel of Lab. Because L channel represents luminance which is suit for the white lane line, and b channel is suit for the yellow lane line. Finally combine them together is prefect for the lane line.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In this step I will use birds_eye() to transform the undistorted image to "birdseye" of the lane, and displays 
them in such a way that they appear to be relatively parallel to eachother. It can be convenient for fit polynomials
to the lane line and measure the curvature.

The code for my perspective transform includes a function called `birds_eye()`.  I used the method get_src_dest_warp_points to the source and destination points, and them show below:

``` Python
src = [[  253.   697.]
 [  585.   456.]
 [  700.   456.]
 [ 1061.   690.]]

dest = [[  303.   697.]
 [  303.     0.]
 [ 1011.     0.]
 [ 1011.   690.]]
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After applying calibration, thresholding, and a perspective transform to a road image, I got a binary image.

1. I first token a histogram along all the columns in the lower half of the image like this:
> histogram = np.sum(img[img.shape[0]//2:,:], axis=0)


2. According to the above histogram, you can figure out the left and right peaks, which represent the left and right lane line represently. 

3. So I used the above peak values as the start point of the left and right lines. Then I used the fixed window move the bottom to the top of image. During the procedure, I needed to adjust the center of sliding window.

4. Finally I got the line pixel positions, and fitted a second order polynomial.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I refered a awesome [tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) to calculate the radius of curvature of the lane.

I used the below formula to calculate the center position
> center = abs(middle-of-x-axis - ((position-of-left-line-bottom + position-of-right-line-bottom)/2))

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is a screenshot of video output:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

At the beginning, I used sliding windows to search each frame. But I found it can't work well. So I imporved it, and searched in a margin around the previous line position.

- **where will your pipeline likely fail?**
I guess it could tend to be failed if exist similar line near the lane lines.

 
- **what could you do to make it more robust?**

1. Investigate other colour space and their channels to see which still shows the lanes the best over the concrete sections of road.
2. Check the mean squared error between subsequent polynomial fits. If the error is greater than a determined threshold then drop the frame (not the frame itself but the lane line finding result).
