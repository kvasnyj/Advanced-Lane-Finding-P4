import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
from Line import Line

h, w = None, None

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255

    gradmag = (gradmag / scale_factor).astype(np.uint8)

    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)

    dir = np.arctan2(abs_sobely, abs_sobelx)

    dir_binary = np.zeros_like(dir)
    dir_binary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1

    return dir_binary

def color_threshold(img, thresh=(0, 255)):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

      # Threshold color channel
    color_binary = np.zeros_like(s_channel)
    color_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    return color_binary

def define_warper():
    src = np.float32([
        [255, 685],
        [1050, 685],
        [590, 455],
        [695, 455]
    ])

    dst = np.float32([
        [320, 685],
        [950, 685],
        [320, 0],
        [950, 0]
    ])

    return src, dst

def warper(img, debug = False):
    # Compute and apply perpective transform
    M = cv2.getPerspectiveTransform(warp_src, warp_dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image

    if debug:
        plt.imshow(warped)
        plt.pause(0)

    return warped

def peaks_histogram(img, debug = False):
    left_fitx, right_fitx, yvals = [],[],[]
    p01, p02, wide0 = 0, 0, 0

    topx = h - int(h/4)

    for i in range(0, topx):
        if i == 0:
            histogram = np.sum(img[topx:, :], axis=0)
        else:
            histogram = np.sum(img[topx-i:-i, :], axis=0)

        offset = 100
        p1 = np.argmax(histogram[offset:w / 2])+offset
        if p01>0 and abs(p1-p01)>200: continue

        p2 = w / 2 + np.argmax(histogram[w / 2:1200])
        if p02>0 and abs(p2-p02)>200: continue

        wide = p2 - p1
        if wide0 > 0 and abs(wide0 - wide) > 200: continue

        p01, p02, dist0 = p1, p2, wide

        left_fitx.append(p1)
        right_fitx.append(p2)
        yvals.append(h-i)

    if debug:
        print(left_fitx)
        print(right_fitx)
        plt.plot(histogram)
        plt.show()
    return left_fitx, right_fitx, yvals

def draw_lines(img, x1, y1, x2, y2, color=[255, 0, 0], thickness=2):
    cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)

def fillPoly(undist, warped, left_fitx, right_fitx, yvals):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(warp_dst, warp_src)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def pipeline(img, debug = False):
    img = gaussian_blur(img, kernel_size=5)

    ksize = 5  # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(60, 255))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.65, 1.05))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    color_binary = color_threshold(img, thresh=(160, 255))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    total_binary = np.zeros_like(combined)
    total_binary[(color_binary > 0) | (combined > 0)] = 1

    vertices = np.array([[(100, h), (450, 400), (800, 400), (1200, h)]], dtype=np.int32)
    img = region_of_interest(total_binary, vertices)
    img[img!=0] = 1

    if debug:
        plt.imshow(img)
        plt.pause(0)

    return total_binary

def curvature(leftx, rightx, yvals, debug = False):
    yvals = np.float32(yvals)
    leftx = np.float32(leftx)
    rightx = np.float32(rightx)

    left_fit = np.polyfit(yvals, leftx, 2)
    right_fit = np.polyfit(yvals, rightx, 2)

    # Define y-value where we want radius of curvature
    y_eval = np.max(yvals)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                                 /np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                    /np.absolute(2*right_fit[0])

    if debug: print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(yvals * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    if debug: print(left_curverad, 'm', right_curverad, 'm')

    yvals = np.arange(h - h / 2, h, 1.0)
    left_fitx = sanity_check(left_lane, left_fit, yvals,  left_curverad)
    right_fitx = sanity_check(right_lane, right_fit, yvals, right_curverad)

    return left_fitx, right_fitx, yvals

def sanity_check(lane, polyfit, yvals, curvature, debug = False):
    if lane.polyfit== None: #new object
        lane.radius_of_curvature = curvature
        lane.polyfit = polyfit
        lane.detected = True
    elif abs(lane.radius_of_curvature - curvature) < 1000:
        lane.radius_of_curvature = curvature
        lane.polyfit = polyfit
        lane.detected = True
    else:
        if debug: print(lane.radius_of_curvature, curvature, lane.radius_of_curvature - curvature)
        lane.detected = False

    return lane.polyfit[0] * yvals ** 2 + lane.polyfit[1] * yvals + lane.polyfit[2]


def process_image(src_img, debug = False):
    global h, w
    if h == None: h = src_img.shape[0]
    if w == None: w= src_img.shape[1]

    img = pipeline(src_img)
    img = undistort(img)
    img = warper(img)

    leftx, rightx, yvals = peaks_histogram(img)
    left_fitx, right_fitx, yvals = curvature(leftx, rightx, yvals)
    img = fillPoly(src_img, img, left_fitx, right_fitx, yvals)

    if debug:
        plt.imshow(img)
        plt.pause(0)

    return img


with open("distort.p", "rb") as input_file:
    e = pickle.load(input_file)
    mtx = e["mtx"]
    dist = e["dist"]
warp_src, warp_dst = define_warper()

left_lane = Line()
right_lane = Line()

from moviepy.editor import VideoFileClip
clip = VideoFileClip('project_video.mp4')#.subclip(0, 2)
new_clip = clip.fl_image(process_image)
new_clip.write_videofile('project_video_result.mp4', audio=False)
clip.save_frame("workbook/pv1.jpeg", t=0)

image = mpimg.imread('workbook/pv1.jpeg')
new_image = process_image(image, True)

