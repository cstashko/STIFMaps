import numpy as np

# Normalize an image to have intensity values between 0 and 1.
# This function finds the 1st and 99th percentiles of the pixel values
# and uses them to scale the image. If the image is uniform (all pixel values are the same),
# it returns a blank image (all zeros).
def norm_pic(im):
    # Find the thresholds
    hmin = np.quantile(im.flatten(), .01)
    hmax = np.quantile(im.flatten(), .99)

    # Return a blank image if the image is uniform
    if hmax == hmin:
        return np.zeros_like(im, dtype=np.float32)  # or return np.ones_like(im)

    # Create the new thresholded image
    im2 = (im - hmin) / (hmax-hmin)
    im2 = np.clip(im2, 0, 1)
    
    return im2

# Given the step size between squares and the scale factor between the input
# image and the training data, find the actual step size to use in the input image
# Note that the output step size must be an even number 
def get_step(step, scale_factor):
    step = step / scale_factor

    if round(step) % 2 == 0:
        return round(step)
    elif round(step) == int(step):
        return round(step)+1
    else:
        return int(step)
