#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
CS 4391 Homework 2 Programming: Part 1&2 - mean and gaussian filters
Implement the linear_local_filtering() and gauss_kernel_generator() functions in this python script
"""

import cv2
import numpy as np
import math
import sys
 
def linear_local_filtering(
    img: np.uint8,
    filter_weights: np.ndarray,
) -> np.uint8:
    
    """
    Homework 2 Part 1
    Compute the filtered image given an input image and a kernel 
    """

    img = img / 255
    img = img.astype("float32") # input image
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    kernel_size = filter_weights.shape[0] # filter kernel size
    sizeX, sizeY = img.shape
    
    # filtering for each pixel
    for i in range(kernel_size // 2, sizeX - kernel_size // 2):
        for j in range(kernel_size // 2, sizeY - kernel_size // 2):

            # Todo: For current position [i, j], you need to compute the filtered pixel value: img_filtered[i, j] 
            # using the kernel weights: filter_weights and the neighboring pixels of img[i, j] in the
            # kernel_sizexkernel_size local window
            # The filtering formula can be found in slide 3 of lecture 6
            # ********************************
            # Your code is here.
            # ********************************
            #img_filtered[i, j] = 0 # todo: replace 0 with the correct updated pixel value
     
            sum1=0

            for x in range(i-3, i+3):
                for y in range(j-3, j+3):
                    sum1 += img[x, y]*filter_weights[x-i+3,y-j+3] 
                    
            img_filtered[i, j]=sum1
            
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

def gauss_kernel_generator(kernel_size: int, spatial_variance: float) -> np.ndarray:
    
    """
    Homework 2 Part 2
    Create a kernel_sizexkernel_size gaussian kernel of given the variance. 
    """
    
    # Todo: given variance: spatial_variance and kernel size, you need to create a kernel_size x kernel_size gaussian kernel
    # Please check out the formula in slide 15 of lecture 6 to learn how to compute the gaussian kernel weight: g[k, l] at each position [k, l].
    
    kernel_weights = np.zeros((kernel_size, kernel_size))

    for x in range(0, kernel_size):
        for y in range(0, kernel_size):
            kernel_weights[x ,y]=np.exp(-(x**2+y**2)/(2*spatial_var))
                    
    return kernel_weights
 
if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # mean filtering
    box_filter = np.ones((7, 7))/49
    img_avg = linear_local_filtering(img_noise, box_filter) # apply the filter to process the image: img_noise
    cv2.imwrite('results/im_box.png', img_avg)

    # Gaussian filtering
    kernel_size = 7  
    spatial_var = 15 # sigma_s^2 
    gaussian_filter = gauss_kernel_generator(kernel_size, spatial_var)
    gaussian_filter_normlized = gaussian_filter / (np.sum(gaussian_filter)+1e-16) # normalization term
    im_g = linear_local_filtering(img_noise, gaussian_filter_normlized) # apply the filter to process the image: img_noise
    cv2.imwrite('results/im_gaussian.png', im_g)


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


"""
CS 4391 Homework 2 Programming: Part 3 - bilateral filter
Implement the bilateral_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def bilateral_filtering(
    img: np.uint8,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int,
) -> np.uint8:
    
    """
    Homework 2 Part 3
    Compute the bilaterally filtered image given an input image, kernel size, spatial variance, and intensity range variance
    """

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    
    # Todo: For each pixel position [i, j], you need to compute the filtered output: img_filtered[i, j]
    # step 1: compute kernel_sizexkernel_size spatial and intensity range weights of the bilateral filter 
        #in terms of spatial_variance and intensity_variance. 
    # step 2: compute the filtered pixel img_filtered[i, j] using the obtained kernel weights and the neighboring 
        #pixels of img[i, j] in the kernel_sizexkernel_size local window
    # The bilateral filtering formula can be found in slide 15 of lecture 6
    # Tip: use zero-padding to address the black border issue.
    # ********************************
    # Your code is here.
    # ********************************
    
    pad = np.pad(img, (((kernel_size // 2), (kernel_size // 2)), ((kernel_size // 2), (kernel_size // 2))), mode='constant')
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            sum2 = 0
            kernel_weight = 0

            for x in range(-(kernel_size // 2), (kernel_size // 2)+ 1):
                for y in range(-(kernel_size // 2), (kernel_size // 2)+ 1):
                    if (i + x) >= 0 and (i + x) < h and (j + y) >= 0 and (j + y) < w:   
                        spatial_weight = np.exp(-((np.sqrt(x**2 + y**2))**2) / (2 * spatial_variance))
                        intensity_weight = np.exp(-((np.abs(img[i, j] - img[(i + x), (j + y)]))**2) / (2 * intensity_variance))
                        sum2 += img[(i + x), (j + y)] * (spatial_weight * intensity_weight)
                        kernel_weight += (spatial_weight * intensity_weight)

            img_filtered[i, j] = sum2 / kernel_weight

    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # Bilateral filtering
    spatial_variance = 30 # signma_s^2
    intensity_variance = 0.5 # sigma_r^2
    kernel_size = 7
    img_bi = bilateral_filtering(img_noise, spatial_variance, intensity_variance, kernel_size)
    cv2.imwrite('results/im_bilateral.png', img_bi)


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


"""
CS 4391 Homework 2 Programming: Part 4 - non-local means filter
Implement the nlm_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def nlm_filtering(
    img: np.uint8,
    intensity_variance: float,
    patch_size: int,
    window_size: int,
) -> np.uint8:
    
    """
    Homework 2 Part 4
    Compute the filtered image given an input image, kernel size of image patch, spatial variance, and intensity range variance
    """

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    
    # Todo: For each pixel position [i, j], you need to compute the filtered output: img_filtered[i, j] using a non-local means filter
    # step 1: compute window_sizexwindow_size filter weights of the non-local means filter in terms of intensity_variance. 
    # step 2: compute the filtered pixel img_filtered[i, j] using the obtained kernel weights and the pixel values in the search window
    # Please see slides 30 and 31 of lecture 6. Clarification: the patch_size refers to the size of small image patches (image content in yellow, 
    # red, and blue boxes in the slide 30); intensity_variance denotes sigma^2 in slide 30; the window_size is the size of the search window as illustrated in slide 31.
    # Tip: use zero-padding to address the black border issue. 
    # ********************************
    # Your code is here.
    # ********************************
    

    pad = np.pad(img, (((window_size // 2), (window_size // 2)), ((window_size // 2), (window_size // 2))), mode='constant')
    h1, w1 = img.shape
    for i in range((window_size // 2), h1 - (window_size // 2)):
        for j in range(window_size // 2, w1 - window_size // 2):
            patch = img[i - (patch_size // 2):i + (patch_size // 2) + 1, 
                        j - (patch_size // 2):j + (patch_size // 2) + 1]
            filter_weight = np.zeros((window_size, window_size), dtype=np.float32)             
            sum3 = 0
            kernel_weight1 = 0
            
            for u in range(window_size):
                for v in range(window_size):
                    window_patch = img[i + u - (window_size // 2) - (patch_size // 2):i + u - (window_size // 2) + (patch_size // 2) + 1, 
                                       j + v - (window_size // 2) - (patch_size // 2):j + v - (window_size // 2) + (patch_size // 2)+ 1]
                    if window_patch.shape == patch.shape:
                        filter_weight[u, v] = (np.exp(-(np.sum(np.square(patch - window_patch))) / (2.0 * intensity_variance ** 2)))
                    sum3 += filter_weight[u, v] * img[i + u - (window_size // 2), j + v - (window_size // 2)]
                    kernel_weight1 += filter_weight[u, v]

            img_filtered[i, j] = sum3 / kernel_weight1
    
    img_filtered = img_filtered[(window_size // 2):-(window_size // 2), (window_size // 2):-(window_size // 2)]
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

 
if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # Bilateral filtering
    intensity_variance = 1
    patch_size = 5 # small image patch size
    window_size = 15 # serach window size
    img_bi = nlm_filtering(img_noise, intensity_variance, patch_size, window_size)
    cv2.imwrite('results/im_nlm.png', img_bi)


# In[ ]:





# In[ ]:




