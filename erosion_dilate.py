import cv2
import numpy as np
import matplotlib.pyplot as plt

def erosion_dilate_masks(img1):
    kernel_sizes = [(2, 2), (3, 3), (4, 4), (5, 5)]
    
    masks = []
    previous_mask = np.zeros_like(img1)
    
    prev_image = img1.copy()
    num = 0
    for kernel_size in kernel_sizes:
        kernel = np.ones(kernel_size, np.uint8)
        
        dilated_image = cv2.dilate(prev_image, kernel)
        eroded_image = cv2.erode(prev_image, kernel)
        
        dilated_added = dilated_image - prev_image
        eroded_removed = prev_image - eroded_image
        
        # 存储新增减少部分
        current_mask = np.zeros_like(img1, dtype=np.uint8)
        current_mask[dilated_added == 255] = 255  # 新增的像素为 1
        current_mask[eroded_removed == 255] = 255  # 减少的像素为 1
        if num > 0:
            current_mask -= masks[num - 1]
        num += 1
        
        masks.append(current_mask)
        
    return masks

if __name__ == '__main__':

    image = cv2.imread('rgb1.png', cv2.IMREAD_GRAYSCALE)
    masks = erosion_dilate_masks(image)
    num = 0
    for mask in masks:
        mask = mask.astype(np.uint8)
        cv2.imwrite('erosion_img_' + str(num) + '_rgb1.png', mask)
        num += 1