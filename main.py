import rawpy
import cv2
import numpy as np
import pandas as pd

import helper_fcts

# image_raw = rawpy.imread("image.CR2")
#
# # a_RGB_8bit = ImageFunctions.preview_raw_image(image_raw)
# img = image_raw.postprocess(
#         output_bps=16,
#         demosaic_algorithm=0,
#         use_camera_wb=True,
#         no_auto_bright=True)
#
# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', RGB_img)
# cv2.resizeWindow('image', 800, 600)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

imgs = helper_fcts.read_pol_img_set(set_name="Salad", folder_path="Salad")
# helper_fcts.preview_images(imgs)

filtered_imgs = helper_fcts.filter_images(imgs)
imgs_stretched = helper_fcts.stretch_images(filtered_imgs)
helper_fcts.preview_images(imgs_stretched)
