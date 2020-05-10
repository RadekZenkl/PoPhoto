import rawpy
import cv2
import numpy as np


def read_pol_img_set(set_name="", folder_path="RAW"):
    """
    :param folder_path: path to images folder
    :param set_name: name of 4 polarized images
    :return: return numpy array with rgb, degree and angle of polarization
    """

    # Read in images
    img_names = ["m45", "0", "45"]
    imgs = []

    for name in img_names:
        imgs.append(rawpy.imread(folder_path + "/" + set_name + "_" + name + ".CR2"))

    # Demosaic and convert to RGB
    for i in range(len(imgs)):
        img = imgs[i].postprocess(
                output_bps=16,
                demosaic_algorithm=0,
                use_camera_wb=True,
                no_auto_bright=True)

        imgs[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calculate degree of polarization and polarization direction with stokes parameters
    i = imgs[0] + imgs[2]
    q = imgs[0] - imgs[2]
    u = 2*imgs[1] - imgs[0] - imgs[2]

    i_mask = (i == 0)

    dop = np.divide(np.sqrt(np.multiply(q, q) + np.multiply(u, u)), i)
    theta = np.arctan2(u, q) / 2
    # TODO adjust to 0 orientation

    # TODO keep working with 16bit - looks like a loss of information, Cast only when previewing
    # cast to 8bit
    dop[i_mask] = 0
    dop[dop > 1] = 0
    dop = (256*dop).astype('uint8')

    theta = ((theta/np.pi + 1/2) * 180.0).astype('uint8')

    # Add new layers from individual channels
    # dop_r, dop_g, dop_b = cv2.split(dop)
    # imgs.append(dop_r)
    # imgs.append(dop_g)
    # imgs.append(dop_b)
    #
    # theta_r, theta_g, theta_b = cv2.split(theta)
    #
    # imgs.append(theta_r)
    # imgs.append(theta_g)
    # imgs.append(theta_b)

    imgs.append(dop)
    imgs.append(theta)

    return imgs


def stretch_images(images):
    """
    :param images:
    :return: stretched images
    """
    # TODO seems a bit fishy
    stretched_imgs = images
    dop_yuv = cv2.cvtColor(images[3], cv2.COLOR_BGR2YUV)
    theta_yuv = cv2.cvtColor(images[4], cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    dop_yuv[:, :, 0] = cv2.equalizeHist(dop_yuv[:, :, 0])
    theta_yuv[:, :, 0] = cv2.equalizeHist(theta_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    stretched_imgs[3] = cv2.cvtColor(dop_yuv, cv2.COLOR_YUV2BGR)
    stretched_imgs[4] = cv2.cvtColor(theta_yuv, cv2.COLOR_YUV2BGR)

    return stretched_imgs


def filter_images(images):
    """
    :param images:
    :return: filtered_images
    """

    filtered_images = images
    filtered_images[3] = cv2.medianBlur(images[3], 5)
    filtered_images[4] = cv2.medianBlur(images[4], 5)

    return filtered_images


def preview_images(images):
    """
    :param images: a
    :return:
    """

    for image, image_id in zip(images, range(len(images))):
        image_name = 'image_' + str(image_id)
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        cv2.imshow(image_name, image)
        cv2.resizeWindow(image_name, 800, 600)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
