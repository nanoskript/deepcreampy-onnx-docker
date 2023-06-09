"""
Cleaned from: DeepCreamPy/decensor.py
"""

import sys

import numpy as np
from PIL import Image

from predict import run_predictions

sys.path.append("DeepCreamPy/libs")
from utils import find_regions, image_to_array, expand_bounding

# Green.
MASK_COLOR = [0, 1, 0]


def find_mask(colored):
    mask = np.ones(colored.shape, np.uint8)
    i, j = np.where(np.all(colored[0] == MASK_COLOR, axis=-1))
    mask[0, i, j] = 0
    return mask


# TODO: Profile memory usage.
def decensor(ori: Image, colored: Image, is_mosaic: bool):
    # save the alpha channel if the image has an alpha channel
    has_alpha = False
    if ori.mode == "RGBA":
        has_alpha = True
        alpha_channel = np.asarray(ori)[:, :, 3]
        alpha_channel = np.expand_dims(alpha_channel, axis=-1)
        ori = ori.convert('RGB')

    ori_array = image_to_array(ori)
    ori_array = np.expand_dims(ori_array, axis=0)

    if is_mosaic:
        # if mosaic decensor, mask is empty
        colored = colored.convert('RGB')
        color_array = image_to_array(colored)
        color_array = np.expand_dims(color_array, axis=0)
        mask = find_mask(color_array)
    else:
        mask = find_mask(ori_array)

    # colored image is only used for finding the regions
    regions = find_regions(colored.convert('RGB'), [v * 255 for v in MASK_COLOR])
    print("Found {region_count} censored regions in this image!".format(region_count=len(regions)))
    if len(regions) == 0 and not is_mosaic:
        print("No green (0,255,0) regions detected! Make sure you're using exactly the right color.")
        return ori

    output_img_array = ori_array[0].copy()
    prediction_requests = []
    bounding_boxes = []
    for region in regions:
        bounding_box = expand_bounding(ori, region, expand_factor=1.5)
        crop_img = ori.crop(bounding_box)

        # convert mask back to image
        mask_reshaped = mask[0, :, :, :] * 255.0
        mask_img = Image.fromarray(mask_reshaped.astype('uint8'))
        # resize the cropped images

        crop_img = crop_img.resize((256, 256), resample=Image.NEAREST)
        crop_img_array = image_to_array(crop_img)

        # resize the mask images
        mask_img = mask_img.crop(bounding_box)
        mask_img = mask_img.resize((256, 256), resample=Image.NEAREST)

        # convert mask_img back to array
        mask_array = image_to_array(mask_img)
        # the mask has been upscaled so there will be values not equal to 0 or 1

        if not is_mosaic:
            a, b = np.where(np.all(mask_array == 0, axis=-1))
            crop_img_array[a, b, :] = 0.

        # Normalize.
        crop_img_array = crop_img_array * 2.0 - 1

        # Queue prediction request.
        bounding_boxes.append(bounding_box)
        prediction_requests.append([crop_img_array, mask_array])

    # Run predictions for this batch of images.
    predictions = run_predictions(prediction_requests, is_mosaic)

    # Apply predictions.
    for pred_img_array, bounding_box, region in zip(predictions, bounding_boxes, regions):
        pred_img_array = (255.0 * ((pred_img_array + 1.0) / 2.0)).astype(np.uint8)

        # scale prediction image back to original size
        bounding_width = bounding_box[2] - bounding_box[0]
        bounding_height = bounding_box[3] - bounding_box[1]

        # convert np array to image
        pred_img = Image.fromarray(pred_img_array.astype('uint8'))
        pred_img = pred_img.resize((bounding_width, bounding_height), resample=Image.BICUBIC)
        pred_img_array = image_to_array(pred_img)
        pred_img_array = np.expand_dims(pred_img_array, axis=0)

        # copy the decensored regions into the output image
        for i in range(len(ori_array)):
            for col in range(bounding_width):
                for row in range(bounding_height):
                    bounding_width_index = col + bounding_box[0]
                    bounding_height_index = row + bounding_box[1]
                    if (bounding_width_index, bounding_height_index) in region:
                        output_img_array[bounding_height_index][bounding_width_index] \
                            = pred_img_array[i, :, :, :][row][col]

    output_img_array = output_img_array * 255.0

    # restore the alpha channel if the image had one
    if has_alpha:
        output_img_array = np.concatenate((output_img_array, alpha_channel), axis=2)

    print("Decensored image. Returning it.")
    return Image.fromarray(output_img_array.astype('uint8'))