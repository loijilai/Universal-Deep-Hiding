import shutil
import os

# Copy the first 1000 images from the original dataset to the new dataset
src = "/home/lai/Research/coco/images/val2017/val_class"
dest = "/home/lai/Research/coco/small_images/val2017/val_class"

def copy_images(src, dest, num_images):
    # list the first num_images images in src
    img_names = os.listdir(src)
    for img_name in img_names[:num_images]:
        shutil.copy(os.path.join(src, img_name), dest)

copy_images(src, dest, 100)