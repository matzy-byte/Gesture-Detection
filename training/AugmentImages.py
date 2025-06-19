import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tqdm import tqdm


DATASET_DIR = "./training/gestures-dataset"
AUGMENTATIONS_PER_IMAGE = 5
DATAGEN = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode="nearest"
)


def augment_image(img_path, save_dir, prefix, count):
    img = load_img(img_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    aug_iter = DATAGEN.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix=prefix, save_format="jpg")
    
    for i in range(count):
        next(aug_iter)


def main():
    for label_dir in os.listdir(DATASET_DIR):
        full_label_path = os.path.join(DATASET_DIR, label_dir)
        if not os.path.isdir(full_label_path):
            continue

        images = [f for f in os.listdir(full_label_path) if f.lower().endswith(("jpg"))]

        for img_name in tqdm(images):
            img_path = os.path.join(full_label_path, img_name)
            img_base = os.path.splitext(img_name)[0]
            augment_image(img_path, full_label_path, f"aug_{img_base}", AUGMENTATIONS_PER_IMAGE)


if __name__ == "__main__":
    main()
