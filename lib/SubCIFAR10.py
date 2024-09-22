import sys
sys.path.append('E:\\summerwork\\PaConvert-master/paddle_project/utils')
#from  utils import paddle_aux
import os
import random
import pickle
import argparse
import numpy as np
from PIL import Image
random.seed(0)
parser = argparse.ArgumentParser('Generate SubCIFAR10', add_help=False)
parser.add_argument('--data-path', type=str, required=True, help='dataset path'
    )
args = parser.parse_args()
data_path = args.data_path
subCIFAR10_name = 'subCIFAR10'
subCIFAR10_path = os.path.join(os.path.dirname(data_path), subCIFAR10_name)
if not os.path.exists(subCIFAR10_path):
    os.makedirs(subCIFAR10_path)
batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
    'data_batch_5']
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
    'horse', 'ship', 'truck']
class_to_images = {cls: [] for cls in classes}
for batch in batches:
    with open(os.path.join(data_path, batch), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        for label, image in zip(data[b'labels'], data[b'data']):
            image = image.reshape((32, 32, 3)).astype(np.uint8) * 255
            class_to_images[classes[label]].append(image)
for cls, images in class_to_images.items():
    cls_path = os.path.join(subCIFAR10_path, cls)
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)
    selected_images = random.sample(images, 100)
    for idx, image_data in enumerate(selected_images):
        image_array = image_data
        np.save(os.path.join(cls_path, f'{idx}.npy'), image_array)
with open(os.path.join(subCIFAR10_path, 'info.txt'), 'w') as f:
    for cls in classes:
        f.write(f'{cls}\n')
for cls in classes:
    cls_path = os.path.join(subCIFAR10_path, cls)
    for idx in range(100):
        image_array = np.load(os.path.join(cls_path, f'{idx}.npy'))
        image = Image.fromarray(image_array, 'RGB')
        image_filename = f'{idx:04d}.png'
        image.save(os.path.join(cls_path, image_filename))
