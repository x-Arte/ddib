import os
import matplotlib.pyplot as plt
import blobfile as bf
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import math
from PIL import Image
from skimage.transform import resize

class LowDoseDataset(Dataset):
    def __init__(self,  image_path, image_size = 0, in_channels = 3):
        super().__init__()
        self.imgs, self.image_shape = self.read_images(
            self.get_image_paths(image_path), image_size = image_size)
        self.resize = resize
        self.n_samples = self.imgs.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.imgs[index], {}
    
    def shape(self):
        return self.image_shape
    
    @staticmethod
    def get_image_paths(directory_path):
        # Check if the path is a directory and list all files
        if os.path.isdir(directory_path):
            return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        else:
            raise ValueError(f"Provided path is not a directory: {directory_path}")

    @staticmethod
    def read_images(image_paths, image_size,  ratio=0):
        """Reads the image at the given path as grayscale. Returns (arr, shape):
        the image and their shape."""
        images = []
        for path in image_paths:
            arr, shape = LowDoseDataset.read_image(path, image_size = image_size)
            images.append(arr)
        imgs = np.array(images) 
        print(imgs.shape)
        return imgs, shape

    @staticmethod
    def normalize(arr):
        """Normalize the given array to [0, 1]."""
        return arr / 255.0

    @staticmethod
    def unnormalize(arr):
        """Unnormalize arr back to image range [0, 255]."""
        return arr * 255.0

    @staticmethod
    def read_image(image_path, image_size, ratio=0):
        """Reads the image at the given path as grayscale. Returns (arr, shape):
        points in the image and their shape."""
        data_type = image_path.split(".")[-1]
        if data_type == "png":
            with bf.BlobFile(image_path, "rb") as f:
                image = Image.open(f)
                image.load()
            image = image.convert("L")
            arr = np.array(image).astype(np.float32)
            arr = LowDoseDataset.normalize(arr)
        elif data_type == "npy":
            arr = np.load(image_path).astype(np.float32)
        if image_size:
            arr = resize(arr, (image_size, image_size), anti_aliasing=True)
            grey_arr = np.stack([arr]*3, axis=0)
            shape = grey_arr.shape

        #print(arr.shape)
        return grey_arr, shape

def load_lowdose_data(batch_size, image_path, logger, ratio=0, training = True, image_size = 256):
    """Loads the low dose data from the given directory."""
    dataset = LowDoseDataset(
        image_path=image_path,
        image_size = image_size
    )
    logger.log(f"dataset length: {len(dataset)}, {dataset.image_shape}")

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, drop_last=False)
    if training:
        while True:
            yield from loader
    else:
        yield from loader

def gener_image_2D(points, filename='image.png', shape = (360,360), logger = None, FORCED = False):
    if logger:
        logger.log(f"generating image {filename} original value: {points}")
    
    points = LowDoseDataset.unnormalize(points)
    if shape[0] * shape[1] == points.shape[0] * 2:
        img = points.reshape((shape))
        # print(img.max(), img.min())
    elif FORCED:
        source_shape = (int(math.sqrt(points.shape[0] * 2)), int(math.sqrt(points.shape[0] * 2))) 
        img = Image.fromarray(points.reshape(source_shape))
        print(source_shape, shape)
        img = img.resize(shape)     
    else:
        print(f"Failed to generate img: shape[0] * shape[1] = {shape[0] * shape[1]}, but total points number is: {points.shape[0] * 2}")
    plt.axis('off')
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.savefig(filename, bbox_inches='tight', transparent=False, pad_inches=0)
