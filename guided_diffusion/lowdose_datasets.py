import os
import matplotlib.pyplot as plt
import blobfile as bf
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class LowDoseDataset(Dataset):
    def __init__(self,  image_path, resize=False, ratio=0):
        super().__init__()
        self.points, self.image_shape = self.read_images(
            self.get_image_paths(image_path), resize=resize, ratio=ratio)
        self.resize = resize
        self.n_samples = self.points.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.points[index], {}

    @staticmethod
    def get_image_paths(directory_path):
        # Check if the path is a directory and list all files
        if os.path.isdir(directory_path):
            return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        else:
            raise ValueError(f"Provided path is not a directory: {directory_path}")

    @staticmethod
    def read_images(image_paths, resize, ratio=0):
        """Reads the image at the given path as grayscale. Returns (arr, shape):
        points in the image and their shape."""
        images = []
        for path in image_paths:
            arr, shape = LowDoseDataset.read_image(path, resize=resize, ratio=ratio)
            images.append(arr)
        points = np.array(images) 
        points = np.array(images).reshape(-1, 2)
        print(points.shape)
        return points, shape

    @staticmethod
    def normalize(arr):
        """Normalize the given array to [0, 1]."""
        return arr / 255.0

    @staticmethod
    def unnormalize(arr):
        """Unnormalize arr back to image range [0, 255]."""
        return arr * 255.0

    @staticmethod
    def read_image(image_path, resize=False, ratio=0):
        """Reads the image at the given path as grayscale. Returns (arr, shape):
        points in the image and their shape."""
        with bf.BlobFile(image_path, "rb") as f:
            image = Image.open(f)
            image.load()
        image = image.convert("L")
        arr = np.array(image).astype(np.float32)
        if resize:
            arr = arr[::ratio, ::ratio]
        shape = arr.shape
        arr = arr.reshape(-1, 2)  # Reshape to (N, 2) for a single channel
        arr = LowDoseDataset.normalize(arr)
        #print(arr.shape)
        return arr, shape

def load_lowdose_data(batch_size, image_path, logger, ratio=0, training = True):
    """Loads the low dose data from the given directory."""
    dataset = LowDoseDataset(
        image_path=image_path,
        ratio=ratio
    )
    logger.log(f"dataset length: {len(dataset)}, {dataset.image_shape}")

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, drop_last=False)
    if training:
        while True:
            yield from loader
    else:
        yield from loader

def gener_image_2D(points, filename='image.png', shape = (360,360), logger = None):
    if shape[0] * shape[1] == points.shape[0] * 2:
        if logger:
            logger.log(f"generating image {filename} original value: {points}")
        img = points.reshape(shape)
        img = LowDoseDataset.unnormalize(img)
        # print(img.max(), img.min())
    plt.axis('off')
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.savefig(filename, bbox_inches='tight', transparent=False, pad_inches=0)
