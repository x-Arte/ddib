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
    def __init__(self,  image_path, image_size = 0, in_channels = 3, num_25D = 3):
        super().__init__()
        self.imgs, self.image_shape = self.read_images(
            self.get_image_paths(image_path), image_size = image_size, num_25D = num_25D)
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
            return sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        else:
            raise ValueError(f"Provided path is not a directory: {directory_path}")

    @staticmethod
    def read_images(image_paths, image_size,  ratio = 0, num_25D = 1):
        """Reads the image at the given path as grayscale. Returns (arr, shape):
        the image and their shape."""
        original_images = {"arr":[], "name":[], "index":[]}
        images = []
        for path in image_paths:
            # arr, shape = LowDoseDataset.read_image(path, image_size = image_size)
            arr = LowDoseDataset.read_image(path, image_size = image_size)
            index = path.split("/")[-1].split("_")[-1].split(".")[0]
            name = path.split("_"+index+".")[0].split("/")[-1]
            original_images["arr"].append(arr)
            original_images["index"].append(index)
            original_images["name"].append(name)
            
            # print(path, index, name)
        images, shape = LowDoseDataset.stack_channels(original_images, num_25D)     
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
    def stack_channels(original_images, num_25D):
        images = []
        for i in range (len(original_images)):
            if num_25D == 1:
                grey_arr = np.stack([original_images["arr"][i]]*3, axis=0)
                shape = grey_arr.shape
                images.append(grey_arr)
            elif num_25D == 3:
                print("Stacked!")
                if len(images) == 0 or original_images["index"][i] == "000":
                    arr = np.stack([original_images["arr"][0], original_images["arr"][0], original_images["arr"][1]], axis=0)
                    shape = arr.shape
                elif i == len(original_images["arr"]) - 1 or original_images["name"][i]!=original_images["name"][i-1]:
                    arr = np.stack([original_images["arr"][i-1], original_images["arr"][i], original_images["arr"][i]], axis=0)
                else:
                    arr = np.stack([original_images["arr"][i-1], original_images["arr"][i], original_images["arr"][i+1]], axis=0)
                images.append(arr)
 
        return images, shape

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
            # grey_arr = np.stack([arr]*3, axis=0)
            # shape = grey_arr.shape

        #print(arr.shape)
        #return grey_arr, shape
        return arr

def load_lowdose_data(batch_size, image_path, logger, ratio=0, training = True, image_size = 256, in_channels = 3, num_25D = 3):
    """Loads the low dose data from the given directory."""
    dataset = LowDoseDataset(
        image_path=image_path,
        image_size = image_size,
        in_channels = in_channels,
        num_25D = num_25D
    )
    logger.log(f"dataset length: {len(dataset)}, {dataset.image_shape}")

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, drop_last=False)
    if training:
        while True:
            yield from loader
    else:
        yield from loader

def gener_image_2D(arr, filename='image.png', shape = (64,64), logger = None, FORCED = False):
    if logger:
        logger.log(f"generating image {filename}") # original value: {arr}")
    
    # points = LowDoseDataset.unnormalize(points)
    # if shape[0] * shape[1] == points.shape[0] * 2:
    #     img = points.reshape((shape))
    #     # print(img.max(), img.min())
    # elif FORCED:
    #     source_shape = (int(math.sqrt(points.shape[0] * 2)), int(math.sqrt(points.shape[0] * 2))) 
    #     img = Image.fromarray(points.reshape(source_shape))
    #     print(source_shape, shape)
    #     img = img.resize(shape)     
    # else:
    #     print(f"Failed to generate img: shape[0] * shape[1] = {shape[0] * shape[1]}, but total points number is: {points.shape[0] * 2}")
        # points = LowDoseDataset.unnormalize(points)
    arr = np.mean(arr, axis=1)
    img = arr[0]
    # # print(arr.shape)
    # if shape[0] * shape[1] == arr.shape[1] * arr.shape[2]:
    #     img = arr.reshape((shape))
    #     # print(img.max(), img.min())
    # elif FORCED:
    #     source_shape = (arr.shape[1], arr.shape[2]) 
    #     img = Image.fromarray(arr.reshape(source_shape))
    #     #print(source_shape, shape)
    #     img = img.resize(shape) 
    #print(np.mean(img))
    
    # img = img / np.max(img) * 255
    plt.axis('off')
    plt.imshow(img, cmap="gray")
    # plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.savefig(filename, bbox_inches='tight', transparent=False, pad_inches=0)
