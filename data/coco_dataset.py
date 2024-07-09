# File to create a dataset class specifically for the COCO dataset

# Imports
from torch.utils import data as data
import json
from pycocotools.coco import COCO
import nltk
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import torch

class CocoDataset(data.Dataset):
    """Efficient data generation utility to allow the COCO dataset to be read in along with captions"""
    def __init__(
            self,
            transform,          # image pre-proccessing transform function
            mode,               # dataset mode may be either test or train
            batchSize,          # size of training batch to select
            imagesFolder,       # path to images folder
            annotationsFile,    # path to coco annotations json
            vocab,              # instance of dataset vocab class (dependancy injection)
    ):
        """Initialise the dataset class"""
        self.transform = transform
        self.mode = mode
        self.batchSize = batchSize
        self.imagesFolder = imagesFolder
        self.annotationsFile = annotationsFile   
        self.vocab = vocab
        
        # training specific initialisation
        if self.mode == "train":
            self.coco = COCO(annotationsFile)       # coco api initialisation for annotations file
            self.ids = list(self.coco.anns.keys())  # get image ids

            print("Identifying Caption Lengths...")
            # tokenize each caption
            captionsTokenized = [nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.ids[index]]["caption"]).lower()
            )
            for index in tqdm(np.arrange(len(self.ids)))
            ]

            # get caption lengths
            self.captionLengths = [len(tk) for tk in captionsTokenized]
        
        # test specific initialisation
        else:
            testInformation = json.loads(open(annotationsFile).read())
            self.paths = [testItm["file_name"] for testItm in testInformation["images"]]
    
    def __getitem__(self, index):
        """Generate samples from the data"""
        # training specific image and caption return
        if self.mode == "train":
            annotationId = self.ids[index]
            caption = self.coco.anns[annotationId]["caption"]
            imageId = self.coco.anns[annotationId]["image_id"]
            path = self.coco.loadImgs(imageId)[0]["file_name"]

            # convert the image to a tensor and perform image pre-processing
            image = Image.open(os.path.join(self.imagesFolder, path)).convert("RGB")
            image = self.transform(image)

            # convert caption to tensor
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = [self.vocab(self.vocab.start_word)]
            caption.extend([self.vocab(tk) for tk in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption
        
        # test specific image return
        else:
            image = Image.open(os.path.join(self.imagesFolder, path)).conver("RGB")
            originalImage = np.array(image)
            image = self.transform(image)

            # return pre-processed image and original image
            return originalImage, image

    def __len__():
        """The total number of samples"""
        #TODO: