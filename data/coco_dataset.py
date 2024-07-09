# File to create a dataset class specifically for the COCO dataset

# Imports
from torch.utils import data as data
import json
from pycocotools.coco import COCO
import nltk
import numpy as np
from tqdm import tqdm

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
    
    def __getitem__():
        """Generate samples from the data"""
        #TODO:
    
    def __len__():
        """The total number of samples"""
        #TODO: