# Class to load the COCO dataset

# Imports
import torch.utils.data as data
import os

class CocoDataLoader:
    """Class to create data loader for the COCO dataset"""
    def __init__(
        self,
        mode = "train",                 # dataset mode may be either test or train
        vocabFile = "../data/vocabulary.pkl",   # saved vocabulary
        vocabFileExists = True,         # flag for if vocab file already exists
        numWorkers = 0,                 # number of subprocesses to use for data loading
    ):
        assert mode in ["train", "test"], "mode must be one of 'train' or 'test'."
        
        if not vocabFileExists:
            assert mode == "train", "To generate vocab from captions file, must be in training mode (mode='train')."

        self.mode = mode
        self.vocabFile = vocabFile
        self.vocabFileExists = vocabFileExists
        self.numWorkers = numWorkers

        # Based on mode (train, test), obtain img_folder and annotations_file.
        if mode == "train":
            if vocabFileExists:
                assert os.path.exists(self.vocabFile), "vocab_file does not exist. Change vocabFileExists to False to create vocabFile."
            imagesFolder = "../data/images/train2017"
            annotationsFile = "../data/annotations/captions_train2017.json"
        elif mode == "test":
            assert os.path.exists(self.vocabFile), "Must first generate vocab.pkl from training data."
            assert self.vocabFileExists, "Change vocabFileExists to True."
            imagesFolder = "../data/images/test2017"
            annotationsFile = "../data/annotations/image_info_test2017.json"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.imagesFolder = imagesFolder
        self.annotationsFile = annotationsFile

    def getLoader (
            self,
            dataset,    # dependancy injection of the dataset class instance
            ):
        """Returns the data loader"""

        # set parameters for training dataloader
        if self.mode == "train":
            # randomly sample a catpion length and return indicies
            indicies = dataset.getTrainIndicies()
            # create and assign a batch sampler to retrieve a batch sample
            initialSample = data.sampler.SubsetRandomSampler(indicies=indicies)
            # init the dataloader
            dataLoader = data.DataLoader(
                dataset= dataset,
                num_workers= self.numWorkers,
                batch_sampler= data.sampler.BatchSampler(
                    sampler= initialSample,
                    batch_size= dataset.batchSize,
                    drop_last= False,
                ),
            )
        # set parameters for test dataloader
        else:
            dataLoader = data.DataLoader(
                dataset= dataset,
                batch_size= dataset.batchSize,
                shuffle= True,
                num_workers= self.numWorkers
            )

        return dataLoader
