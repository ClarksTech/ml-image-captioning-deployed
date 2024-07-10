# File to create a vocabulary class specifically for the COCO dataset

# Imports
import os
import pickle
from pycocotools.coco import COCO
from collections import Counter
import nltk
nltk.download("punkt")

class Vocabulary(object):
    """Class to generate and store COCO dataset Vocabulary"""
    def __init__(
            self,
            vocabThreshold,                                             # minimum number of occurances in dataset to be added to vocabulary
            vocabFile = "vocabulary.pkl",                               # saved vocabulary
            startWord = "<start>",                                      # magic start word
            endWord = "<end>",                                          # magic end word
            unkWord = "<unk>",                                          # unknown word not in the dataset
            annotationFile = "data/annotations/captions_train2017.json",# COCO dataset captions file
            vocabFileExists = False,                                    # flag for if vocab file already exists
    ):
        self.vocabThreshold = vocabThreshold
        self.vocabFile = vocabFile
        self.startWord = startWord
        self.endWord = endWord
        self.unkWord = unkWord
        self.annotationFile = annotationFile
        self.vocabFileExists = vocabFileExists
        self.getVocab()

    def getVocab(self):
        """Identify and store COCO dataset vocab or load from file if exists"""
        # load existing vocab library
        if os.path.exists(self.vocabFile) and self.vocabFileExists:
            with open(self.vocabFile) as file:
                vocab = pickle.load(file)       # load from existing saved file
            self.word2idx = vocab.word2idx      # dictionary mapping word to integer 
            self.idx2word = vocab.idx2word      # dictionary mapping integer to word
            print("Successfully loaded COCO dataset vocabulary!")
        
        else:
            # make the vocab library and save
            self.buildVocab()
            with open(self.vocabFile, "wb") as file:
                pickle.dump(self, file)
    
    def buildVocab(self):
        """Populate dictionaries for converting word (token) to integer and the reverse"""
        self.initVocab()
        self.addWord(self.startWord)
        self.addWord(self.endWord)
        self.addWord(self.unkWord)
        self.addCaptions()

    def initVocab(self):
        """Initialise the dicts for converting between word (tokens) and integers"""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def addWord(self, word):
        """Add word (token) to the vocab"""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx  # add with current index (integer)
            self.idx2word[self.idx] = word  # add reverse mapping
            self.idx +=1                    # increment for next new word

    def addCaptions(self):
        """Add all captions to the vocabulary that exceed the minimum occurance threshold"""
        coco = COCO(self.annotationFile)
        counter = Counter()     # counter for word occurance count
        ids = coco.anns.keys()
        for i, idx in enumerate(ids):
            caption = str(coco.anns[idx]["caption"])                # get caption for current id
            tokens = nltk.tokenize.word_tokenize(caption.lower())   # tokenise the caption
            counter.update(tokens)                                  # update counter with tokens from current caption

            # print progress for every 100,000 captions processed
            if i % 100000 == 0:
                print("[%d/%d] Tokenizing COCO dataset captions..." %(i, len(ids)))

        # only keep words (tokens) above occurance threshold
        words = [word for word, cnt in counter.items() if cnt >= self.vocabThreshold]

        # add each word meeting occurance threshold to the vocab
        for i, word in enumerate(words):
            self.addWord(word)

    def __call__(self, word):
        """Retrieve the index of a word. If the word is not in the vocabulary, return the index for the unknown word token"""
        if word not in self.word2idx:
            return self.word2idx[self.unkWord]
        return self.word2idx[word]
    
    def __len__(self):
        """Return the number of words in the vocabulary"""
        return len(self.word2idx)
    
            
