# File to contain any data pre-processing utility defs / functions

# Imports
from torchvision import transforms

def getTransform():
    """return the image pre-processing transform"""
    transform = transforms.Compose(
        [
            transforms.Resize(256),             # resize image on shorlest edge so all are same size
            transforms.RandomCrop(224),         # perform a random crop
            transforms.RandomHorizontalFlip(),  # randomly flip image 0.5% probability
            transforms.ToTensor(),              # create the tensor
            transforms.Normalise(               # normalise as per the required pre-trained model
                (0.485, 0.456, 0.406),
                (0.299, 0.244, 0.225),
            ),
        ]
    )

    # return the transform
    return transform