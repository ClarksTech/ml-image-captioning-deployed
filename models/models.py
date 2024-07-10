# File containing the ML model classes (CNN encoder and RNN decoder)

# Imports
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """Class to build the pre-trained resNet50 model"""
    def __init__(self, embedSize):
        super(EncoderCNN, self).__init__()          # inherit from super
        resnet = models.resnet50(pretrained=True)   # trained model

        # make sure parameters can not be changed
        for param in resnet.parameters():
            param.requires_grad_(False)

        # remove the last fully connected layer of the model as we are encoding not classifying
        modules = list(resnet.children())[:-1]      # Exclude the last layer (fully connected layer)
        self.resnet = nn.Sequential(*modules)       # Create a new sequential model without the last layer

        # Add a new fully connected layer to map the ResNet50 features to the desired embedding size
        self.embed = nn.Linear(resnet.fc.in_features, embedSize)    # Create a linear layer with input size from ResNet and output size of embedSize

    def forward(self, images):
        """Forward pass through the EncoderCNN, returning tensor of mapped image features"""
        features = self.resnet(images)                  # Pass the images through the ResNet model to extract features
        features = features.view(features.size(0), -1)  # Flatten the features to a 2D tensor (batch_size, num_features)
        features = self.embed(features)                 # Map the features to the embedding size using the fully connected layer
        return features

