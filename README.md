# Text Denoiser
## Introduction  
This denoiser extracts only text from the image by filtering the irrelevant background and noises, using Convolutional Neural Network to extract deep features in an image which includes loads of texts and remove non-text pixels.
## Limitation
This network only accept the input size of 128 X 128. The image over this size might need to be split and then merge. However, this solution makes the final image have some pixel distortion. More reliable solution is still being exploring.   
Colourful text images are not yet tested, currently only grayscale texts are ensured to be valid. Colourful input might produce unwanted result.  
## Dependency  
Pytorch 2.5
## Network Structure

encoder:  
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
  (1): ReLU()  
  (2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
  (3): ReLU()  
  (4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
  (5): ReLU()  
  (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
  
middle:  
  (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
  (1): ReLU()  
  (2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
  (3): ReLU()  
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
    
decoder:  
  (0): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))  
  (1): ReLU()  
  (2): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))  
  (3): ReLU()  
  (4): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
  (5): Sigmoid()    
