from DenoisingPipeline import DenoisingPipeline
import torch 
from DenoisingCNN import DenoisingCNN
from PIL import Image


'''Only execute when trained model exists'''
if __name__ == '__main__':
    # Load existed model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # try to use cuda acceleration
    state_dict = torch.load("models/denoiser.pth")
    denoiser = DenoisingCNN()
    denoiser = state_dict["denoiser"].to(device)


    pipeline = DenoisingPipeline(denoiser)
    input_img = Image.open(".\\data\\input_noisy_images\\134.png")
    denoised_image = pipeline.process_image(input_img)
    denoised_image.save(".\\path_to_output_image.jpg") 
