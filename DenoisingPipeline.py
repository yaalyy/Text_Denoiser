import numpy as np   
from PIL import Image  
import torch  


class DenoisingPipeline:
    def __init__(self, model, block_size=128, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.model.eval()  
        self.block_size = block_size
        self.device = device

    def process_image(self, img):
        """
        param img: input PIL.Image 
        return: denoised PIL.Image 
        """
        img = img.convert("RGB")
        img_width, img_height = img.size
        block_size = self.block_size
        
        
        padded_width = ((img_width + block_size - 1) // block_size) * block_size
        padded_height = ((img_height + block_size - 1) // block_size) * block_size
        padded_img = Image.new("RGB", (padded_width, padded_height))
        padded_img.paste(img, (0, 0))

        
        padded_img_np = np.array(padded_img) / 255.0


        output_img_np = np.zeros_like(padded_img_np, dtype=np.uint8)

        
        for i in range(0, padded_height, block_size):
            for j in range(0, padded_width, block_size):
                block = padded_img_np[i:i+block_size, j:j+block_size]
                
                # Fill in the block, if block size is not enough
                if block.shape[0] != block_size or block.shape[1] != block_size:
                    temp_block = np.zeros((block_size, block_size, 3), dtype=np.float32)
                    temp_block[:block.shape[0], :block.shape[1]] = block
                    block = temp_block

                
                block_tensor = torch.from_numpy(block).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    denoised_block_tensor = self.model(block_tensor).squeeze(0).cpu()

                # convert back to np array
                denoised_block = denoised_block_tensor.permute(1, 2, 0).numpy() * 255.0
                denoised_block = np.clip(denoised_block, 0, 255).astype(np.uint8)
   
                
                output_img_np[i:i+block_size, j:j+block_size] = denoised_block
       
        # remove padding
        output_img_np = output_img_np[:img_height, :img_width]
        output_img = Image.fromarray(output_img_np)

        return output_img
