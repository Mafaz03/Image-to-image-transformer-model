
import os
from torch.utils.data import Dataset
from PIL import Image
from model import TransformerConfig
from torchvision import transforms

transform_ds = transforms.Compose([
    transforms.Resize((
                        TransformerConfig().image_size,
                        TransformerConfig().image_size*2
                        )), # Resize the input image

    transforms.ToTensor(), # Convert to torch tensor (scales data into [0,1])
    transforms.Lambda(lambda t: (t * 2) - 1), # Scale data between [-1, 1] 
])

class CarDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
        
        if self.transform:
            image = self.transform(image)  # Convert to Tensor
            width_half = image.shape[-1] // 2  # Get half width
            return image[:, :, :width_half], image[:, :, width_half:]  # Left and right split
        
        else:
            # For PIL Image (if no transform applied)
            width, height = image.size
            left_half = image.crop((0, 0, width // 2, height))  # Left side
            right_half = image.crop((width // 2, 0, width, height))  # Right side
            return left_half, right_half