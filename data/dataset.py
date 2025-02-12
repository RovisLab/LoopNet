from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.image_paths = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
        
        print(f"\nInitializing dataset from: {root_dir}")
        
        if not is_test:
            # Training/Validation mode
            self.classes = sorted([d for d in os.listdir(root_dir) 
                                if os.path.isdir(os.path.join(root_dir, d))])
            
            print(f"Found {len(self.classes)} classes: {self.classes}")
            
            for idx, class_name in enumerate(self.classes):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    images = [f for f in os.listdir(class_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    print(f"Class {class_name}: {len(images)} images")
                    
                    for img_name in images:
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)
        else:
            # Test mode - just load all images from directory
            self.classes = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 
                          'class_5', 'class_6', 'class_7', 'class_8', 'class_9']  # Hardcoded class names
            images = [f for f in os.listdir(root_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(images)} test images")
            
            for img_name in images:
                self.image_paths.append(os.path.join(root_dir, img_name))
                self.labels.append(-1)  # Use -1 to indicate no label
                
        print(f"Total images loaded: {len(self.image_paths)}\n")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {img_path}: {str(e)}")
                dummy = Image.new('RGB', (224, 224))
                image = self.transform(dummy)
        
        return image, self.labels[idx], img_path