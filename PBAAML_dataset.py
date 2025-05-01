import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class Dataset(Dataset):
    def __init__(self,image_dir,labels_file,transform=None):
        super().__init__()

        self.image_dir = image_dir
        self.transform = transform
        self.labels_file = labels_file

        df = pd.read_excel(self.labels_file)

        self.labels = df.values.flatten()

        self.images_name = sorted(os.listdir(self.image_dir))

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images_name[index]

        img_path = os.path.join(self.image_dir,img_name)

        label = self.labels[index]

        image = Image.open(img_path).convert("L")

        if self.transform is not None:
            
            image = self.transform(image)

        return image,label
    
# dataset = Dataset("./subset","./sub_label.xlsx")

# img,label = dataset[1]

# print("label:",label)