import torch
from torch.utils.data import Dataset
import json
import PIL

# load the kvasirseg dataset to '/contents/Kvasir-SEG' path
def getKvasir():
    root = "Kvasir-SEG"
    if os.path.exists('/content/Kvasir-SEG'):
      print("Kvasir-Seg already downloaded at path /content/Kvasir-SEG")
    else:
      !wget https://datasets.simula.no/downloads/kvasir-seg.zip
      !unzip kvasir-seg
      print("Kvasir-Seg downloaded successfully at '/content/Kvasir-SEG")
    return root

# dataloader for the kvasir seg dataset

class LoadKvasirSeg(Dataset):
  def __init__(self, transform = None):
      super().__init__()
      self.root = getKvasir()
      # path to images
      self.images = os.path.join(self.root,"images")
      # getting the respective ids of the images
      # we have to remove the extension (jpg) from the file names to make ids
      self.file_names = sorted(os.listdir(self.images))
      self.idx = []
      for i in self.file_names:
        self.idx.append(i.split('.')[0])
      self.transform = transform
      # the bounding boxes are stored as a dictionary with idx as keys
      self.bboxes = json.load(open(os.path.join(self.root, "kavsir_bboxes.json")))



  def __len__(self):
    return len(self.idx)

  def __getitem__(self, index):
    # This function will provide the image and its label
    # The label is a vector containing id, bbox coordinates (x_min, x_max, y_min, y_max), class
    img_path = os.path.join(self.root,'images',self.file_names[index])
    image = PIL.Image.open(img_path)
    labels_from_json = self.bboxes.get(self.idx[index]).get("bbox")
    label_vector = []
    for i in labels_from_json:
      x_min = i.get('xmin')
      y_min = i.get('ymin')
      x_max = i.get('xmax')
      y_max = i.get('ymax')
      label_vector.append(0)
      label_vector.append(x_min)
      label_vector.append(y_min)
      label_vector.append(x_max)
      label_vector.append(y_max)
    

    if self.transform:
      image = self.transform(image)

    return image, label_vector