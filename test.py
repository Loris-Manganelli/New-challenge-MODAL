import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision.transforms import v2

img = Image.open("dataset/val/PARMESAN/000004.jpg")

transfo1 = transforms.Compose([
              transforms.ToTensor(),
              transforms.RandomResizedCrop(size=224, scale=[0.1, 0.4])
            ])

transfo2 = transforms.v2.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(224,224))
])

img1 = transfo1(img)
img2 = transfo2(img)


plt.imshow(img1.permute(1,2,0))
plt.show()


plt.imshow(img2.permute(1,2,0))
plt.show()