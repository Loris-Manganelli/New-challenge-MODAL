import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

img = torch.randint(0, 256, size=(3, 20, 20), dtype=torch.uint8)

transforms = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.9, 0.9, 0.9], std=[0.229, 0.224, 0.225]),
])

img = transforms(img)
plt.imshow(img.permute(1, 2, 0))
plt.show()