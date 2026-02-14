import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load image and preprocess
image_path = "sample.jpg"
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = transform(image).unsqueeze(0)
image.requires_grad_()

# Load model
model = models.resnet18(pretrained=True)
model.eval()

# Forward pass
output = model(image)
class_idx = torch.argmax(output)
output[0, class_idx].backward()

# Generate saliency map
saliency = image.grad.data.abs().squeeze().permute(1, 2, 0)
plt.imshow(saliency.numpy(), cmap="hot")
plt.title("Saliency Map")
plt.axis("off")
plt.show()