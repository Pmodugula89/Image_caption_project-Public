import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess images
def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

content_image = load_image("content.jpg")
style_image = load_image("style.jpg")
image = content_image.clone().requires_grad_(True)

# Load VGG model
vgg = models.vgg19(pretrained=True).features.eval()

# Loss functions
def content_loss(target, content):
    return torch.mean((target - content) ** 2)

def gram_matrix(feature_map):
    _, C, H, W = feature_map.size()
    features = feature_map.view(C, H * W)
    return torch.mm(features, features.t()) / (C * H * W)

def style_loss(target, style):
    return torch.mean((gram_matrix(target) - gram_matrix(style)) ** 2)

# Optimization
optimizer = optim.Adam([image], lr=0.01)
for i in range(500):
    target_content = vgg(content_image)
    target_style = vgg(style_image)
    loss = content_loss(target_content, content_image) + 1e6 * style_loss(target_style, style_image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Display result
plt.imshow(image.squeeze().permute(1, 2, 0).detach().numpy())
plt.title("Stylized Image")
plt.axis("off")
plt.show()