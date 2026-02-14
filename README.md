# Image_caption_project
This project implements image captioning using RNNs and attention mechanisms, visualizes image gradients through saliency maps, and applies neural style transfer with content and style losses. Built with PyTorch, Transformers, and OpenCV, the code is version-controlled via GitHub and includes a video walkthrough.
# 🧠 Deep Learning for Image Captioning, Visualization, and Style Transfer

This project demonstrates the application of deep learning techniques to image captioning, visual explanation, and artistic style transfer using PyTorch and Hugging Face Transformers. It includes implementations of RNN-based and attention-based image captioning, saliency map generation, and neural style transfer.

## 📌 Project Objectives

- Implement image captioning using Recurrent Neural Networks (RNNs)
- Explore attention mechanisms to enhance captioning performance
- Visualize image gradients using saliency maps and adversarial examples
- Apply artistic style transfer using content and style losses
- Maintain version control with GitHub and provide a video walkthrough

## 🛠️ Tools & Libraries

- **Development Environment**: VS Code, GitHub
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Image Captioning Models**: Hugging Face Transformers (BLIP, ViT-GPT2)
- **Visualization & Processing**: NumPy, Matplotlib, OpenCV, Pillow

## 📁 Project Structure
image_captioning_project/ ├── image_captioning.py           # RNN-based image captioning using BLIP ├── attention_captioning.py       # Attention-based captioning using ViT-GPT2 ├── image_gradients.py            # Saliency map visualization using ResNet ├── style_transfer.py             # Artistic style transfer using VGG19 ├── sample.jpg                    # Sample image for captioning and gradients ├── content.jpg                   # Content image for style transfer ├── style.jpg                     # Style image for style transfer

## 📦 Installation

Install the required Python libraries:

```bash
pip install torch torchvision numpy matplotlib pillow transformers opencv-python




## 📦 Installation

Install the required Python libraries:

```bash
pip install torch torchvision numpy matplotlib pillow transformers opencv-python
 Usage Instructions
Run each script independently:
 Usage Instructions
Run each script independently:
python image_captioning.py         # RNN-based captioning
python attention_captioning.py     # Attention-based captioning
python image_gradients.py          # Saliency map visualization
python style_transfer.py           # Artistic style transfer
Ensure the image files (sample.jpg, content.jpg, style.jpg) are placed in the project directory or update the paths in the scripts accordingly.
💡 Key Features
- BLIP Captioning: Generates captions using a pretrained RNN-based model.
- ViT-GPT2 Captioning: Leverages attention mechanisms for improved contextual understanding.
- Saliency Maps: Highlights important regions in an image using gradient-based visualization.
- Neural Style Transfer: Blends content and style from two images using VGG19 and custom loss functions.
🎥 Video Walkthrough
A recorded walkthrough is included, demonstrating:
- Code execution and outputs
- GitHub version control process
- Explanation of each module and its purpose
📚 References
- Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929. https://doi.org/10.48550/arXiv.2010.11929 (doi.org in Bing)
- Donahue, J., et al. (2015). Long-term recurrent convolutional networks for visual recognition and description. CVPR. https://doi.org/10.1109/CVPR.2015.7298878 (doi.org in Bing)
- Venugopalan, S., et al. (2015). Sequence to sequence – video to text. ICCV. https://doi.org/10.1109/ICCV.2015.515 (doi.org in Bing)


