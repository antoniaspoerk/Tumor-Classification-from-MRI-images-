# Brain Tumor Classification from MRI Images

This project applies deep learning to classify brain MRI scans into four categories (glioma, meningioma, pituitary tumor, and no tumor) using PyTorch. It compares four modeling approaches of increasing complexity: a simple multilayer perceptron (MLP), a convolutional neural network (CNN) trained from scratch, a frozen pretrained VGG16, and a fine-tuned VGG16. The notebook covers the full pipeline, including data exploration, augmentation strategies tailored to MRI images, model training, and evaluation using accuracy, F1 score, precision, and recall, with a particular focus on which tumor types are most often confused and why.

## How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/antoniaspoerk/Tumor-Classification-from-MRI-images-/tree/main.git
   cd Tumor-Classification-from-MRI-images
2. **Install the dependencies**
   ```bash
   pip install torch torchvision matplotlib seaborn numpy scikit-learn jupyter

3. **Get the dataset**
   Download the MRI dataset (train/val/test folders, one subfolder per class) and place it somewhere on your machine, e.g. ./data.

4. **Update the data path**
   The notebook was originally built for Google Colab, so it mounts Google Drive. Replace that cell with a local path instead:
   ```bash
   DATA_PATH = "./data"

5. **Launch the notebook**
   Tumor_Classification.ipynb
