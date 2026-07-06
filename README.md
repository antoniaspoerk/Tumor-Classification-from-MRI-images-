# Brain Tumor Classification from MRI Images

Deep learning pipeline for classifying brain MRI scans into four categories
(glioma, meningioma, pituitary tumor, no tumor) using PyTorch. Compares four
modeling approaches of increasing complexity: a simple MLP, a CNN trained
from scratch, a frozen pretrained VGG16, and a fine-tuned VGG16.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(Kaggle), split into 4 balanced classes:

| Split | Total | Per class |
|---|---|---|
| Train | 4,480 | 1,120 |
| Validation | 1,120 | 280 |
| Test | 1,600 | 400 |

Classes: glioma, meningioma, pituitary, no tumor.

## Approach

- Images resized to 96x96 and normalized.
- Training-set augmentation: random horizontal flip, slight rotation (±15°),
  and Gaussian blur. Vertical flips are deliberately excluded, since
  flipping a brain scan top-to-bottom produces an anatomically implausible
  image.
- Four models, in order of increasing complexity:
  1. **MLP** — fully connected baseline
  2. **CNN** — 5 conv layers trained from scratch
  3. **VGG16 (frozen)** — ImageNet-pretrained, only the final classifier
     layer is trainable
  4. **VGG16 (fine-tuned)** — last two convolutional blocks and the
     classifier head unfrozen, rest of the network stays frozen
- All models trained with Adam (lr=1e-4), up to 20 epochs with early
  stopping on validation loss.

## Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| MLP | 77.5% | 0.77 |
| CNN (from scratch) | 87.7% | 0.87 |
| VGG16 (frozen) | 73.8% | 0.72 |
| **VGG16 (fine-tuned)** | **89.3%** | **0.89** |

Fine-tuning VGG16 gave the best result. The frozen VGG16 was the
*worst* performer of the four, even behind the from-scratch CNN, which
suggests ImageNet features transfer poorly to grayscale MRI scans unless at
least some convolutional layers are allowed to adapt. Unfreezing the last
two blocks was enough to make the pretrained model the strongest model
overall.

### Which classes get confused

Across every model, **pituitary and meningioma were consistently the
hardest classes to separate**: pituitary had the lowest recall of any class
in every model (as low as 0.47 for the MLP, still only 0.73 even for the
best model), while meningioma showed correspondingly reduced precision —
indicating many pituitary scans were being misclassified as meningioma.
Glioma and no-tumor, by contrast, were classified reliably (recall ≥0.9)
across all four models.

## How to Run Locally

1. Clone the repository and install dependencies
``` bash
git clone https://github.com/antoniaspoerk/Tumor-Classification-from-MRI-images.git
cd Tumor-Classification-from-MRI-images
pip install torch torchvision matplotlib seaborn numpy scikit-learn jupyter
```
2. Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
from Kaggle and split it into `train`/`val`/`test` folders, e.g. under `./data`.
3. The notebook was originally built for Google Colab (mounts Google
Drive). Replace that cell with a local path instead:
```python
DATA_PATH = "./data"
```
4. Launch the notebook:
``` bash
jupyter notebook Tumor_Classification.ipynb
```
