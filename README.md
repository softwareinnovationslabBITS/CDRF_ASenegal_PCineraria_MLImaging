# Plant Species Identification and Health Diagnosis

This repository provides the official implementation, trained model and evaluation code for the research paper " Machine learning-based model for automated stress phenotyping of bipinnate leaf in underutilized desert plants Acacia senegal and Prosopis cineraria "


The framework implements two primary models:
1. **Species Classifier**: A deep convolutional neural network based on EfficientNetB0 that distinguishes *Acacia senegal* (AS) from *Prosopis cineraria* (PC).
2. **PC Health Classifier**: A multimodal architecture that extracts fine-tuned visual representations from EfficientNetB0 and fuses them with plant tabular data, followed by and Support Vector Machine (SVM) classification head.

---

## Quick Start

### 1. Installation

Clone this repository and install the dependencies:

```bash
# Clone repository
git clone <REPO_URL>
cd <REPO_NAME>

# Install dependencies (Python 3.10 or 3.11 recommended)
pip install -r requirements.txt
```

Alternatively, you can create an isolated Conda environment:

```bash
conda env create -f environment.yml
conda activate plant_classification
```

### 2. Run Test Evaluation

To evaluate both models on the unseen test splits and display all evaluation metrics, execute:

```bash
python main_script.py
```

To force CPU execution (disabling GPU acceleration):

```bash
python main_script.py --cpu
```

---

## Hardware Requirements & Adaptive Execution

The evaluation script is designed to run seamlessly on diverse computing environments:
* **GPU Execution (Auto-detected)**: If a compatible NVIDIA CUDA GPU is available, the script automatically enables memory growth (`TF_FORCE_GPU_ALLOW_GROWTH=true`) to optimize throughput.
* **CPU Execution (Adaptive Fallback)**: If no GPU is present (e.g., standard laptops, CPU cloud instances), the script automatically executes in CPU mode without errors or missing library warnings.
* **Execution Time**: The test evaluation evaluates 1,415 species images and 2,400 health samples, completing in **under 30 seconds on a standard multi-core CPU**.

---

## Benchmark Results

### Model 1: Species Classifier (*Acacia senegal* vs. *Prosopis cineraria*)
* **Backbone**: EfficientNetB0 (Pretrained on ImageNet + Fine-Tuned Classification Head)
* **Test Split**: 1,415 unseen samples (762 *A. senegal*, 653 *P. cineraria*)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.51%** |
| **ROC-AUC** | **0.9999** |
| **Precision** | **99.85%** |
| **Recall (Sensitivity)** | **99.08%** |
| **Specificity** | **99.87%** |
| **F1-Score** | **99.46%** |

**Confusion Matrix**:
```text
                  Predicted AS    Predicted PC
Actual AS (0):        761               1
Actual PC (1):          6             647
```

---

### Model 2: PC Health Classifier (Healthy vs. Unhealthy)
* **Backbone**: Fine-Tuned EfficientNetB0 + Plant Tabular Data + RBF-SVM Head
* **Test Split**: 2,400 unseen samples (1,200 Healthy, 1,200 Unhealthy)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **87.25%** |
| **ROC-AUC** | **0.9469** |
| **Precision** | **86.34%** |
| **Recall (Sensitivity)** | **88.50%** |
| **Specificity** | **86.00%** |
| **F1-Score** | **87.41%** |

**Confusion Matrix**:
```text
                       Predicted Healthy    Predicted Unhealthy
Actual Healthy (0):          1032                   168
Actual Unhealthy (1):         138                  1062
```

## PC Health SVM Model Download

The 353 MB SVM model is distributed separately as a GitHub Release asset because it exceeds GitHub's regular Git file-size limit.
- Release asset filename: `svm_model.pkl`
- Required local path: `pc_health_classifier/svm_model.pkl`
- SHA-256: `8890662f73e178d12a1b870890a1bc3103a7c1ba38ab2be38073b878f8ebb8b0`

After a Release containing the asset is published, download [`svm_model.pkl`](https://github.com/softwareinnovationslabBITS/CDRF_ASenegal_PCineraria_MLImaging/releases/latest/download/svm_model.pkl) and place it at the path above before running `python main_script.py` to evaluate the PC health model.

---

## Repository Structure

```text
├── .gitignore                          # Ignores large caches, logs, and temporary files
├── README.md                           # Documentation and instructions
├── environment.yml                     # Conda environment definition
├── requirements.txt                    # Pip package requirements
├── main_script.py                      # Single execution entrypoint for evaluation
│
├── as_pc_classifier/                   # Model 1: Species Identification
│   ├── best_weights.weights.h5         # Trained model weights (20.7 MB)
│   ├── config.py                       # Configuration parameters
│   ├── core.py                         # Model architecture & evaluation routines
│   ├── train.py                        # Training script with CLI arguments
│
├── pc_health_classifier/               # Model 2: PC Health Diagnosis
│   ├── finetuned_backbone_weights.h5   # Fine-tuned EfficientNetB0 backbone (31 MB)
│   ├── combined_scaler.pkl             # Feature normalization scaler (31 KB)
│   ├── svm_model.pkl                   # GitHub Release asset; not committed to Git
│   └── tab_cols/                       # Tabular data column mappings
│   │   └── columns.json
│
└── data/                               # Bundled test splits
    ├── as_pc_test_part1.npz            # Species test images (Part 1, 750 samples)
    ├── as_pc_test_part2.npz            # Species test images (Part 2, 665 samples)
    └── pc_health_test.npz              # PC health test samples (2,400 samples)
```

---

## Training

The AS-PC species classifier can be retrained with a dataset root directory:

```bash
python as_pc_classifier/train.py --data_dir /path/to/dataset
```

The PC health training pipeline is not included. `main_script.py` evaluates the packaged model artifacts on the bundled test data.

---

## License

This project is licensed under the GNU General Public License v3.0. See the repository's `LICENSE` file for the full text.
