# DeepNup: Nucleosome Position Prediction with Sequence and Shape Features

## Project Overview

This repo is for nucleosome positioning prediction using deep learning, comparing sequence-only and sequence+shape feature approaches. I used the original DeepNup model and an enhanced variant that uses DNA shape information extracted via DNAShapeR as an auxillary input.

## Main Contributions

### 1. **Sequence-Only Model** (Original DeepNup)
- Deep learning model trained exclusively on DNA sequence information
- Baseline performance for nucleosome classification
- Dataset: *Drosophila melanogaster* 147bp sequences

### 2. **Sequence + Shape Model** (DeepNup with DNA Shape)
- Enhanced architecture incorporating 5 DNA shape features
- Dual-branch neural network architecture
- Shape features extracted using DNAShapeR from FASTA sequences
- Comparative analysis of performance gains

## Experimental Results

### Performance Comparison

| Model | Dataset | AUC | Sensitivity | Specificity | MCC | F1 Score |
|-------|---------|-----|-------------|-------------|-----|----------|
| **DeepNup (Original)** | Sequence Only | 0.4893 | 0.4882 | 0.5029 | -0.0065 | 0.6209 |
| **DeepNup + Shapes** | Sequence + Shapes | 0.5092 | 0.4990 | 0.5077 | +0.0061 | 0.3808 |
| **Improvement** | - | **+1.99%** | **+1.07%** | **+0.48%** | **+0.0126** | **-23.41%** |

3-fold cross validation was used with only 20 epochs per fold. This model evalutaion was used on both approaches.

### Key Findings

✅ **Advantages of Shape Features**:
- AUC improved by ~2% (0.4893 → 0.5092)
- Better MCC (Matthews Correlation Coefficient) reflecting improved class balance
- Modest but consistent improvements in sensitivity/specificity trade-off

⚠️ **Trade-offs**:
- F1 Score decreased (0.6209 → 0.3808), indicating precision-recall shift
- Performance still near baseline (AUC ≈ 0.5) suggests dataset or modeling challenges
- Shape features provide complementary signal but not transformative

## Architecture Comparison

### Original DeepNup
```
Input: DNA Sequence (147bp)
  ↓
One-Hot Encoding (147, 4) + PseTNC (64, 1)
  ↓
Conv1D (kernel=5) → ReLU → BatchNorm → Dropout
  ↓
GRU (50 units)
  ↓
MaxPooling → Flatten → Dense(256) → Dense(1)
  ↓
Output: Binary Classification (Nucleosomal/Linker)
```

### DeepNup with Dual Inputs (Sequence + Shape)
```
Input 1: Sequence (147bp)           Input 2: DNA Shapes (147bp × 5)
  ↓                                        ↓
One-Hot (147,4) + PseTNC (64,1)     Conv1D Multi-Scale
  ↓                                        ↓
Conv1D Multi-Branch                  Dropout + Flatten
(kernels: 3, 5, 7)                         ↓
  ↓                                   Shape Features
Concat + BatchNorm + Dropout              ↓
  ↓                                   (merged)
GRU (50 units)                             ↓
  ↓                                   Dense Layers
MaxPooling → Flatten → Dense(256)         ↓
  ↓________________________________Shape Path
         ↓
    Concatenate (307K params)
         ↓
   Dense(256) → Dropout
         ↓
   Dense(1) → Sigmoid
         ↓
   Output: Binary Prediction
```

**Architecture Enhancements**:
- **Parallel branches**: Separate processing streams for sequence and shape
- **Multi-scale convolutions**: Kernels 3, 5, 7 capture features at different resolutions
- **Late fusion**: Features combined after independent transformation
- **Increased capacity**: From 200K to 307K trainable parameters

### MLSNet-Inspired Variant (Implemented but Not Fully Tested)
```
Advanced Components:
├── StokenAttention: Efficient attention (O(n) vs O(n²))
├── Multi-branch Conv1D: 3 parallel kernels per path
├── LSTM processing: Instead of GRU (50 units)
├── Bi-LSTM for shapes: Forward+backward (32 units each)
└── Late concatenation: After full processing

Total Parameters: ~919K (3× capacity of dual-input model)
```

## Data & Methodology

### Dataset
- **Source**: *Drosophila melanogaster* genome
- **Total Sequences**: 5,750 (2,900 nucleosomal, 2,850 linker)
- **Sequence Length**: 147 bp (standard nucleosome core particle)
- **Class Split**: ~50/50 balanced

### Feature Extraction
1. **Sequence Features**:
   - One-Hot Encoding: 4-dimensional binary vector per nucleotide
   - PseTNC (Pseudo Trinucleotide Composition): 64-dimensional physicochemical representation

2. **Shape Features** (DNAShapeR):
   - EP (Electron Density at Phosphate)
   - HelT (Helix Twist)
   - MGW (Minor Groove Width)
   - ProT (Protein-DNA Twist)
   - Roll (Helical Roll)
   - **Resolution**: Computed for each of 147 bp positions

### Validation Strategy
- **K-Fold Cross-Validation**: 3-fold stratified split
- **Training**: 20 epochs per fold with early stopping
- **Evaluation**: Full dataset (10,400 sequences) using best model from fold 1

## Project Structure

```
DeepNup/
├── README.md                               (this file)
│
├── DeepNupShape/                           (Sequence + Shape variant)
│   ├── README.md                           (detailed documentation)
│   ├── Data_encoded.py                     (data preprocessing)
│   ├── model_dn.py                         (model architectures)
│   ├── training.py                         (training pipeline)
│   ├── predict.py                          (evaluation script)
│   ├── evaluator.py                        (metrics & utilities)
│   ├── encoded_shapes/                     (generated data)
│   ├── results_original/                   (trained models & outputs)
│   └── Dataset/                            (input FASTA & shapes)
│
└── [Original DeepNup scripts]              (sequence-only variant)
    ├── Data_encoded.py
    ├── model_dn.py
    ├── training.py
    ├── predict.py
    └── evaluator.py
```

## Technical Details

### Model Parameters
- **Batch Size**: 64
- **Learning Rate**: 0.0003 (Adam optimizer)
- **Loss Function**: Binary Crossentropy
- **Early Stopping Patience**: 15 epochs
- **Metrics**: Accuracy, AUC, Precision, Recall, F1 Score

### Evaluation Metrics
- **AUC-ROC**: Area Under Receiver Operating Characteristic curve
- **Sensitivity (TPR)**: True Positive Rate
- **Specificity (TNR)**: True Negative Rate
- **MCC**: Matthews Correlation Coefficient (handles class imbalance)
- **F1 Score**: Harmonic mean of precision and recall

## Interpretation

### Why Did Shape Features Only Improve AUC by ~2%?

1. **Complementary but Limited Signal**: DNA shape provides spatial information but may be partially correlated with sequence content
2. **Dataset Limitations**: Current dataset (~50/50 class balance) may not fully leverage shape advantages
3. **Architecture Constraints**: Simpler fusion strategy (concatenation) may not optimally integrate heterogeneous features
4. **Species-Specific Patterns**: Nucleosome positioning in *D. melanogaster* may be primarily sequence-driven

### Future Optimization Directions

- 🔄 **Advanced Fusion**: Use attention mechanisms to weight feature contributions dynamically
- 🧠 **Deeper Networks**: Implement full MLSNet architecture with StokenAttention for better pattern learning
- 🔬 **Multi-Species Training**: Transfer learning across species to improve generalization
- ⚖️ **Feature Engineering**: Explore interactions between sequence and shape features
- 📊 **Hyperparameter Optimization**: Grid/Bayesian search for architecture parameters

## Dependencies

```
TensorFlow >= 2.10
Keras >= 2.10
NumPy >= 1.21
Scikit-learn >= 1.0
Matplotlib >= 3.4
Biopython >= 1.79
```

## Installation

```bash
# Clone and navigate
cd DeepNup

# Install dependencies
pip install tensorflow keras numpy scikit-learn matplotlib biopython

# For shape feature extraction (if generating new data)
# Requires R and DNAShapeR package (not included)
```

## Usage

### Quick Start: Sequence + Shape Model

```bash
cd DeepNupShape

# Step 1: Preprocess sequence and shape data
python Data_encoded.py \
    -p "Dataset/Sequence" \
    -sp "Dataset/Shape" \
    -f "nucleosomes_vs_linkers_melanogaster.fas" \
    -o "encoded_shapes"

# Step 2: Train with 3-fold cross-validation
python training.py \
    -p "encoded_shapes" \
    -o "results_original" \
    -e "Mel_Original"

# Step 3: Evaluate on full dataset
python predict.py \
    -p "results_original" \
    -e "Mel_Original" \
    -f "folds_shape.pickle" \
    -pn "DeepNup with Shapes"
```

### Output Files

- **Models**: `results_original/Mel_Original/models/dn_shape/dn_shape_bestModel-fold*.keras`
- **Plots**: `results_original/Mel_Original/models/dn_shape/plot/`
- **Metrics**: Console output and saved evaluation results

## Citation

This project builds upon and extends the following works:

### 1. DeepNup (Original Model)
```bibtex
@inproceedings{liang2017deepnup,
  title={DeepNup: predicting nucleosome positioning from DNA sequences},
  author={Liang, Wenzheng and others},
  year={2017},
  url={https://github.com/lennylv/DeepNup.git}
}
```
Repository: [https://github.com/lennylv/DeepNup.git](https://github.com/lennylv/DeepNup.git)

### 2. MLSNet (Feature Learning Architecture)
```bibtex
@inproceedings{mlsnet,
  title={MLSNet: A Multi-Level SpatioTemporal Network for Action Recognition},
  author={Ming-hai Sun and others},
  year={2020},
  url={https://github.com/minghaidea/MLSNet.git}
}
```
Repository: [https://github.com/minghaidea/MLSNet.git](https://github.com/minghaidea/MLSNet.git)

Adapted techniques:
- **StokenAttention**: Efficient spatial attention mechanism
- **Multi-scale convolution**: Multi-branch processing with varying kernel sizes
- **Late fusion strategy**: Combining features after independent transformations

### 3. DNAShapeR (Shape Feature Extraction)
Used for computing DNA physicochemical shape features:
- Zhou, T., Yang, L., Lu, Y., et al. (2013). DNAShapeR: an R/Bioconductor package for DNA shape prediction based on sequence
- Features: EP, HelT, MGW, ProT, Roll

## Author Notes

This implementation explores the hypothesis that DNA structural properties (shape) complement sequence information for nucleosome position prediction. While results show modest improvements, they demonstrate the value of multi-modal feature integration in genomics deep learning.

### Recommendations

- ✅ Use sequence-only model for baseline/reference
- ✅ Consider shape features for improved sensitivity/specificity balance
- ⏳ Evaluate MLSNet variant on larger/more diverse datasets
- 📝 Investigate feature interactions for better fusion strategies

## License

Follows the license of original DeepNup project. See respective repositories for details.
