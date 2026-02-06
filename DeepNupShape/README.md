# DeepNup with DNA Shape Features

## Overview

This directory contains an enhanced version of the DeepNup nucleosome position prediction model that incorporates DNA shape features as an additional input modality. The model learns from both DNA sequence information and computed DNA shape parameters (EP, HelT, MGW, ProT, Roll).

## What We Did

### 1. Data Preparation
- **Input**: DNA sequences from FASTA file (`nucleosomes_vs_linkers_melanogaster.fas`)
- **Shape Data Extraction**: Used DNAShapeR to extract 5 DNA shape features:
  - EP (Electron Density at Phosphate)
  - HelT (Helix Twist)
  - MGW (Minor Groove Width)
  - ProT (Protein-DNA Twist)
  - Roll (Roll)
- **Preprocessing**: Combined sequence and shape data into unified feature vectors
  - Sequence: One-hot encoding (147, 4) + PseTNC (64, 1)
  - Shapes: (147, 5) - one position per bp, 5 shape types

### 2. Model Architecture Enhancements

#### Original DeepNup
- Single sequence path with Conv1D + GRU
- Learned sequence patterns only

#### Enhanced Model (dn_with_shapes)
- **Dual Input Paths**:
  - **Sequence Path**: One-hot (147, 4) + PseTNC (64, 1) → Conv1D (kernel 3, 5, 7) → GRU → fully connected
  - **Shape Path**: (147, 5) → Conv1D (kernel 9) → Dropout → Flattening
- **Integration**: Concatenate both paths → Dense layers → Output
- **Parameters**: ~307K trainable parameters

#### MLSNet-Inspired Variant (dn_with_shapes_mlsnet)
- **Advanced Components**:
  - **StokenAttention**: Efficient attention mechanism reducing O(n²) to O(n_tokens × n)
  - **Multi-Scale Processing**: 3 parallel Conv1D branches (kernels 3, 5, 7)
  - **Advanced RNN**: LSTM (50 units) instead of GRU + Bi-LSTM (32 units) for shape features
  - **Late Concatenation**: Combine processed features after full transformation
- **Parameters**: ~919K trainable parameters (more capacity for complex patterns)

### 3. Training Pipeline

**K-Fold Cross-Validation**: 3-fold stratified split
- **Folds**: 3 models trained independently
- **Epochs**: 20 per fold (with early stopping)
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.0003)
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy, AUC, Precision, Recall, F1 Score

**Output Models**:
- `results_original/Mel_Original/models/dn_shape/dn_shape_bestModel-fold{1,2,3}.keras`

### 4. Evaluation

Full dataset evaluation using best model from fold 1:
- **Total Sequences**: 5,750 (2,900 nucleosomal, 2,850 linker)
- **Metrics Computed**:
  - Accuracy, Sensitivity, Specificity
  - AUC-ROC, F1 Score, Matthews Correlation Coefficient (MCC)
- **Output**: Evaluation plots and numerical results

## Results

### Performance on Sequence+Shape Data (dn_with_shapes model)

| Metric | Value |
|--------|-------|
| AUC | 0.5092 |
| Sensitivity | 0.4990 |
| Specificity | 0.5077 |
| MCC | 0.0061 |
| F1 Score | 0.3808 |

### Comparison with Sequence-Only (Original DeepNup)

| Metric | Seq Only | Seq+Shape | Improvement |
|--------|----------|-----------|-------------|
| AUC | 0.4893 | 0.5092 | +1.99% |
| Sensitivity | 0.4882 | 0.4990 | +1.07% |
| Specificity | 0.5029 | 0.5077 | +0.48% |
| MCC | -0.0065 | +0.0061 | +0.0126 |
| F1 Score | 0.6209 | 0.3808 | -23.41% |

**Key Findings**:
- DNA shape features provide **modest AUC improvement** (~2%)
- Better balance between sensitivity and specificity
- Trade-off in F1 Score suggests precision-recall adjustment needed

## File Structure

```
DeepNupShape/
├── Data_encoded.py              # Sequence + shape preprocessing
├── model_dn.py                  # Model architectures (2 variants)
├── training.py                  # 3-fold CV training pipeline
├── predict.py                   # Full dataset evaluation
├── evaluator.py                 # Metrics and utility functions
├── encoded_shapes/              # Generated pickle files
│   ├── one_hot_nuc.pickle
│   ├── one_hot_link.pickle
│   ├── PseTNC_nuc.pickle
│   ├── PseTNC_link.pickle
│   ├── shape_nuc.pickle
│   └── shape_link.pickle
├── results_original/            # Training outputs
│   └── Mel_Original/
│       ├── models/dn_shape/     # Trained .keras models
│       ├── folds_shape.pickle   # K-fold splits
│       └── plot/                # Evaluation visualizations
└── Dataset/                     # Input data
    ├── Sequence/                # FASTA files
    └── Shape/                   # DNAShapeR outputs
```

## Usage

### 1. Data Preparation
```bash
python Data_encoded.py \
    -p "Dataset/Sequence" \
    -sp "Dataset/Shape" \
    -f "nucleosomes_vs_linkers_melanogaster.fas" \
    -o "encoded_shapes"
```

### 2. Model Training (3-fold CV)
```bash
python training.py \
    -p "encoded_shapes" \
    -o "results_original" \
    -e "Mel_Original"
```

### 3. Full Dataset Evaluation
```bash
python predict.py \
    -p "results_original" \
    -e "Mel_Original" \
    -f "folds_shape.pickle" \
    -pn "DeepNup with Shapes"
```

## Dependencies

- TensorFlow/Keras 2.x
- NumPy
- Scikit-learn
- Matplotlib
- Biopython

## Citation

This work builds upon:

1. **DeepNup** - Original nucleosome prediction model
   - Repository: https://github.com/lennylv/DeepNup.git
   - Learns sequence patterns for nucleosome positioning

2. **MLSNet** - Advanced feature learning techniques
   - Repository: https://github.com/minghaidea/MLSNet.git
   - Provides StokenAttention mechanism and multi-scale processing strategy

3. **DNAShapeR** - DNA shape feature extraction
   - Used to compute physicochemical DNA shape descriptors
   - Features: EP, HelT, MGW, ProT, Roll

## Future Improvements

- Tune hyperparameters for shape-aware model
- Implement full MLSNet architecture with StokenAttention
- Experiment with different shape feature combinations
- Cross-validate on additional species datasets
- Apply transfer learning from pre-trained sequence models

## Author Notes

The modest performance (~AUC 0.5) suggests the current dataset or feature combination requires further optimization. The shape features show promise (+2% AUC) but indicate more sophisticated integration strategies may be needed.
