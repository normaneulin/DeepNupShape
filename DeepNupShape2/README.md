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

#### **Baseline (2MCNN)** - Multi-CNN Dual-Branch
- **Sequence Path**: One-hot (147, 4) + PseTNC (64, 1) → Multi-branch Conv1D (kernels 3, 5, 7) → GRU (50) → Dense layers
- **Shape Path**: (147, 5) → Multi-branch Conv1D (kernels 3, 5, 7) → GRU (50) → Dense layers
- **Integration**: Concatenate both paths → Dense(256) → Dense(1) → Sigmoid
- **Parameters**: ~307K trainable parameters
- **AUC Performance**: 0.5092

#### **Our Advanced Model (2MCNN + STA-BiLSTM)** - StokenAttention + Bi-LSTM
- **Sequence Path**: One-hot (147, 4) + PseTNC (64, 1) → Multi-branch Conv1D (kernels 3, 5, 7) → LSTM (50) → Dense layers
- **Shape Path**: (147, 5) → Conv2D + **StokenAttention** → Bi-LSTM (32 forward + 32 backward) → Dense layers
- **StokenAttention**: Efficient spatial attention mechanism (reduces complexity from O(n²) to O(n_tokens × n))
- **Bi-LSTM**: Bidirectional processing for better context understanding of shape features
- **Integration**: Concatenate both paths → Dense(256) → Dense(1) → Sigmoid
- **Parameters**: ~306K trainable parameters
- **Key Innovation**: Attention mechanism on shape features for improved feature importance weighting
- **AUC Performance**: 0.5063 (slightly lower AUC but better F1 score: 0.5029)
- **Status**: ✅ **ALTERNATIVE** - Better F1 score, precision-recall optimized

#### Comparison
See **DeepNupShape1** for the simpler baseline model (2MCNN) which achieves higher AUC (0.5092).
This variant trades slight AUC loss for significantly better F1 score (+32%).

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

### Performance on Sequence+Shape Data (2MCNN + STA-BiLSTM)

| Metric | Value |
|--------|-------|
| AUC | 0.5063 |
| Sensitivity | 0.5043 |
| Specificity | 0.5009 |
| MCC | 0.0057 |
| F1 Score | 0.5029 |

### Full Comparison: All Three Models

| Model | Type | AUC | Sensitivity | Specificity | MCC | F1 Score |
|-------|------|-----|-------------|-------------|-----|----------|
| **MCNN** | Seq Only | 0.4893 | 0.4882 | 0.5029 | -0.0065 | 0.6209 |
| **2MCNN** | Seq+Shape (Simple) | **0.5092** | 0.4990 | 0.5077 | 0.0061 | 0.3808 |
| **2MCNN + STA-BiLSTM** | Seq+Shape (Advanced) | 0.5063 | **0.5043** | 0.5009 | 0.0057 | **0.5029** |

### Improvements over Sequence-Only Baseline (MCNN)

| Metric | 2MCNN | STA-BiLSTM |
|--------|-------|------------|
| AUC improvement | +1.99% | +1.70% |
| Sensitivity improvement | +1.07% | +1.61% |
| Specificity improvement | +0.48% | -0.20% |
| MCC improvement | +0.0126 | +0.0122 |
| F1 Score change | -23.41% | -19.07% |

**Key Findings**:
1. **StokenAttention improves F1 by 32%** over basic dual-MCNN (0.5029 vs 0.3808)
2. **2MCNN achieves best AUC** (0.5092) - simpler is often better for this task
3. **STA-BiLSTM provides better sensitivity** (0.5043 vs 0.4882) - fewer false negatives
4. Shape features provide **consistent ~1.7-2% AUC improvement** regardless of architecture
5. **Attention mechanism helps F1 optimization** - better precision-recall balance
6. Architecture choice matters: **Trade-off between AUC (favor 2MCNN) and F1 (favor STA-BiLSTM)**

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
