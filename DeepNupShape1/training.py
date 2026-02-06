import argparse
import model_dn
import evaluator
import os
import pickle
import numpy as np
import tensorflow as tf
import pdb


parser = argparse.ArgumentParser(description='Nucleosome Classification Experiment with DNA Shape')
parser.add_argument('-p', '--path', dest='path', type=str, default=r"D:\DeepNup\code_and_data\data\setting1\pickle_H",
                    help='Pickle file Path (contains encoded sequences)')

parser.add_argument('-ohn', '--ohnuc', dest='nuc_one_hot_Pickle', type=str, default="one_hot_nuc.pickle",
                    help='Nucleosome one-hot filename')
parser.add_argument('-ohl', '--ohlin', dest='link_one_hot_Pickle', type=str, default="one_hot_link.pickle",
                    help='Linker one-hot filename')

parser.add_argument('-pn', '--pnuc', dest='nuc_PseTNC_Pickle', type=str, default="PseTNC_nuc.pickle",
                    help='Nucleosome PseTNC filename')
parser.add_argument('-pl', '--plin', dest='link_PseTNC_Pickle', type=str, default="PseTNC_link.pickle",
                    help='Linker PseTNC filename')

parser.add_argument('-sn', '--snuc', dest='nuc_shape_Pickle', type=str, default="shape_nuc.pickle",
                    help='Nucleosome shape features filename')
parser.add_argument('-sl', '--slin', dest='link_shape_Pickle', type=str, default="shape_link.pickle",
                    help='Linker shape features filename')

parser.add_argument('-o', '--out', dest='outPath', type=str, default=r"D:\DeepNup\result",
                    help='Output file Path')
parser.add_argument('-e', '--experiments', dest='exp', default='Experiment_H_Shape',
                    help='Experiments Name')
parser.add_argument('-f', '-foldName', dest='foldName', default="folds_shape.pickle",
                    help='Folds Filename')

args = parser.parse_args()
inPath = args.path
nuc_one_hot = args.nuc_one_hot_Pickle
link_one_hot = args.link_one_hot_Pickle
nuc_PseTNC = args.nuc_PseTNC_Pickle
link_PseTNC = args.link_PseTNC_Pickle
nuc_shape = args.nuc_shape_Pickle
link_shape = args.link_shape_Pickle

outPath = args.outPath
expName = args.exp
foldName = args.foldName

metricsList = [evaluator.acc, evaluator.precision, evaluator.recall, evaluator.f1score, evaluator.aucScore]

epochs = 20
batch_size = 64
shuffle = False 
seed = None  

# Model identifier
m = "dn_shape"

# Create and set model save dir
modelPath = os.path.join(outPath, expName, "models", m)
if (not os.path.isdir(modelPath)):
    os.makedirs(modelPath)

# Load nucleosome and linker data
print("Loading one-hot encoded sequences...")
with open(os.path.join(inPath, nuc_one_hot), "rb") as fp:
    nuc_one_hot_list = pickle.load(fp)
with open(os.path.join(inPath, link_one_hot), "rb") as fp:
    link_one_hot_list = pickle.load(fp)

print("Loading PseTNC features...")
with open(os.path.join(inPath, nuc_PseTNC), "rb") as fp:
    nuc_PseTNC_list = pickle.load(fp)
with open(os.path.join(inPath, link_PseTNC), "rb") as fp:
    link_PseTNC_list = pickle.load(fp)

print("Loading shape features...")
with open(os.path.join(inPath, nuc_shape), "rb") as fp:
    nuc_shape_list = pickle.load(fp)
with open(os.path.join(inPath, link_shape), "rb") as fp:
    link_shape_list = pickle.load(fp)

# Verify data consistency
print(f"Nucleosomal sequences: {len(nuc_one_hot_list)}")
print(f"Linker sequences: {len(link_one_hot_list)}")
assert len(nuc_one_hot_list) == len(nuc_shape_list), "Nucleosomal data mismatch!"
assert len(link_one_hot_list) == len(link_shape_list), "Linker data mismatch!"

# Create combined dataset with labels
labels = np.concatenate(
    (np.ones((len(nuc_one_hot_list), 1), dtype=np.float32), 
     np.zeros((len(link_one_hot_list), 1), dtype=np.float32)),
    axis=0)

one_hot_feature = np.concatenate((nuc_one_hot_list, link_one_hot_list), 0)
PseTNC_feature = np.concatenate((nuc_PseTNC_list, link_PseTNC_list), 0)
shape_feature = np.concatenate((nuc_shape_list, link_shape_list), 0)

data1 = one_hot_feature      # (N, 147, 4)
data2 = PseTNC_feature        # (N, 64, 1)
data3 = shape_feature         # (N, 5, 147)

print(f"One-hot shape: {data1.shape}")
print(f"PseTNC shape: {data2.shape}")
print(f"Shape data shape: {data3.shape}")

# Build k-fold cross-validation splits
print("Building k-fold cross-validation splits...")
foldPath = os.path.join(outPath, expName, foldName)
folds = evaluator.build_kfold_with_shape(data1, data2, data3, labels, k=3, shuffle=shuffle, seed=seed)

with open(foldPath, "wb") as fp:
    pickle.dump(folds, fp)

# Training
evaluations = {
    "Accuracy": [],
    "Precision": [],
    "TPR": [],
    "FPR": [],
    "AUC": [],
    "Sensitivity": [],
    "Specificity": [],
    "MCC": []
}

i = 1
for fold in folds:
    tf.keras.backend.clear_session()
    
    modelCallbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(modelPath, "{}_bestModel-fold{}.keras".format(m, i)),
                                           monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                                           mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto',
                                         baseline=None, restore_best_weights=False)
    ]
    
    # Create model with three input paths
    model = model_dn.dn_with_shapes(metrics=metricsList)
    print(f"\n{'='*60}")
    print(f"Fold {i}/{len(folds)}")
    print(f"{'='*60}")
    print(model.summary())
    
    # Train on three inputs
    model.fit(x=[fold["X1_train"], fold['X2_train'], fold['X3_train']], 
              y=fold["y_train"],
              batch_size=batch_size, epochs=epochs, verbose=1, callbacks=modelCallbacks, 
              validation_split=0.05, validation_freq=1)

    i += 1

del model
print("\nTraining complete!")
