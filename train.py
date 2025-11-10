# MINE

import os.path
import random
import math

import numpy as np

#from patches import get_patches
from unet    import unet_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

import tensorflow as tf  
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))  
print("TensorFlow version:", tf.__version__)  
print("Built with CUDA:", tf.test.is_built_with_cuda())


"""
returns image normalized to be in [-1, 1]
"""
def normalize(img):
    minv = img.min()
    maxv = img.max()
    return 2.0 * (img - minv) / (maxv - minv) - 1.0


NB_BANDS      = 16     # CHANGED
NB_CLASSES    = 1     # CHANGED
NB_EPOCHS     = 50
BATCH_SIZE    = 64
UPCONV        = True
# PATCH_SIZE    = 128   # should be divisible by 16
# NB_TRAIN      = 1200
# NB_VAL        = 300


def get_model():
    return unet_model(NB_CLASSES, 256, nb_channels=NB_BANDS, upconv=UPCONV)


data_path = 'data/'
weights_path = 'weights/'  
if not os.path.exists(weights_path):  
    os.makedirs(weights_path)  
weights_path = os.path.join(weights_path, 'unet_weights.keras')  # CHANGED from .hdf5
train_ids = [str(i) for i in range(1713)]  # CHANGED

if __name__ == '__main__':

    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VAL   = dict()
    Y_DICT_VAL   = dict()


    # CHANGED till Done reading images
    print("Reading images")  
    
    # Load the entire datasets  
    X_train_full = np.load(os.path.join(data_path, 'X_train_256.npy'))  # Shape: [1713, 16, 256, 256]  
    Y_train_full = np.load(os.path.join(data_path, 'Y_train_256.npy'))  # Shape: [1713, 256, 256]  
    
    print(f"Loaded X_train shape: {X_train_full.shape}")  
    print(f"Loaded Y_train shape: {Y_train_full.shape}")  
    
    # Split into train/val: 75% train, 25% validation  
    split_idx = int(0.75 * len(X_train_full))  # ~1285 for train, ~428 for val  
    
    # Process training images  
    for i in range(split_idx):  
        # Transpose from [16, 256, 256] to [256, 256, 16]  
        img_m = X_train_full[i].transpose([1, 2, 0]) / 3.0  # Normalize from [-3, 3] to [-1, 1]  
        
        # Expand mask dimensions: [256, 256] -> [256, 256, 1]  
        mask = np.expand_dims(Y_train_full[i], axis=-1).astype(np.float32)  
        
        X_DICT_TRAIN[str(i)] = img_m  
        Y_DICT_TRAIN[str(i)] = mask  
    
    print(f"Loaded {len(X_DICT_TRAIN)} training images")  
    
    # Process validation images  
    for i in range(split_idx, len(X_train_full)):  
        img_m = X_train_full[i].transpose([1, 2, 0]) / 3.0  
        mask = np.expand_dims(Y_train_full[i], axis=-1).astype(np.float32)  
        
        X_DICT_VAL[str(i)] = img_m  
        Y_DICT_VAL[str(i)] = mask  
    
    print(f"Loaded {len(X_DICT_VAL)} validation images")  
    print("Done reading images")


    def train_net():
        print("Started training")  
      
        # Convert dictionaries to numpy arrays  
        x_train = np.array([X_DICT_TRAIN[img_id] for img_id in X_DICT_TRAIN.keys()])  
        y_train = np.array([Y_DICT_TRAIN[img_id] for img_id in Y_DICT_TRAIN.keys()])  
        x_val = np.array([X_DICT_VAL[img_id] for img_id in X_DICT_VAL.keys()])  
        y_val = np.array([Y_DICT_VAL[img_id] for img_id in Y_DICT_VAL.keys()])  
        
        print(f"Training data shape: {x_train.shape}, {y_train.shape}")  
        print(f"Validation data shape: {x_val.shape}, {y_val.shape}")  
        
        model = get_model()  
        
        # load saved weights  
        if os.path.isfile(weights_path):  
            model.load_weights(weights_path)  
        
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)  
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=',')  
        
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS,  
                verbose=2, shuffle=True, callbacks=[model_checkpoint, csv_logger],  
                validation_data=(x_val, y_val))  
        
        return model

    print(f"Training images: {len(X_DICT_TRAIN)}")  
    print(f"Validation images: {len(X_DICT_VAL)}")  
        
    sample_id = list(X_DICT_TRAIN.keys())[0]  
    print(f"Sample image shape: {X_DICT_TRAIN[sample_id].shape}")  # Should be [256, 256, 16]  
    print(f"Sample mask shape: {Y_DICT_TRAIN[sample_id].shape}")   # Should be [256, 256, 1]  

    train_net()



    # import numpy as np
# import matplotlib.pyplot as plt

# # --- 1. Define Correct File Paths ---
# X_train_path = 'data/X_train_256.npy'
# Y_train_path = 'data/Y_train_256.npy'

# # --- 2. Load Data using Memory-Mapping ---
# # mmap_mode='r' prevents the entire huge file from being read into RAM.
# try:
#     X_train = np.load(X_train_path, mmap_mode='r')
#     Y_train = np.load(Y_train_path, mmap_mode='r')
    
#     print("Data memory-mapped successfully!")

# except FileNotFoundError:
#     print("Error: Check your file paths or folder structure.")
#     exit()

# # --- 3. Print Dimensions (Shapes) ---
# print("\n--- Dataset Dimensions ---")
# print(f"X_train (Images) Full Shape: {X_train.shape}")
# print(f"Y_train (Labels) Full Shape: {Y_train.shape}")

# # Extract sample and resolution info
# N, C, H, W = X_train.shape
# print(f"\nSummary: {N} samples, {C} channels, {H}x{W} resolution.")

# # --- 4. Select a Single Sample for Visualization ---
# sample_index = 0  # Look at the first image/mask pair
# image_sample = X_train[sample_index]
# mask_sample = Y_train[sample_index]

# print(f"\nSingle Image Shape: {image_sample.shape}")
# print(f"Single Mask Shape: {mask_sample.shape}")



