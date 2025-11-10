import os  
import numpy as np  
  
from train import (weights_path,  
                   get_model,  
                   normalize,  
                   NB_CLASSES)  
  
  
def predict(x, model, nb_classes=1):  
    """  
    Runs model in inference mode on given input x  
    x: input image of shape [H, W, nb_channels]  
    """  
    x_batch = np.expand_dims(x, axis=0)  
    prediction = model.predict(x_batch, batch_size=1)  
    return prediction[0]  
  
  
def picture_from_mask(mask, threshold=0.5):  
    """  
    Returns a binary mask visualization  
    mask: mask of shape (height, width, 1)  
    """  
    binary_mask = (mask[:, :, 0] > threshold).astype(np.uint8) * 255  
    pict = np.stack([binary_mask, binary_mask, binary_mask], axis=0)  
    return pict  
  
  
if __name__ == "__main__":  
    data_path = 'data/'  
    X_test = np.load(os.path.join(data_path, 'X_test_256.npy'))  
      
    if not os.path.exists('results'):  
        os.makedirs('results')  
      
    model = get_model()  
    model.load_weights(weights_path)  
      
    predictions = []  
    for idx in range(len(X_test)):  
        img = X_test[idx].transpose([1, 2, 0]) / 3.0  
        prediction = predict(img, model, nb_classes=1)  
        predictions.append(prediction)  
        print(f"Processed {idx + 1}/{len(X_test)}")  
      
    # Save all predictions  
    predictions_array = np.array(predictions)  
    np.save('results/test_predictions.npy', predictions_array)  
    print(f"Saved predictions with shape: {predictions_array.shape}")