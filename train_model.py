from dataloader import loadImages
import tensorflow as tf
from model import unet
import numpy as np
#pip install focal-loss
from focal_loss import SparseCategoricalFocalLoss
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True) # Enable XLA

#you have to run the program in  __name__ == "__main__" 
#because the program loads the data using more than one threads 
#if you dont run the program in main you will get an error 

if __name__ == "__main__":
    #first train the network for 150 epochs using the cityscapes data set
    #the training take around 6-8 hours on a rtx 3090

    #change the data directory if you have the data in diferent folder 
    x_path  = "datasets/city scapes/leftImg8bit/train" 
    y_path = "datasets/city scapes/gtFine/train"
    X, Y = loadImages(
        x_path=x_path, y_path=y_path, images_scale=1/2.6, number_of_images = 3000, skip_images = 0, 
        x_dtype="uint8", y_dtype="sparce", parallel=True, threads=12, flip=True, 
        data_from = "cityscapes", category2names = {0:["vehicle"], 1:["road"]}
        )
    x_path  = "datasets/city scapes/leftImg8bit/val"
    y_path = "datasets/city scapes/gtFine/val"
    X_val, Y_val = loadImages(
        x_path=x_path, y_path=y_path, images_scale=1/2.6, number_of_images = 600, skip_images = 0, 
        x_dtype="uint8", y_dtype="sparce", parallel=True, threads=12,
        data_from = "cityscapes", category2names = {0:["vehicle"], 1:["road"]}
        )

    #we get images of shape 393*787 for unet to work the images must be divisible by 2, 5 times 
    #2^5 = 32 so they must be divisible by 32, so we shrink the image to 384x768 because 384 and 768 is divisible by 32 
    
    X = X[:, :384, :768, :]
    Y = Y[:, :384, :768, :]
    X_val = X_val[:, :384, :768, :]
    Y_val = Y_val[:, :384, :768, :]

    callbacks = [tf.keras.callbacks.TensorBoard("unet_logs")]
    sparse = True
    metrics = [tf.keras.metrics.IoU(num_classes=3, name = "cat0_IOU", target_class_ids = [0], sparse_y_true= sparse, sparse_y_pred=False), 
            tf.keras.metrics.IoU(num_classes=3, name = "cat1_IOU", target_class_ids = [1], sparse_y_true= sparse, sparse_y_pred=False),
            tf.keras.metrics.IoU(num_classes=3, name = "cat2_IOU", target_class_ids = [2], sparse_y_true= sparse, sparse_y_pred=False),
            tf.keras.metrics.MeanIoU(num_classes=3, name = "Mean_IOU", sparse_y_true= sparse, sparse_y_pred=False)
            ]
    
    #train model
    model = unet(n_classes=3, input_shape=(384,768,3))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = SparseCategoricalFocalLoss(gamma=2) , metrics = metrics)
    model.fit(X, Y, validation_data=(X_val, Y_val), batch_size = 16, epochs = 150, verbose = 0, callbacks = None) 
    
    #now train the model for the data collected in the carla simuator 20 epochs are enough 
    x_path = "datasets/carla/train/x"
    y_path = "datasets/carla/train/y"
    X, Y = loadImages(
        x_path=x_path, y_path=y_path, images_scale=1, number_of_images = 300, skip_images = 0, 
        x_dtype="uint8", y_dtype="sparce", parallel=True, threads=12, flip=True, 
        data_from = "carla", category2names = {0:["vehicle"], 1:["road", "roadline"]}
        )
    model.fit(X, Y, batch_size = 16, epochs = 20, verbose = 0, callbacks = None) 

    model.save("./unet")
