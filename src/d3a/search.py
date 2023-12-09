import inspect
import os 
import sys

# Setting up path for importing local modules
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# Configuring TensorFlow to minimize log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
import traceback
import random
import pickle as pk
from multiprocessing import Pool
from glob import glob
import importlib
import logging
import nets
import random 

# Setting up logging
logging.basicConfig(level = logging.INFO)
logging.info("Working dir: " + os.getcwd())

TSLEN = 128

models = ['mscnn', 'bilstm']
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    # Adding optional argument
    parser.add_argument("-d", "--dir", help = "Directory where found the params file", type=str, required=True)
    parser.add_argument("-g", "--gpu", help = "GPU to use [0, 1]", choices=["0", "1"], required=True)
    
    
    
    # Read arguments from command line
    args = parser.parse_args()
        
    logging.info(f"Using GPU {args.gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import tensorflow as tf
    from meta import compute_metaattributes, maxmin_attributes
    import data
    from d3a import get_SAT_arrays, A2SDiffusionGenerator, Att2SignalGenerator
    from d3a import net
    from d3a import meta

    
    def show_trained_results(file):
        history = json.load(open(file, 'r'))       
        for key, values in history.items():
            logging.info(f"{key}: {values[-1]}")
    
    def train_S2A_model(result_dir, config,  X_train, A_train, X_test, A_test):
        """
        Train a model to estimate the meta-attributes of a time-series.

        Parameters:
        - result_dir (str): Directory to save the trained model.
        - config (dict): Dictionary with model hyperparameters.
        - X_train: Training time-series.
        - A_train: Training meta-attributes.
        - X_test: Test time-series.
        - A_test: Test meta-attributes.
        """
        
        # Define filenames for saving/loading the S2A model
        S2A_model_filename = os.path.join(result_dir, "s2a.h5") 
        S2A_model_history_filename = os.path.join(result_dir, "s2a.json") 
        
        # Check if the trained S2A model already exists
        if os.path.exists(S2A_model_filename):
            # Load the pre-trained model
            logging.info("Reading signal to attributed trained model")
            s2a_model = tf.keras.models.load_model(S2A_model_filename)
            show_trained_results(S2A_model_history_filename) 
        else:
            # Train a new S2A model if it doesn't exist
            logging.info("Training signal to attribute model")
            
            # Create the S2A model 
            s2a_model = meta.create_S2A_model(TSLEN, A_train.shape[1], **config)
            
            # Train the model using the provided training data
            history = s2a_model.fit(
                X_train, A_train, validation_data=(X_test, A_test), 
                batch_size=128, epochs=1000, 
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
            )
            
            # Save the trained model and its training history
            s2a_model.save(S2A_model_filename)
            json.dump(history.history, open(S2A_model_history_filename, 'w')) 
            
        return s2a_model
    
    def train_denoise_model(result_dir,  s2a_model, config, X_train, A_train, X_test, A_test):
        """
        Train a denoising model for time-series. The model can be an autoencoder or a diffusion model.

        Parameters:
        - result_dir (str): Directory to save the trained model.
        - s2a_model: Model to predict the meta-attributes of a time-series.
        - config (dict): Dictionary with model hyperparameters.
        - X_train: Training time-series.
        - A_train: Training meta-attributes (used for metric purposes).
        - X_test: Test time-series.
        - A_test: Test meta-attributes (used for metric purposes).
        """
        
        # Extract hyperparameters from the configuration
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        learning_rate = config['learning_rate']
        cond = config['cond']
        feature_loss = config['feature_loss']
        timesteps = config['timesteps']
        
        # Determine the model class: AutoEncoder or DiffusionModel        
        if not 'class' in config or config['class'] == 'noise_renoval':
            prefix = ''
            noise_rates = [float(v) for v in config['noise_rates']]
            
            logging.info("Noise rates: " + str(noise_rates))
    
             # Create an AutoEncoderModel
            dm = net.AutoEncoderModel(
                    TSLEN, [16, 32, 32, 64], 
                    block_depth=2, 
                    noise_rates=noise_rates, 
                    feature_loss_net=s2a_model, 
                    num_features=A_train.shape[1], 
                    feature_names=meta.ATTRIBUTE_NAMES,
                    cond=cond, 
                    feature_loss=feature_loss, 
                    timesteps=timesteps)
            
        elif config['class'] == 'dpm':
            max_noise_level = config['max_noise_level']
            
            prefix = "dpm_"
            
            # Create a DiffusionModel
            dm = net.DiffusionModel(
                    TSLEN, [16, 32, 32, 64], 
                    block_depth=2, 
                    timesteps=timesteps, 
                    max_noise_level=max_noise_level,
                    feature_loss_net=s2a_model, 
                    num_features=A_train.shape[1], 
                    feature_names=meta.ATTRIBUTE_NAMES,
                    cond=cond, 
                    feature_loss=feature_loss)
            
        # Compile the denoising model
        dm.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss=tf.keras.losses.mean_absolute_error,
            run_eagerly=True 
        )
        
        # Define filenames for saving/loading the denoising model
        filename = f"diffusion_model_{prefix}c{int(cond)}_fl{int(feature_loss)}_ts{int(timesteps)}"
        dif_model_filename = os.path.join(result_dir, f"{filename}.h5") 
        dif_model_history_filename = os.path.join(result_dir, f"{filename}.json") 
        
        # Check if the trained denoising model already exists
        if os.path.exists(dif_model_filename):
            # Load the pre-trained model
            logging.info("Reading diffusion trained model")
            dm.network = tf.keras.models.load_model(dif_model_filename)
            show_trained_results(dif_model_history_filename)   
        else:
            # Clear temporary files
            for f in os.listdir('tmp'):
                os.remove(os.path.join('tmp', f))

            # Train the denoising model using the provided training data
            history = dm.fit(
                (np.expand_dims(X_train, axis=-1), np.expand_dims(A_train, axis=-1)),
                epochs=num_epochs,
                batch_size=batch_size,
                steps_per_epoch=min(100, X_train.shape[0]//batch_size),
                verbose=2,
                validation_data=(np.expand_dims(X_test, axis=-1)[:128], np.expand_dims(A_test, axis=-1)[:128]),
                callbacks=[
                    tf.keras.callbacks.LambdaCallback(on_epoch_end=dm.plot_images),
                ],
            )
            
            # Save the trained denoising model and its training history
            dm.network.save(dif_model_filename)
            json.dump(history.history, open(dif_model_history_filename, 'w'))
         
        # Return the trained denoising model  
        return dm
         

    def train_final_model(result_dir, model, model_config, lr, train_mode, dm,  X_tran, X_test, 
                          input_size, denoise_steps=1, noise_level=0.5):
        """
        Train the final model using raw signals, noisy samples, or a denoising model for data augmentation.

        Parameters:
        - result_dir (str): Directory to save the trained model.
        - model: Model architecture.
        - model_config (dict): Dictionary with model hyperparameters.
        - lr (float): Learning rate for training.
        - train_mode (bool): Indicates whether to train meta-attributes (used for metric purposes).
        - dm: Denoising model for data augmentation.
        - X_train: Training data.
        - X_test: Test data.
        - input_size: Size of the input.
        - denoise_steps (int): Number of denoising steps. Set to 0 to train the model with noisy samples (if noise_level > 0).
        - noise_level (float): Noise rate to add. Set to 0 to train the model with raw signals.
        """
        # Set default batch size
        batch_size = 32
        
        # Check if batch size is specified in the model configuration
        if 'batch_size' in model_config:
            batch_size = model_config.pop('batch_size')
            
        # Extract other hyperparameters
        patience = model_config.pop('patience')
        task_type = 'regression'
        
        # Check if task type is specified in the model configuration
        if 'task_type' in model_config:
            task_type = model_config.pop('task_type')
            
        logging.info(f"Task type: {task_type}")

        # Define the filename for saving/loading the final model's training history
        final_model_history_filename = os.path.join(result_dir, 
                                                    f"{model}_{train_mode}_NL{noise_level:0.02f}_DS{denoise_steps}_model.json") 
        
        # Check if the trained final model already exists
        if os.path.exists(final_model_history_filename):
            # Load the pre-trained model
            logging.info(f"Reading {model} model trained with {train_mode}")
            show_trained_results(final_model_history_filename)   
        else:
            # Train the final model based on the specified training mode
            logging.info(f"Training {model} model with {train_mode}")
            
            # Generate data using Att2SignalGenerator for training and validation
            val = Att2SignalGenerator(X_test, expand_dims=True, 
                                      return_attributes=False, batch_size=batch_size,
                                      batches_per_epoch=100)
            a2sgen_base = Att2SignalGenerator(X_train, expand_dims=True, 
                                              return_attributes=True, batch_size=batch_size,
                                              batches_per_epoch=1000)
            a2sgen = A2SDiffusionGenerator(dm, a2sgen_base, denoise_steps=denoise_steps, 
                                           noise_level=noise_level, raw_samples_factor=0)

            # Create the base model using the specified architecture
            model_base = getattr(getattr(nets, model), 'create_model')(input_size, **model_config)
            loss = tf.keras.losses.MeanSquaredError(name="mean_squared_error") 
            
            # Compile the model based on the task type
            if task_type == 'regression':
                model_base.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=loss, 
                               metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],)
            elif task_type == 'classification':
                model_base.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acurracy')],
                )
            elif task_type == 'binary':
                model_base.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), 
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'],
                )
            else:
                raise Exception('Invalid task type: ' + task_type)
                
            # Early stopping callback to prevent overfitting
            ea = tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss')
            logging.info(f"Training shape: {a2sgen[0][0].shape}, target shape: {a2sgen[0][1].shape}")
            logging.info(f"Test shape: {val[0][0].shape}, target shape: {val[0][1].shape}")
            
            # Train the model and save the training history
            history = model_base.fit(a2sgen, validation_data=val,
                                    epochs=200, steps_per_epoch=100,
                                    callbacks=[ea ])
            
            json.dump(history.history, open(final_model_history_filename, 'w'))
    
    def read_or_create_file(creator, filename):
        """
        Read the contents of a file if it exists, otherwise create the file using the provided creator function.

        Parameters:
        - creator (function): A function that creates the object to be stored in the file.
        - filename (str): The name of the file.

        Returns:
        - obj: The object read from or created for the file.
        """        
        if os.path.exists(filename):
            # If the file exists, read its contents
            return pk.load(open(filename, 'rb'))
        else:
            # If the file does not exist, create it using the provided creator function
            logging.info(f"File {filename} not found. Creating it...")
            obj = creator()
            pk.dump(obj, open(filename, 'wb'))
            return obj
            
    def create_metaatt(dataset, tslen, dtype):
        """
        Create meta-attributes for a given dataset and time-series length.

        Parameters:
        - dataset: The dataset from which to generate meta-attributes.
        - tslen (int): Time-series length.
        - dtype (str): 'train' or 'test' indicating whether to create attributes for training or testing.

        Returns:
        - attributes: The computed meta-attributes.
        """        
        if dtype == 'train':
            # Load generators for training data
            gen, _ = dataset.load_generators(return_test=False, tslen=tslen)
        elif dtype == 'test':
             # Load generators for testing data
            _, gen = dataset.load_generators(return_train=False, tslen=tslen)
            
        # Compute meta-attributes using the loaded generators and time-series length            
        _, attributes = compute_metaattributes(gen, tslen)
        
        return attributes
    
    def train():
        """
        This function manages the creation or retrieval of essential meta-attributes from the dataset, computes 
        working arrays, and, optionally, removes outliers. Following this, it calculates baseline metrics and 
        trains a model (A2T) to predict target values from meta-attributes. The script then proceeds to train 
        a model (S2A) predicting meta-attributes from raw time-series data or loads a pre-trained one. Subsequently, 
        a denoising model is trained for time-series data, and its history and model are saved. The final step 
        involves training the ultimate model using raw signals, noisy samples, or denoised data, depending on 
        the specified training mode.
        """
        global TSLEN
        
        # Load experiment configuration from params.json file
        experiment_config = json.load(open(os.path.join(args.dir, 'params.json'), 'r'))
        
        # Set up directories and import necessary modules
        base_dir = os.path.dirname(args.dir)
        metaatt_dir = os.path.dirname(base_dir)
        
        dataset = importlib.import_module(experiment_config['package'])
        
        result_dir = args.dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        # Update the global time-series length variable if specified in the configuration            
        if 'ts_len' in experiment_config:
            TSLEN = experiment_config['ts_len']
        
        logging.info(f"TS_LEN: {TSLEN}")
        
        s2a_config = experiment_config['s2a_net']

        
        # Read or create meta-attributes for training and testing
        filename = metaatt_dir + f'/train-meta-attributes_{TSLEN}.pk'
        train_attributes = read_or_create_file(lambda: create_metaatt(dataset, TSLEN, 'train'), filename)
        
        filename = metaatt_dir + f'/train-max_min_{TSLEN}.pk'
        train_max_min = read_or_create_file(lambda: maxmin_attributes(train_attributes), filename)
        
        filename = metaatt_dir + f'/test-meta-attributes_{TSLEN}.pk'
        test_attributes = read_or_create_file(lambda: create_metaatt(dataset, TSLEN, 'test'), filename)
        
        filename = metaatt_dir + f'/test-max_min_{TSLEN}.pk'
        test_max_min = read_or_create_file(lambda: maxmin_attributes(test_attributes), filename)
        
        
        logging.info("Computing working arrays") 
        X_train, A_train, T_train = get_SAT_arrays(train_attributes, train_max_min, TSLEN)
        X_test, A_test, T_test = get_SAT_arrays(test_attributes, train_max_min, TSLEN)

        # Remove outliers from meta-attributes if specified in the configuration        
        if 'remove_outliers' in experiment_config and experiment_config['remove_outliers'] == True:
            print(experiment_config['remove_outliers'])
            for i in range(A_train.shape[1]):
                A_train[:, i] = meta.remove_outliers(A_train[:, i])

            for i in range(A_test.shape[1]):
                A_test[:, i] = meta.remove_outliers(A_test[:, i])

        # Compute baseline for the dataset                
        dataset.compute_baseline(T_train, T_test)
       
        # Train the S2A (signal to attributes) model
        s2a_model = train_S2A_model(base_dir, s2a_config, X_train, A_train, X_test, A_test)
                                                   
        

        num_features = len(dataset.FEATURE_NAMES)
        input_size = (num_features, TSLEN, 1)
        logging.info(f"Input size: {input_size}")   
        
        
        # Train denoise model to make data augmentation
        dm  = train_denoise_model(base_dir, s2a_model,  experiment_config['diffusion_net'],  X_train, A_train,X_test, A_test)
        
        # For each model architecture, train A2T model
        model_name = experiment_config['model']

        net_config = experiment_config['net_config']['net']
        if 'task_type' in experiment_config:
            net_config['task_type'] = experiment_config['task_type']
        else:
            net_config['task_type'] = 'regression' 


        # Train the final model with raw signals    
        train_final_model(result_dir, model_name, net_config.copy(), 
                          experiment_config['learning_rate'], "raw_signals", dm, 
                          train_attributes, test_attributes, input_size, denoise_steps=0, noise_level=0.0)

        # For each noise rate, train a model with only noisy samples and 3 models with
        # 1, 2, and 3 denoising steps
        noise_rates = [float(v) for v in experiment_config['net_config']['noise_rates']]
        for noise_level in  noise_rates:
            # Train model with noise data augmentation
            train_final_model(result_dir, model_name, net_config.copy(), 
                              experiment_config['learning_rate'], "noise_aug", dm, 
                              train_attributes, test_attributes, input_size, denoise_steps=0, noise_level=noise_level)

            for denoise_steps in range(1, 4):
                # Train model with noise data augmentation
                train_final_model(result_dir, model_name, net_config.copy(), 
                                  experiment_config['learning_rate'], "diff_aug", dm, 
                                  train_attributes, test_attributes, input_size, denoise_steps=denoise_steps, noise_level=noise_level)


                                           
    train()
