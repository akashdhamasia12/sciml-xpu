#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# mnist_tf_keras.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Benchmark: mnist_tf_keras
Classifying MNIST using a CNN implemented with tf.keras
This is a single device training/inference example.
"""

# libs from sciml_bench
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.utils import list_files
from sciml_bench.core.tensorflow import LogEpochCallback
import warnings
import time
import h5py
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import skimage.io
import horovod
import horovod.tensorflow.keras as hvd
warnings.simplefilter(action='ignore', category=FutureWarning)

# Some utility codes
def load_dataset_mnist(file_path, batch_size):
    """
    Provides a data loader for training dataset
    """

    # generator
    def hdf5_generator(path, batch):
        with h5py.File(path, 'r') as h5_file:
            for i in range(0, h5_file['image'].shape[0], batch):
                # read, expand channel dim and normalize
                images = np.expand_dims(h5_file['image'][i:i + batch], -1) / 255
                # read and one-hot encoding
                labels = np.eye(10)[h5_file['label'][i:i + batch]]
                yield images, labels

    # load dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: hdf5_generator(file_path, batch_size),
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, 28, 28, 1), (None, 10)))

    # Dataset length
    len_dataset = len(list(hdf5_generator(file_path, batch_size)))

    return dataset, len_dataset

def get_inference_dataset(inference_path: Path, batch_size: int):
    """
    Load inference dataset in batches and return it along 
    with the number of files
    """
    inference_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=inference_path, labels='inferred', label_mode='categorical',
    class_names=None, color_mode='grayscale', batch_size=batch_size, image_size=(28,28),
    shuffle=False, seed=None, validation_split=None, subset=None,
    interpolation='bilinear', follow_links=False, smart_resize=False
    )

    n_elements = len(np.concatenate([i for x, i in inference_ds], axis=0))

    return inference_ds, n_elements


def create_model_mnist():
    """
    Creates a simple, yet effective CNN model
    """

    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Default learning rate in Adam
    learning_rate = 0.001

    # Scale the model to multiple GPUs if possible
    scaled_lr = learning_rate * hvd.size()
    opt = tf.keras.optimizers.Adam(scaled_lr)
    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)

    # Define horovod callbacks
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), hvd.callbacks.MetricAverageCallback(), hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1)]

    # compile model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'], experimental_run_tf_function=False)

    return model, callbacks


def load_images(image_dir_path):
    file_names =  list_files(image_dir_path)
    images = np.zeros((len(file_names), 28, 28, 1))
    for idx, url in enumerate(file_names):
        images[idx, :, :, 0] = skimage.io.imread(url)
    return images, file_names

# Training Code
def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the training routine to be called by SciML-Bench
    """

    # Horovod: initialize Horovod.
    hvd.init()

    params_out.activate(rank=hvd.rank(), local_rank=hvd.local_rank(), activate_log_on_host=False,
                        activate_log_on_device=False, console_on_screen=True)

    log = params_out.log

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('XPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'XPU')

    # We expect two benchmark-specific arguments here: 
    # batch_size and epochs. If not, we will assign 
    # default values.
    with log.subproc('Parsing input arguments'):
        # hyperparameters
        suggested_args = {
            'batch_size': 128,
            'epochs': 2,
            'num_warmup': 10
        } 

        args = params_in.bench_args.try_get_dict(default_args=suggested_args)
        batch_size = args['batch_size']
        epochs = args['epochs']
        num_warmup = args['num_warmup']
        log.message(f'batch_size = {batch_size}')
        log.message(f'epochs     = {epochs}')

    # Create CNN model & warmup
    with log.subproc('Create CNN Model & Warmup steps'):
        model, callbacks = create_model_mnist()
        input_size = (num_warmup * batch_size, 28, 28, 1)
        x = tf.random.normal(input_size)
        model.predict(x, verbose=0)

    log.begin('Running benchmark mnist_tf_keras on training mode')

    # Save training parameters
    with log.subproc('Writing the argument file'):
        args_file = params_in.output_dir / 'arguments_used.yml'
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # Create datasets
    with log.subproc('Creating datasets'):
        dataset_dir = params_in.dataset_dir
        train_set, train_size = load_dataset_mnist(dataset_dir / 'train.hdf5', batch_size)
        test_set, test_size = load_dataset_mnist(dataset_dir / 'test.hdf5', batch_size)
        log.message(f'Dataset directory: {dataset_dir}')

    # Define training params
    with log.subproc('Defining training params'):
        steps_per_epoch = train_size // hvd.size()
        validation_steps = test_size // hvd.size()
        validation_steps = 1 if batch_size > test_size else validation_steps
        verbose = 1 if hvd.rank() == 0 else 0

    # Train model
    log.begin('Training CNN model')
    with log.subproc('Running model.fit()'):
        start_time = time.time()
        params_out.system.stamp_event('model.fit')
        history = model.fit(train_set, epochs=epochs, steps_per_epoch=steps_per_epoch,
                            validation_data=test_set, validation_steps=validation_steps,
                            verbose=verbose, callbacks=callbacks)
        end_time = time.time()
        time_taken = end_time - start_time

    if hvd.rank() == 0:
        # Save model
        with log.subproc('Saving the model'):
            model_file = params_in.output_dir / 'mnist_tf_keras_model.h5'
            model.save(model_file)
            log.message(f'Saved to: {model_file}')

        # Save history
        with log.subproc('Saving training history'):
            history_file = params_in.output_dir / 'training_history.yml'
            with open(history_file, 'w') as handle:
                yaml.dump(history.history, handle)
            log.message(f'Saved to: {history_file}')
        log.ended('Training CNN model')

    # Predict
    with log.subproc('Making predictions on test set'):
        with h5py.File(dataset_dir / 'test.hdf5', 'r') as h5_file:
            # stamp model.predict in system monitor
            params_out.system.stamp_event('model.predict')
            pred = model.predict(np.expand_dims(h5_file['image'][:], -1) / 255)
            correct = np.sum(pred.argmax(axis=1) == h5_file['label'][:])

        accuracy = float(correct / len(pred) * 100)
        log.message(f'{correct} correct predictions for {len(pred)} images '
                    f'(accuracy: {accuracy:.2f}%)')

    # Save metrics
    metrics = dict(time=time_taken, accuracy=accuracy, loss=history.history['val_loss'][-1])
    metrics_file = params_in.output_dir / 'metrics.yml'

    if hvd.rank() == 0:
        with log.subproc('Saving inference metrics to a file'):
            with open(metrics_file, 'w') as handle:
                yaml.dump(metrics, handle)

    # end top level
    log.ended('Running benchmark mnist_tf_keras on training mode')


#####################################################################
# Inference mode                                                    #
#####################################################################

def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the inference routine to be called by SciML-Bench
    """

    default_args = {
        'use_gpu': True,
        "batch_size" : 64,
        "num_warmup": 10
    }

    params_out.activate(rank=0, local_rank=0)

    log = params_out.log

    # Parse input arguments
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    # Load the model and move it to the right device
    with log.subproc(f'Loading the model & Warm up for inference'):
        model = load_model(params_in.model)
        input_size = (args['num_warmup'] * args['batch_size'], 28, 28, 1)
        x = tf.random.normal(input_size)
        model.predict(x, verbose=0)

    log.begin('Running benchmark mnist_tf_keras on inference mode')

    # Parse input arguments
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    # Save inference parameters
    args_file = params_in.output_dir / 'inference_arguments_used.yml'
    with log.subproc('Saving inference arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)  

    # Create datasets
    with log.subproc(f'Setting up a data loader for inference'):
        inference_dataset, n_total_elements = get_inference_dataset(params_in.dataset_dir, args["batch_size"])

    # Perform bulk inference
    criterion = tf.keras.losses.CategoricalCrossentropy()
    with log.subproc(f'Doing inference'):
        start_time = time.time()
        total_loss = 0
        accuracy = tf.keras.metrics.CategoricalAccuracy()
        for inference_data, inference_label in inference_dataset: 
            outputs = model.predict(inference_data)
            accuracy.update_state(outputs, inference_label)
            loss = criterion(outputs, inference_label)
            total_loss += loss
        rate = accuracy.result().numpy() * 100
        total_loss = total_loss / n_total_elements
        end_time = time.time()
    time_taken = end_time - start_time

    throughput = math.floor(n_total_elements / time_taken)

    # Log outputs
    with log.subproc('Inference Performance'):
        log.message(f'Throughput  : {throughput} Images / sec')
        log.message(f'Overall Time: {time_taken:.4f} s')
        log.message(f'Correctness : {rate:.4f}%')
        log.message(f'Loss        : {total_loss:.4f}')

    
    # Save metrics
    metrics = dict(throughput=throughput, time=time_taken, accuracy=rate, loss=total_loss)
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics_file = params_in.output_dir / 'metrics.yml'
    with log.subproc('Saving inference metrics to a file'):
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  

    # End the top-level
    log.ended('Running benchmark mnist_tf_keras on inference mode')
