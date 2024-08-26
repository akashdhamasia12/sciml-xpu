#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# optical_damage.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import warnings
import math
import gc
import resource
import time
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sciml_bench.core.tensorflow import LogEpochCallback
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.utils import MultiLevelLogger
from sciml_bench.benchmarks.science.optical_damage.model import autoencoder
from sciml_bench.benchmarks.science.optical_damage.utils import IMAGE_SHAPE, load_images
import horovod
import horovod.tensorflow.keras as hvd
warnings.simplefilter(action='ignore', category=FutureWarning)


#####################################################################
# Training mode                                                     #
#####################################################################
# For training use this command:
# sciml-bench run --mode training optical_damage

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the training routine to be called by SciML-Bench
    """
    default_args = {
        'batch_size': 32,
        'epochs': 5,
        'latent_size': 512,
        'lr': 0.001,
        'use_gpu': True
    }

    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('XPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'XPU')

    # No distributed training in this one
    params_out.activate(rank=0, local_rank=0)
    log = params_out.log.console

    # Parse input arguments against default ones
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    latent_dim = args['latent_size']
    learning_rate = args['lr']
    batch_size = args['batch_size']

    # Scale the model to multiple GPUs if possible
    scaled_lr = learning_rate * hvd.size()
    opt = tf.keras.optimizers.Adam(scaled_lr)
    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)

    with log.subproc('Warmup steps'):
        # Loading model
        model = autoencoder(IMAGE_SHAPE, latent_dim)
        model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'], experimental_run_tf_function=False)
        input_size = (128, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])
        x = tf.random.normal(input_size)
        for _ in range(10):
            model.predict(x, verbose=0)

    # Log top level process
    if hvd.rank() == 0:
        log.begin(f'Running benchmark optical_damage on training mode')

    # Save parameters
    args_file = params_in.output_dir / 'training_arguments_used.yml'
    with log.subproc('Saving arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # Decide which device to use
    basePath = params_in.dataset_dir
    modelPath = params_in.output_dir / f'opticsModel.h5'
   
    # Load training images
    with log.subproc('Loading training data'):
        training_path = basePath / 'training/undamaged'
        training_images = load_images(training_path)
        log.message(f'Training images: {len(training_images)}')

    # Load validation images
    with log.subproc('Loading validation data'):
        validation_path = basePath / 'validation/undamaged'
        validation_images = load_images(validation_path)
        log.message(f'Validation images: {len(validation_images)}')

    # Prepare training and validation data
    train_data = tf.data.Dataset.from_tensor_slices((training_images, training_images))
    train_data = train_data.shuffle(2000)
    train_data = train_data.repeat(args['epochs']).batch(batch_size)

    validation_data = tf.data.Dataset.from_tensor_slices((validation_images, validation_images))
    validation_data = validation_data.batch(batch_size)

    # Define horovod callbacks
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), hvd.callbacks.MetricAverageCallback(), hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1)]

    # Define steps_per_epoch and validation_steps
    steps_per_epoch = math.floor(len(training_images) / batch_size) // hvd.size()
    validation_steps = math.floor(len(validation_images) / batch_size) // hvd.size()
    verbose = 1 if hvd.rank() == 0 else 0

    with log.subproc('Fitting model parameters'):
        start_time = time.time()
        history = model.fit(train_data, validation_data=validation_data, epochs=args['epochs'], steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps, verbose=verbose, callbacks=callbacks)
        end_time = time.time()
        time_taken = end_time - start_time

    with log.subproc('Predicting on validation data'):
        predictions = model.predict(validation_data, verbose=0)
        mse = ((predictions - validation_images)**2).reshape(len(validation_images), -1).mean(-1).mean()

    # Free up memory 
    del training_images
    del validation_images
    gc.collect()

    # Save model
    if hvd.rank() == 0:
        with log.subproc('Saving model file'):
            tf.keras.models.save_model(model, modelPath)
            log.message(f'model saved in:{str(modelPath)}')

    # Peak memory usage
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    log.message(f'memory_usage in Gbytes: {(memory_usage/(1024*1024)):.2f}')

    # Save history
    if hvd.rank() == 0:
        with log.subproc('Saving training history'):
            history_file = params_in.output_dir / 'training_history.yml'
            with open(history_file, 'w') as handle:
                yaml.dump(history.history, handle)

    # Save metrics
    metrics = dict(mse=float(mse), time=time_taken, loss=mse)
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics_file = params_in.output_dir / 'metrics.yml'

    if hvd.rank() == 0:
        with log.subproc('Saving inference metrics to a file'):
            with open(metrics_file, 'w') as handle:
                yaml.dump(metrics, handle)  

    # End top level
    if hvd.rank() == 0:
        log.ended(f'Running benchmark optical_damage on training mode')
    

#####################################################################
# Inference mode                                                    #
#####################################################################
# For inference use this command:
# sciml-bench run --mode inference --model ~/sciml_bench/models/optical_damage/opticsModel.h5 --dataset_dir ~/sciml_bench/datasets/optical_damage_ds1 optical_damage

def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the inference routine to be called by SciML-Bench
    """

    default_args = {
        'use_gpu': True,
        "batch_size" : 64
    }
   
    params_out.activate(rank=0, local_rank=0)
    log = params_out.log

    with log.subproc('Warmup steps'):
        # Loading model
        modelPath = params_in.model
        model = tf.keras.models.load_model(modelPath)
        input_size = (64, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])
        x = tf.random.normal(input_size)
        for _ in range(10):
            model.predict(x, verbose=0)

    log.begin('Running benchmark optical_damage on inference mode')

    # Parse input arguments
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    # Save inference parameters
    args_file = params_in.output_dir / 'inference_arguments_used.yml'
    with log.subproc('Saving inference arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    basePath = params_in.dataset_dir

    # The inference folder contains: 4499 undamaged and 2047 damaged images
    with log.subproc('Load damaged images'):
        damaged_path = basePath / 'damaged'
        damaged_images = load_images(damaged_path)
        log.message(f'Number of damaged images: {len(damaged_images)}')
  
    with log.subproc('Load undamaged images'):
        undamaged_path = basePath / 'undamaged'
        undamaged_images = load_images(undamaged_path)
        log.message(f'Number of damaged images: {len(undamaged_images)}')

    # Concatenate two arrays
    imgs = np.concatenate((damaged_images, undamaged_images))

    batch_size = args['batch_size']
    dataset = tf.data.Dataset.from_tensor_slices(imgs)
    dataset = dataset.batch(batch_size)

    # Make labels for each image
    test_labels = np.zeros((len(imgs)))
    test_labels[:len(damaged_images)] = 1

    log.message(f'Total number of images use for inferencing: {len(imgs)}') 

    startInference = time.time()

    with log.subproc('Start inference'):
        recons = model.predict(dataset)

    endInference = time.time()

    time_taken = (endInference - startInference)
    throughput = len(imgs)/(endInference - startInference)

    log.message(f'Inference time: {time_taken: 9.4f}, ' \
                f'Images/s: {throughput:9.1f}')

    squared_error = (imgs - recons)**2.
    # Calculating stats

    recons = np.squeeze(recons)
    imgs = np.squeeze(imgs)
    squared_error = np.squeeze(squared_error)

    with log.subproc('Calculating inference statistics'):
        score_999 = np.any(squared_error > np.percentile(squared_error, 99.9), axis=(1, 2))
        score_995 = np.any(squared_error > np.percentile(squared_error, 99.5), axis=(1, 2))
        score_990 = np.any(squared_error > np.percentile(squared_error, 99.0), axis=(1, 2))

        acc_999 = accuracy_score(score_999, test_labels)
        acc_995 = accuracy_score(score_995, test_labels)
        acc_990 = accuracy_score(score_990, test_labels)

        f1_999 = f1_score(score_999, test_labels)
        f1_995 = f1_score(score_995, test_labels)
        f1_990 = f1_score(score_990, test_labels)

        mse = squared_error.reshape(len(squared_error), -1).mean(-1).mean()

        log.message(f'Accuracy at 99.0%: {acc_990: .2f}, 99.5%: {acc_995:.2f}, 99.9%: {acc_999:.2f}')
        log.message(f'F1 Score at 99.0%: {f1_990: .2f}, 99.5%: {f1_995:.2f}, 99.9%: {f1_999:.2f}')

    # Save metrics
    metrics = dict(time=time_taken, throughput=throughput, accuracy=acc_999 * 100, f1=f1_999, mse=mse, loss=mse)
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics_file = params_in.output_dir / 'metrics.yml'
    with log.subproc('Saving inference metrics to a file'):
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  

    # Max memory usage
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    log.message(f'memory_usage in Gbytes: {memory_usage/(1024*1024): 9.4f}')

    # Saving images, reconstructions and mse
    with log.subproc('Saving images, reconstructions and mse'):
        output_file = params_in.output_dir / 'inference.h5'
        with h5py.File(output_file, 'w') as handle:
            handle.create_dataset('reconstruction', data=recons)
            handle.create_dataset('images', data=imgs)
            handle.create_dataset('mse', data=mse)
        log.message(f"Reconstructions saved to: {output_file}")


    # End top level
    log.ended('Running benchmark optical_damage on inference mode')
