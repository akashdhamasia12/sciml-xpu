import yaml, os, h5py, time, random, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
import tensorflow as tf
import tensorflow.keras.backend as K
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sciml_bench.benchmarks.science.slstr_cloud.constants import PATCH_SIZE, N_CHANNELS, CROP_SIZE, IMAGE_H, IMAGE_W
from sciml_bench.benchmarks.science.slstr_cloud.model import unet
from sciml_bench.benchmarks.science.slstr_cloud.data_loader import SLSTRDataLoader, load_datasets
import math
import horovod
import horovod.tensorflow.keras as hvd
warnings.simplefilter(action='ignore', category=FutureWarning)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Loss function
def weighted_cross_entropy(beta):
    """
    Weighted Binary Cross Entropy implementation
    :param beta: beta weight to adjust relative importance of +/- label
    :return: weighted BCE loss
    """
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            logits=y_pred, labels=y_true, pos_weight=beta)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss

def reconstruct_from_patches(args, patches: tf.Tensor, nx: int, ny: int, patch_size: int) -> tf.Tensor:
    """Reconstruct a full image from a series of patches
    :param patches: array with shape (num patches, height, width)
    :param nx: the number of patches in the x direction
    :param ny: the number of patches in the y direction
    :param patch_size: the size of th patches
    :return: the reconstructed image with shape (1, height, weight, 1)
    """
    h = ny * patch_size
    w = nx * patch_size
    reconstructed = np.zeros((1, h, w, 1))

    for i in range(ny):
        for j in range(nx):
            py = i * patch_size
            px = j * patch_size
            reconstructed[0, py:py + patch_size, px:px + patch_size] = patches[0, i, j]

    # Crop off the additional padding
    offset_y = (h - IMAGE_H) // 2
    offset_x = (w - IMAGE_W) // 2
    reconstructed = tf.image.crop_to_bounding_box(reconstructed, offset_y, offset_x, IMAGE_H, IMAGE_W)
    return reconstructed

# Inference
def cloud_inference(args, model)-> None:
    print('Running benchmark slstr_cloud in inference mode.')

    # Read inference files
    inference_dir = args['dataset_dir']
    file_paths = list(Path(inference_dir).glob('**/S3A*.hdf'))

    # Create data loader in single image mode. This turns off shuffling and
    # only yields batches of images for a single image at a time so they can be
    # reconstructed.
    data_loader = SLSTRDataLoader(file_paths, single_image=True, crop_size=CROP_SIZE)
    dataset = data_loader.to_dataset()

    avg_accuracy = []
    avg_f1 = []
    avg_loss = []

    # Inference Loop
    for i, data in enumerate(dataset):
        patches, file_name = data
        print("="*50 + ">" + " Inference Batch {} ".format(i) + "<" + "="*50)
        file_name = Path(file_name.numpy().decode('utf-8'))

        # convert patches to a batch of patches
        n, ny, nx, _ = patches.shape
        patches = tf.reshape(patches, (n * nx * ny, PATCH_SIZE, PATCH_SIZE, N_CHANNELS))

        # perform inference on patches
        mask_patches = model.predict_on_batch(patches)

        # crop edge artifacts
        mask_patches = tf.image.crop_to_bounding_box(mask_patches, CROP_SIZE // 2, CROP_SIZE // 2, PATCH_SIZE - CROP_SIZE, PATCH_SIZE - CROP_SIZE)

        # reconstruct patches back to full size image
        mask_patches = tf.reshape(mask_patches, (n, ny, nx, PATCH_SIZE - CROP_SIZE, PATCH_SIZE - CROP_SIZE, 1))
        mask = reconstruct_from_patches(args, mask_patches, nx, ny, patch_size=PATCH_SIZE - CROP_SIZE)
        output_dir = args['output_dir']
        mask_name = output_dir / (file_name.name + '.h5')

        with h5py.File(file_name, 'r') as handle:
            ref_mask = handle['bayes'][:]
            ref_mask = ref_mask > .5

        mask = mask.numpy()
        mask = mask > .5

        accuracy = accuracy_score(mask.flatten(), ref_mask.flatten())
        loss = log_loss(mask.flatten(), ref_mask.flatten(),labels=["True","False"])
        f1 = f1_score(mask.flatten(), ref_mask.flatten())
        avg_accuracy.append(accuracy)
        avg_f1.append(f1)
        avg_loss.append(loss)
        print("\n Inference Batch {} statistics: Loss: {:.3f} - F1 Score: {:.3f} - Accuracy: {:.3f} \n".format(i, loss, f1, accuracy))

        with h5py.File(mask_name, 'w') as handle:
            handle.create_dataset('mask', data=mask)

    avg_accuracy = np.array(avg_accuracy).mean()
    avg_f1 = np.array(avg_f1).mean()
    avg_loss = np.array(avg_loss).mean()
    print("\n Average inference statistics: Loss: {:.3f} - F1 Score: {:.3f} - Accuracy: {:.3f} \n".format(avg_loss, avg_f1, avg_accuracy))


    # Return the number of inferences
    return len(file_paths), dict(accuracy=avg_accuracy, f1=avg_f1, loss=avg_loss)

#####################################################################
# Training mode                                                     #
#####################################################################

def cloud_training(args)-> None:

    tf.random.set_seed(args['seed'])
    data_dir = args['dataset_dir']

    # Running training on multiple GPUs
    scaled_lr = args['learning_rate'] * hvd.size()
    optimizer = tf.keras.optimizers.Adam(scaled_lr)
    # Horovod: add Horovod DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1, average_aggregated_gradients=True)

    # load the datasets
    train_dataset, test_dataset, train_data_count, test_data_count  = load_datasets(dataset_dir=data_dir, args=args)
    steps_per_epoch = math.floor(train_data_count / args['batch_size']) // hvd.size()
    validation_steps = math.floor(3 * test_data_count / args['batch_size']) // hvd.size()

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1)
    ]

    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    model = unet(input_shape=(PATCH_SIZE, PATCH_SIZE, N_CHANNELS))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', f1_m], experimental_run_tf_function=False)
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=args['epochs'], steps_per_epoch=steps_per_epoch, validation_steps=validation_steps , batch_size=args['batch_size'], callbacks=callbacks, verbose=verbose)


    # Save model
    if hvd.rank() == 0:
        model_file = args['output_dir'] / 'slstr_cloud_model.h5'
        model.save(model_file)

    return history


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance in the training mode.
    """
    default_args = {
        'use_gpu': True,
        "batch_size" : 32,
        "seed": 1234,
        "learning_rate": 0.001,
        "epochs": 30,
        "train_split": 0.8,
        "clip_offset": 15,
        "no_cache": False
    }

    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('XPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'XPU')

    params_out.activate(rank=0, local_rank=0)

    console = params_out.log.console

    log = params_out.log

    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)
        args['dataset_dir'] = params_in.dataset_dir / 'training'
        args['output_dir'] = params_in.output_dir

    with log.subproc('Warmup steps'):
        actual_epochs = args['epochs']
        args['epochs'] = 1
        history = cloud_training(args)

    if hvd.rank() == 0:
        console.begin('Running benchmark slstr_cloud in training mode')

    with log.subproc('Training the Model'):
        args['epochs'] = actual_epochs
        start = time.time()
        history = cloud_training(args)
        elapsed_time = time.time() - start
    
    # Save metrics
    metrics = dict(
        time=elapsed_time, 
        accuracy=history.history['accuracy'][-1], 
        f1=history.history['f1_m'][-1], 
        loss=history.history['val_loss'][-1]
    )

    metrics_file = params_in.output_dir / 'metrics.yml'

    if hvd.rank() == 0:
        with log.subproc('Saving inference metrics to a file'):
            with open(metrics_file, 'w') as handle:
                yaml.dump(metrics, handle)
        console.ended('Running benchmark slstr_cloud in training mode')

def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the inference routine to be called by SciML-Bench
    """
    default_args = {
        'use_gpu': True,
        "batch_size" : 32,
        "seed": 1234,
        "clip_offset": 15,
        "no_cache": False
    }

    params_out.activate(rank=0, local_rank=0)

    log = params_out.log

    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)
        args['dataset_dir'] = params_in.dataset_dir
        args['model_file'] = params_in.model
        args['output_dir'] = params_in.output_dir

    with log.subproc('Warmup steps inference'):
        # Load model
        modelPath = os.path.expanduser(args['model_file'])
        model = tf.keras.models.load_model(modelPath, custom_objects={"f1_m": f1_m })
        input_shape = (63, PATCH_SIZE, PATCH_SIZE, N_CHANNELS)
        x = tf.random.normal(input_shape)
        for _ in range(10):
            model.predict_on_batch(x)
            

    console = params_out.log.console
    console.begin('Running benchmark slstr_cloud in inference mode')

    with log.subproc('Running inference'):
        start = time.time()
        number_inferences, metrics = cloud_inference(args, model)
        elapsed_time = time.time() - start
        throughput = elapsed_time/number_inferences

    metrics['time'] = elapsed_time
    metrics['throughput'] = throughput
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics_file = params_in.output_dir / 'metrics.yml'
    with log.subproc('Saving inference metrics to a file'):
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  

    console.ended('Running benchmark slstr_cloud in inference mode')
