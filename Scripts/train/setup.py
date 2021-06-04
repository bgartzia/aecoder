"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import os
import time
import numpy as np
import tensorflow as tf
if tf.test.is_built_with_cuda() and tf.test.is_gpu_available():
    tf.config.gpu.set_per_process_memory_growth(True)
from Models import AutoEncoder
from Data import load_data
import train_engine
from random import seed as base_seed


def train(config):
    np.random.seed(config['train.seed'])
    tf.random.set_seed(config['train.seed'])
    base_seed(config['train.seed'])

    ret = load_data(config)
    train_pipe = ret['train']
    val_pipe = ret['val']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    # Setup training operations
    model = AutoEncoder(config['data.shape'], config['model.layers'],
                        config['model.latent_dim'])
    optimizer = tf.keras.optimizers.Adam(config['train.lr'])

    # Metrics to gather
    train_loss = tf.metrics.Mean(name='train_loss')
    val_loss = tf.metrics.Mean(name='val_loss')
    val_losses = []

    @tf.function
    def train_step(loss_func, batch):
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss_out = loss_func(model(batch))
        gradients = tape.gradient(loss_out, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        # Log loss and accuracy for step
        train_loss(loss)

    @tf.function
    def val_step(loss_func, batch):
        loss_out = loss_func(model(batch))
        val_loss(loss_out)

    # Create empty training engine
    train_engine = TrainEngine()

    # Set hooks on training engine
    def on_start(state):
        print("\n############################################")
        print("\t\tTraining started.")
        print("############################################\n")
    train_engine.hooks['on_start'] = on_start

    def on_end(state):
        print("\n############################################")
        print("\t\tTraining ended.")
        print("############################################\n")
    train_engine.hooks['on_end'] = on_end

    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
        train_loss.reset_states()
        val_loss.reset_states()
    train_engine.hooks['on_start_epoch'] = on_start_epoch

    def on_end_epoch(state):
        print(f"Epoch {state['epoch']} ended.")
        epoch = state['epoch']
        template = 'Epoch {}, Loss: {}, Val Loss: {}'
        print(
              template.format(epoch + 1, train_loss.result(), val_loss.result())
             )

        cur_loss = val_loss.result().numpy()
        if cur_loss < state['best_val_loss']:
            print("Saving new best model with loss: ", cur_loss)
            state['best_val_loss'] = cur_loss
            model.save(config['model.save_path'])
        val_losses.append(cur_loss)

        # Early stopping
        patience = config['train.patience']
        if len(val_losses) > patience \
                and max(val_losses[-patience:]) == val_losses[-1]:
            state['early_stopping_triggered'] = True
    train_engine.hooks['on_end_epoch'] = on_end_epoch

    def on_batch(state):
        print(f"Batch {state['total_batches']}")
        batch = state['sample']
        loss_func = state['loss_func']
        train_step(loss_func, batch)
    train_engine.hooks['on_batch'] = on_batch

    def on_batch_end(state):
        # Validation
        val_pipeline = state['val_pipeline']
        loss_func = state['loss_func']
        for batch in val_pipeline:
            val_step(loss_func, batch)
    train_engine.hooks['on_batch_end'] = on_batch_end

    time_start = time.time()
    with tf.device(device_name):
        train_engine.train(
            # Change loss function here
            loss_func=tf.nn.l2_loss,
            train_pipeline=train_pipe,
            val_pipeline=val_pipe,
            epochs=config['train.epochs'],
            n_episodes=config['data.episodes'])
    time_end = time.time()

    elapsed = time_end - time_start
    h, min = elapsed//3600, elapsed%3600//60
    sec = elapsed-min*60
    print(f"Training took: {h} h {min} min {sec} sec")

