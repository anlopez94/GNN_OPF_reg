


from tensorflow import keras
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(run_eagerly=True)
from utils.gnn import GNN

import os
import glob
import logging
import pickle

def evaluate_model(model, Data):

    test_dataset = tf.data.Dataset.from_tensor_slices((Data.x_test, Data.y_test))
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(len(test_dataset))
    for _, (x_test, y_test) in enumerate(test_dataset):
        y_val = tf.cast(y_test, dtype=tf.float32)
        logits = tf.map_fn(lambda x: model(x), x_test, fn_output_signature=tf.float32)
        test_loss = tf.sqrt(tf.reduce_mean(tf.square(y_val - logits)))
    return [test_loss.numpy(), y_test, logits]

def eval_step(writer, epoch, train_loss, val_loss):

    with writer.as_default():
        with tf.name_scope('Eval'):
            if train_loss is not None:
                tf.summary.scalar("train_loss", train_loss, step=epoch)
            if val_loss is not None:
                tf.summary.scalar("val_loss", val_loss, step=epoch)



def train_model(model, Data, loss_fn, optimizer, epochs, batch_size, batch_size_dos, writer, early_stop, checkpoint_path):
    losses = []
    if early_stop:
        # If there is early stopping, initialize class in "model_training"
        early_stopping = EarlyStopping(checkpoint_path, patience=30)
    else:
        early_stopping = None

    for epoch in range(epochs):

        print("\nStart of epoch %d" % (epoch))
        val_dataset = tf.data.Dataset.from_tensor_slices((Data.x_val, Data.y_val))
        val_dataset = val_dataset.shuffle(buffer_size=1024).batch(len(val_dataset))
        train_dataset = tf.data.Dataset.from_tensor_slices((Data.x_train, Data.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size_dos)

        # Iterate over the batches of the dataset.
        loss_value_batch = []
        # for batch_size_dos in range(batch_size_dos):
        #     loss_value_batch = []
        for step, (x_train, y_train) in enumerate(train_dataset):
        # step = 0
        # for start in range(0, len(Data.x_train), batch_size_dos):
        #     step += 1
        #     end = start + batch_size_dos
        #     if end > len(Data.x_train):
        #         end = len(Data.x_train)
        #     x_train = Data.x_train[start:end]
        #     y_train = Data.y_train[start:end]

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                y_train = tf.cast(y_train, dtype=tf.float32)
                logits = tf.map_fn(lambda x: model(x, training=True), x_train, fn_output_signature=tf.float32)
                mean_loss = tf.square(y_train - logits)
                mean_loss = tf.sqrt(tf.reduce_mean(mean_loss))


            # loss_value_batch = tf.constant(loss_value_batch, dtype=tf.float32)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            # tape.watch(model.trainable_weights)
            grads = tape.gradient(mean_loss, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            #
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, mean_loss.numpy())
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size_dos))

        # Run a validation loop at the end of each epoch.
        for _, (x_val, y_val) in enumerate(val_dataset):
        # x_val = Data.x_val
        # y_val = Data.y_val
            y_val = tf.cast(y_val, dtype=tf.float32)
            logits = tf.map_fn(lambda x: model(x, training=True), x_val, fn_output_signature=tf.float32)
            val_loss = tf.square(y_val - logits)
            val_loss = tf.sqrt(tf.reduce_mean(val_loss))
        print(f'val loss {val_loss.numpy()}')

        if early_stop:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break

        # eval_step(writer, epoch, mean_loss, val_loss)
        losses.append([mean_loss.numpy(), val_loss.numpy()])
        # writer.flush()
        if epoch % 10 == 0:
            results_train = {'losses': losses,  'metric': 'mse'}
            save_results(checkpoint_path, 'results_train', results_train)
    return losses, model


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, model_path, patience=30, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.model_path = 'models/' + model_path
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            tf.config.run_functions_eagerly(True)
            model.save(self.model_path, save_format="tf")
        elif self.best_loss - val_loss > self.min_delta:
            self.counter = 0
            self.best_loss = val_loss
            tf.config.run_functions_eagerly(True)
            model.save(self.model_path, save_format="tf")
            print('SAVING MODEL')
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True



def load_model(model_path):
    model_path = 'models/' + model_path
    model = tf.keras.models.load_model(model_path)
    return model

def set_experiment_identifier(base_dir='logs', case_name='case9', batch=100, epochs=100, optimizer='ADAM'):
    case_folder = 'case_name' + str(case_name)

    batch = 'batch' + str(batch)
    epochs = 'epochs' + str(epochs)
    optimizer = 'optimizer' + str(optimizer)

    model_folder = ('-').join([batch, epochs, optimizer])
    experiment_identifier = os.path.join(case_folder, model_folder)
    writer_dir = os.path.join(base_dir, experiment_identifier)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    else:
        for f in glob.glob(os.path.join(writer_dir, 'events.out.tfevents.*')):
            os.remove(f)
    return writer_dir

def set_writer_and_checkpoint_dir(base_dir = 'logs', case_name='case9', batch=100, epochs=100, optimizer='ADAM'):
    writer_dir = set_experiment_identifier(base_dir, case_name, batch, epochs, optimizer)
    checkpoint_dir = writer_dir
    writer = tf.summary.create_file_writer(writer_dir)
    return writer




def save_results(path, name, results):

    path = 'results' + '/' + path + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    name = path + name + '.pkl'
    with open(name, 'wb') as f:
        pickle.dump(results, f)