

from tensorflow.keras import layers
import pandas as pd
import os
from tensorflow import keras
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf




class GNN(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        hidden_units_update='256-128-32',
        hidden_units_readout='64-32',
        hidden_units_message='128-32',
        dropout_rate=0,
        dropout_rate_readout=0.2,
        normalize=True,
        node_state_size=32,
        message_iterations=3,
        activation_fn='tanh',
        final_activation_fn='linear',
        create_message_nn=False,
        moving_edges=False,
        *args,
        **kwargs,
    ):
        super(GNN, self).__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights, generators_nodes = graph_info
        self.edges = edges
        self.moving_edges = moving_edges
        self.create_message_nn = create_message_nn
        self.dropout_rate = dropout_rate
        self.dropout_rate_readout = dropout_rate_readout
        self.num_nodes = node_features.shape[0]
        self.num_features = node_features.shape[1]
        self.num_features_edge = edge_weights.shape[1]
        self.edge_weights = edge_weights
        self.generators_nodes = generators_nodes
        self.normalize = normalize
        self.node_state_size = node_state_size
        self.message_iterations = message_iterations
        self.hidden_units_readout = hidden_units_readout.split("-")
        self.hidden_units_update = hidden_units_update.split("-")
        self.hidden_units_message = hidden_units_message.split("-")
        self.activation_fn = activation_fn
        # if (activation_fn == 'tanh'):
        #     self.activation_fn = tf.nn.tanh
        self.final_activation_fn = final_activation_fn
        self.hidden_layer_initializer = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
        self.final_layer_initializer = tf.keras.initializers.Orthogonal(gain=1)
        self.kernel_regularizer = None  # keras.regularizers.l2(0.01)

        # self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        # Create a compute logits layer.
        fnn_layers = self.create_ffn(self.hidden_units_readout, self.dropout_rate)
        #ultima capa de la readout
        fnn_layers.append(layers.Dropout(self.dropout_rate_readout))
        fnn_layers.append(
            layers.Dense(units=1, activation=self.final_activation_fn, kernel_initializer=self.final_layer_initializer,
                         kernel_regularizer=self.kernel_regularizer, name="readout"))
        self.readout = keras.Sequential(fnn_layers, name='readout')
        self.normalize = normalize
        # self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        fnn_layers = self.create_ffn(self.hidden_units_update, self.dropout_rate)
        self.update_fn = keras.Sequential(fnn_layers, name='update_fn')
        if self.create_message_nn:
            fnn_layers = self.create_ffn(self.hidden_units_message, self.dropout_rate)
            self.create_message = keras.Sequential(fnn_layers, name='message_fn')


    def aggregate(self, node_indices, neighbour_messages):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        num_nodes = tf.math.reduce_max(node_indices) + 1
        num_nodes = self.num_nodes

        aggregated_message_mean = tf.math.unsorted_segment_mean(
            neighbour_messages, node_indices, num_segments=num_nodes
        )
        aggregated_message_max = tf.math.unsorted_segment_max(
            neighbour_messages, node_indices, num_segments=num_nodes
        )
        aggregated_message_min = tf.math.unsorted_segment_min(
            neighbour_messages, node_indices, num_segments=num_nodes
        )
        aggregated_message = tf.concat([aggregated_message_mean, aggregated_message_max, aggregated_message_min], axis=1)
        return aggregated_message


    def update(self, node_repesentations, aggregated_messages):
        h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings


    def prepare(self, nodes_features, edges, edge_weights, num_nodes):
        node_indices, neighbour_indices = edges[0], edges[1]
        weights_mean = tf.math.unsorted_segment_mean(
            edge_weights, node_indices, num_segments=num_nodes
        )
        weights_sum = tf.math.unsorted_segment_sum(
            edge_weights, node_indices, num_segments=num_nodes
        )
        weights_min = tf.math.unsorted_segment_min(
            edge_weights, node_indices, num_segments=num_nodes
        )
        weights_max = tf.math.unsorted_segment_max(
            edge_weights, node_indices, num_segments=num_nodes
        )

        node_repesentations = tf.concat([nodes_features, weights_mean, weights_max, weights_min, weights_sum], axis=-1)
        # node_repesentations = tf.concat([weights_mean, weights_max, weights_min, weights_sum], axis=-1)
        # node_repesentations = tf.concat([nodes_features, node_repesentations], axis=-1)
        padding = [[0, 0], [0, self.node_state_size - node_repesentations.shape[1]]]
        node_repesentations = tf.pad(node_repesentations, padding)

        return node_repesentations


    def build(self):
        self.update_fn.build(input_shape=[None, 4 * self.node_state_size])
        self.readout.build(input_shape=[None, self.node_state_size])
        if self.create_message_nn:
            self.create_message.build(input_shape=[None, 2 * self.node_state_size + self.num_features_edge])
        self.built = True

    def create_ffn(self, hidden_units, dropout_rate):
        fnn_layers = []
        for units in hidden_units:

            initializer = self.hidden_layer_initializer
            # fnn_layers.append(layers.BatchNormalization())
            if dropout_rate > 0:
                fnn_layers.append(layers.Dropout(dropout_rate))
            fnn_layers.append(
                layers.Dense(units, activation=self.activation_fn, kernel_initializer=initializer,
                             kernel_regularizer=self.kernel_regularizer))

        return fnn_layers


    def prepare_message(self, node_repesentations):
        node_indices, neighbour_indices = self.edges[0], self.edges[1]
        neighbour_representation_link = tf.gather(node_repesentations, neighbour_indices)
        node_representation_link = tf.gather(node_repesentations, node_indices)
        message_pre = tf.concat([neighbour_representation_link, self.edge_weights, node_representation_link], axis=1)
        return message_pre

    def call(self, input):
        # Preprocess the node_features to produce node representations.
        if self.moving_edges:
            input_tensor, edges = input
            #We use each time new edges depending on the instance
            self.edges = edges
        else:
            input_tensor = input
        input_tensor = tf.convert_to_tensor(input_tensor)
        node_features = tf.reshape(input_tensor, [self.num_nodes, self.num_features])
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = self.edges[0], self.edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        node_repesentations = self.prepare(node_features, self.edges, self.edge_weights, self.num_nodes)
        for _ in range(self.message_iterations):
            if self.create_message_nn:
                message_pre = self.prepare_message(node_repesentations)
                messages = self.create_message(message_pre)
            else:
                messages = tf.gather(node_repesentations, neighbour_indices)

            # Aggregate the neighbour messages.
            aggregated_messages = self.aggregate(node_indices, messages)
            # Update the node embedding with the neighbour messages.
            node_repesentations = self.update(node_repesentations, aggregated_messages)
        node_repesentations = tf.where(tf.math.is_nan(node_repesentations), tf.zeros_like(node_repesentations), node_repesentations)
        # Skip connection.
        # Postprocess node embedding.
        # node_repesentations = self.postprocess(node_repesentations)
        # Fetch node embeddings for the input node_indices.
        self.generators_nodes = tf.dtypes.cast(self.generators_nodes, tf.int32)
        node_repesentations = tf.gather(node_repesentations, self.generators_nodes)
        # Compute logits
        # node_repesentations = tf.Tensor([self.generators_nodes])
        values = self.readout(node_repesentations)
        return tf.reshape(values, [-1])


