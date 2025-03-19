# fx_classify_custom_train_funcs.py


import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, GlobalAveragePooling1D, LayerNormalization, 
    MultiHeadAttention, Add, Flatten, TimeDistributed, LSTM)
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO


def custom_loss(y_true, y_pred, max_loss=5, 
                penalty_weight=0.01):
    """
    Custom loss function that combines sparse categorical cross-entropy with three penalties:
      1. A penalty for false positives on the minority classes (encoded as 1 and 2).
      2. A **per-sample** penalty for deviation of the predicted probability from the true distribution.
      3. A penalty for imbalance between the predicted probabilities for the minority classes.
    
    Args:
        y_true: Tensor of true labels (sparse integers) with shape (batch_size, 1) or (batch_size,).
        y_pred: Tensor of predicted probabilities (softmax output) with shape (batch_size, num_classes).
        max_loss: Maximum loss value for clipping.
        penalty_weight: Weight for the false positive penalty on minority classes.
        penalty_weight_distribution: Weight for the per-sample distribution penalty.
        penalty_weight_minority_balance: Weight for the minority balance penalty.
        
    Returns:
        A tensor representing the loss for each sample.
    """
    # Minority classes are encoded as 1 and 2.
    minority_classes = [1, 2]
    epsilon = tf.keras.backend.epsilon()

    # Ensure y_pred is float32 and clip predictions to avoid log(0)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Calculate sparse categorical cross-entropy loss.
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    ce_loss = tf.clip_by_value(ce_loss, 0.00001, max_loss)

    # Determine predicted classes using argmax
    y_pred_classes = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    y_true_classes = tf.squeeze(y_true, axis=-1) if len(y_true.shape) > 1 else tf.cast(y_true, tf.int32)

    # --- Minority Class Penalty ---
    is_minority_class_pred = tf.reduce_any(
        tf.equal(tf.expand_dims(y_pred_classes, axis=-1), minority_classes),
        axis=-1
    )
    is_minority_class_true = tf.reduce_any(
        tf.equal(tf.expand_dims(y_true_classes, axis=-1), minority_classes),
        axis=-1
    )
    false_positive_mask = tf.logical_and(is_minority_class_pred, tf.logical_not(is_minority_class_true))
    penalty_minority = tf.cast(false_positive_mask, tf.float32) * penalty_weight
    penalty_minority = tf.clip_by_value(penalty_minority, 0, max_loss)

    # --- Total Loss ---
    total_loss = ce_loss + penalty_minority
    total_loss = tf.clip_by_value(total_loss, 0, max_loss)
    
    return total_loss


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', patience=10):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.best = np.Inf
        self.wait = 0

    def on_aggregated_epoch_end(self, epoch, logs):
        current = logs.get(self.monitor)
        if current is None:
            print(f"[EarlyStopping] '{self.monitor}' not found in logs.")
            return
        if current < self.best:
            self.best = current
            self.wait = 0
            print(f"[EarlyStopping] Epoch {epoch+1}: {self.monitor} improved to {current:.4f}.")
        else:
            self.wait += 1
            print(f"[EarlyStopping] Epoch {epoch+1}: {self.monitor} did not improve (wait {self.wait}/{self.patience}).")
            if self.wait >= self.patience:
                print(f"[EarlyStopping] Early stopping triggered at epoch {epoch+1}.")
                self.model.stop_training = True


class CustomReduceLROnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = np.Inf
        self.wait = 0

    def on_aggregated_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        current = logs.get(self.monitor)
        if current is None:
            print(f"[ReduceLROnPlateau] '{self.monitor}' not found in logs.")
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Get current learning rate
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                
                # Compute the new learning rate based on the last updated learning rate
                new_lr = max(old_lr * self.factor, self.min_lr)

                # Set new learning rate
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

                print(f"[ReduceLROnPlateau] Epoch {epoch+1}: LR reduced from {old_lr:.6e} to {new_lr:.6e}.")
                
                # Reset patience counter
                self.wait = 0


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.best = np.Inf

    def on_aggregated_epoch_end(self, epoch, logs):
        current = logs.get(self.monitor)
        if current is None:
            print(f"[ModelCheckpoint] '{self.monitor}' not found in logs.")
            return
        if current < self.best:
            self.best = current
            self.model.save_weights(self.filepath)
            print(f"[ModelCheckpoint] Epoch {epoch+1}: Checkpoint saved to {self.filepath} with {self.monitor} = {current:.4f}.")


class ConfusionMatrixCallback:
    def __init__(self, log_dir, class_names):
        self.log_dir = log_dir
        self.class_names = class_names
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, "confusion_matrix"))

    def log_conf_matrix(self, epoch, all_preds, all_labels):
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(self.class_names)))
        cm_image = self._plot_confusion_matrix(cm)
        with self.file_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        print(f"[TensorBoard] Confusion Matrix logged for epoch {epoch+1}.")
        
    def plot_to_image(self, figure):
        # Save the plot to a PNG in memory.
        buf = BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def _plot_confusion_matrix(self, cm):
        size = len(self.class_names)
        figure = plt.figure(figsize=(size, size))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        
        indices = np.arange(len(self.class_names))       
        plt.xticks(indices, self.class_names)
        plt.yticks(indices, self.class_names)
        
        # Normalize Confusion Matrix
        cm = np.around(
            cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=3
            )
        
        # Use the "Blues" colormap to determine the background color of each cell.
        cmap = plt.get_cmap("Blues")
        for i in range(size):
            for j in range(size):
                cell_value = cm[i, j]
                rgba = cmap(cell_value)
                # Compute luminance using a standard formula.
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                # Use white text if the cell is dark, black otherwise.
                color = "white" if luminance < 0.5 else "black"
                plt.text(i, j, cell_value, horizontalalignment="center", color=color)
        
        plt.tight_layout()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
        # Convert the figure to a TensorFlow image.
        cm_image = self.plot_to_image(figure)
        return cm_image
        

# --------------------------
# AggregateCallbacks: container for all aggregated callbacks
# --------------------------

class AggregateCallbacks:
    def __init__(self, monitor_metric='val_loss', patience=10, save_dir=None,
                 use_early_stopping=True, use_reduce_lr=True, use_model_checkpoint=False,
                 use_tensorboard=True, use_csv_logger=True, log_dir=None, use_conf_matrix=True, 
                 class_names=[0,1,2]):
        """
        Instantiate only the callbacks you want to use. If a particular callback is not desired,
        set its corresponding use_* flag to False.
        """
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.use_tensorboard = use_tensorboard
        self.use_conf_matrix = use_conf_matrix

        # EarlyStopping
        self.early_stopping = CustomEarlyStopping(monitor=monitor_metric, patience=patience) if use_early_stopping else None
        
        # ReduceLROnPlateau
        self.reduce_lr = CustomReduceLROnPlateau(monitor=monitor_metric, factor=0.8, patience=5, min_lr=1e-6) if use_reduce_lr else None
        
        # ModelCheckpoint (requires a valid save_dir)
        if use_model_checkpoint:
            if save_dir is None:
                raise ValueError("save_dir must be provided for model checkpointing.")
            checkpoint_filepath = os.path.join(save_dir, "best_model.h5")
            self.model_checkpoint = CustomModelCheckpoint(filepath=checkpoint_filepath, monitor=monitor_metric)
        else:
            self.model_checkpoint = None

        # CSV logging: write aggregated metrics to a CSV file.
        if use_csv_logger:
            if save_dir is None:
                raise ValueError("save_dir must be provided for CSV logging.")
            self.csv_log_path = os.path.join(save_dir, "aggregated_training_log.csv")
            # Write header if file doesn't exist.
            if not os.path.exists(self.csv_log_path):
                with open(self.csv_log_path, 'w') as f:
                    f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy\n")
        else:
            self.csv_log_path = None
            
        # Confusion Matrix Callback (Validation Only)
        self.conf_matrix_callback = None
        if use_conf_matrix and log_dir is not None:
            val_log_dir = os.path.join(log_dir, "val")
            self.conf_matrix_callback = ConfusionMatrixCallback(val_log_dir, class_names)

    def on_aggregated_epoch_end(self, epoch, logs, model):
        """
        This method should be called at the end of an outer epoch with aggregated metrics.
        It will invoke each aggregated callback that was configured.
        """
        # Make sure the model is assigned to each callback that uses it.
        if self.early_stopping is not None:
            self.early_stopping.model = model
            self.early_stopping.on_aggregated_epoch_end(epoch, logs)
        if self.reduce_lr is not None:
            self.reduce_lr.model = model
            self.reduce_lr.on_aggregated_epoch_end(epoch, logs)
        if self.model_checkpoint is not None:
            self.model_checkpoint.model = model
            self.model_checkpoint.on_aggregated_epoch_end(epoch, logs)
        if self.csv_log_path is not None:
            import csv
            with open(self.csv_log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, logs.get('train_loss', 0),
                                 logs.get('train_accuracy', 0),
                                 logs.get('val_loss', 0),
                                 logs.get('val_accuracy', 0)])
            print(f"[CSVLogger] Epoch {epoch+1}: Metrics appended to {self.csv_log_path}.")
    
    
# --------------------------
# Model architecture
# --------------------------


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    """
    A single transformer encoder block.
    
    Args:
        inputs (tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        head_size (int): Dimensionality of the query/key vectors in MultiHeadAttention.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward layer.
        dropout (float): Dropout rate.
        
    Returns:
        Tensor of the same shape as inputs.
    """
    # Layer normalization and multi-head self-attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    attn_output = Dropout(dropout)(attn_output)
    x = Add()([inputs, attn_output])  # Skip connection

    # Feed-forward network
    y = LayerNormalization(epsilon=1e-6)(x)
    y = Dense(ff_dim, activation="relu")(y)
    y = Dropout(dropout)(y)
    y = Dense(inputs.shape[-1])(y)
    return Add()([x, y])


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionPooling, self).__init__()
        self.W = tf.keras.layers.Dense(1, activation="tanh")
        
    def call(self, inputs):
        attention_scores = tf.nn.softmax(self.W(inputs), axis=1)
        context_vector = tf.reduce_sum(attention_scores * inputs, axis=1)
        return context_vector


def create_model(n_hours_back_hourly, n_ohlc_features, l2_strength, dropout_rate,
                 n_tech_features, n_eng_features, activation, n_targets, output_activation,
                 initial_bias):
    """
    Create a TensorFlow model with multiple stacked LSTM layers optimized for GPU (cuDNN) support.

    Args:
        timesteps (int): Number of time steps in the input sequence.
        features (int): Number of features for each time step.
        output_dim (int): Number of targets in the output data.
        lstm_layers (list): List specifying the number of units in each LSTM layer.
        output_activation (str or None): Activation function for the output layer.
            Use "sigmoid" for binary classification, "softmax" for multi-class, or
            None for regression (default: "softmax").
        dropout_rate (float): Fraction of units to drop (default: 0.5). 
            Set to 0 to disable dropout.
        batch_size (int or None): Batch size for fixed input size (default: None).

    Returns:
        tf.keras.Model: The compiled TensorFlow model.
    """
    # Hourly LSTM Layers
    hourly_input = Input(shape=(n_hours_back_hourly, n_ohlc_features))
    x_hourly = LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_strength))(hourly_input)
    x_hourly = Dropout(dropout_rate)(x_hourly)
    x_hourly = LSTM(64, kernel_regularizer=l2(l2_strength))(x_hourly)
    
    # Technical Layers
    tech_input = Input(shape=(n_tech_features,))
    x_tech = Dense(32, activation=activation, kernel_regularizer=l2(l2_strength))(tech_input)
    
    # Engineered Layers
    eng_input = Input(shape=(n_eng_features,))
    x_eng = Dense(32, activation=activation, kernel_regularizer=l2(l2_strength))(eng_input)
    
    # Concatenate Layers
    concatenated = Concatenate()([x_hourly, x_tech, x_eng])
    x = Dense(64, activation=activation)(concatenated)
    x = Dense(64, activation=activation)(x)
    
    # Output layer
    output = Dense(n_targets, activation=output_activation, name="output_layer")(x)

    # Define the model
    model = Model(inputs=[hourly_input, tech_input, eng_input], outputs=output)
    model.summary()
    return model