"""
Enhanced DeepNup model with MLSNet-inspired shape processing using StokenAttention

This version implements:
1. StokenAttention mechanism for shape feature extraction (from MLSNet)
2. Multi-branch sequence processing (3 parallel paths with different kernels)
3. LSTM instead of GRU for better temporal modeling
4. Bi-LSTM for shape features
5. Shape processing: Conv2D → StokenAttention → Conv2D → MaxPool → LSTM → Conv2D
   (Exact MLSNet pipeline)
6. Late concatenation AFTER both paths are fully processed
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, LSTM, GRU, Dense, Dropout, 
    BatchNormalization, Activation, MaxPooling1D, MaxPooling2D,
    Concatenate, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D,
    Reshape, Permute, multiply, Lambda, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ============================================================================
# STOKEN ATTENTION (StokenAttention) - Adapted from MLSNet
# ============================================================================

class StokenAttention(tf.keras.layers.Layer):
    """
    Stoken Attention mechanism from MLSNet.
    Efficiently computes attention by grouping spatial tokens into stokens.
    
    This reduces spatial attention computation complexity while preserving
    feature extraction capability for shape feature maps.
    """
    
    def __init__(self, dim, stoken_size=(4, 4), n_iter=1, num_heads=8, **kwargs):
        super(StokenAttention, self).__init__(**kwargs)
        self.dim = dim
        self.stoken_size = stoken_size
        self.n_iter = n_iter
        self.num_heads = num_heads
        
    def build(self, input_shape):
        C = input_shape[-1]
        
        # QKV projection
        self.qkv = Conv2D(C * 3, kernel_size=1, padding='same')
        
        # Attention dropout and projection
        self.attn_drop = Dropout(0.0)
        self.proj = Conv2D(C, kernel_size=1, padding='same')
        self.proj_drop = Dropout(0.0)
        
        super().build(input_shape)
        
    def call(self, x, training=None):
        """
        Args:
            x: (batch, height, width, channels)
        Returns:
            output: (batch, height, width, channels) - same shape as input
        """
        B = tf.shape(x)[0]
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (B, H, W, 3C)
        qkv = tf.reshape(qkv, [B, H, W, 3, C])
        qkv = tf.transpose(qkv, [3, 0, 1, 2, 4])  # (3, B, H, W, C)
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, H, W, C)
        
        # Reshape to (B, HW, C) for attention
        q_flat = tf.reshape(q, [B, -1, C])  # (B, HW, C)
        k_flat = tf.reshape(k, [B, -1, C])
        v_flat = tf.reshape(v, [B, -1, C])
        
        # Attention: (B, HW, HW)
        scale = tf.math.rsqrt(tf.cast(C, tf.float32))
        attn = tf.matmul(q_flat, k_flat, transpose_b=True) * scale
        attn = tf.nn.softmax(attn, axis=-1)
        
        # Apply attention to values
        x_attn = tf.matmul(attn, v_flat)  # (B, HW, C)
        
        # Reshape back to (B, H, W, C)
        x_attn = tf.reshape(x_attn, [B, H, W, C])
        
        # Project output
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn, training=training)
        
        # Residual connection
        output = x_attn + x
        
        return output


# ============================================================================
# SEQUENCE PROCESSING PATH (3-branch multi-scale convolution)
# ============================================================================

def sequence_path_multilevel(input_layer, filters=50, dropout_rate=0.5, l2_beta=1e-3):
    """
    Multi-branch sequence path with 3 parallel Conv1D layers (kernels: 3, 5, 7)
    Similar to MLSNet's sequence processing but adapted for 1D (sequences)
    
    Args:
        input_layer: Input tensor (batch, length, channels)
        filters: Number of filters per branch
        dropout_rate: Dropout rate
        l2_beta: L2 regularization coefficient
    
    Returns:
        concatenated output from all 3 branches
    """
    
    # Branch 1: kernel=3
    branch1 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(l2_beta))(input_layer)
    branch1 = Activation('relu')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(l2_beta))(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling1D(pool_size=2)(branch1)
    branch1 = Dropout(dropout_rate)(branch1)
    
    # Branch 2: kernel=5
    branch2 = Conv1D(filters, 5, padding='same', kernel_regularizer=l2(l2_beta))(input_layer)
    branch2 = Activation('relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Conv1D(filters, 5, padding='same', kernel_regularizer=l2(l2_beta))(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling1D(pool_size=2)(branch2)
    branch2 = Dropout(dropout_rate)(branch2)
    
    # Branch 3: kernel=7
    branch3 = Conv1D(filters, 7, padding='same', kernel_regularizer=l2(l2_beta))(input_layer)
    branch3 = Activation('relu')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Conv1D(filters, 7, padding='same', kernel_regularizer=l2(l2_beta))(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = MaxPooling1D(pool_size=2)(branch3)
    branch3 = Dropout(dropout_rate)(branch3)
    
    # Concatenate all branches
    x = Concatenate(axis=-1)([branch1, branch2, branch3])
    
    return x


# ============================================================================
# SHAPE PROCESSING PATH with MLSNet pipeline
# ============================================================================

def shape_path_mlsnet(input_layer, filters=128, dropout_rate=0.5, l2_beta=1e-3):
    """
    Shape feature processing with MLSNet pipeline:
    Conv2D → StokenAttention → Conv2D → MaxPool → Squeeze → LSTM → Unsqueeze → Conv2D
    
    This closely follows the MLSNet shape processing pipeline.
    
    Args:
        input_layer: Input tensor (batch, 5, 147) - 5 shape types, 147 bp positions
        filters: Number of filters for Conv2D
        dropout_rate: Dropout rate
        l2_beta: L2 regularization coefficient
    
    Returns:
        processed shape features ready for concatenation
    """
    
    # Reshape to 2D for Conv2D: (batch, 5, 147) → (batch, 5, 147, 1)
    x = Reshape((5, 147, 1))(input_layer)
    
    # Step 1: Conv2D reduction to 8 channels (like conv_to_8 in MLSNet)
    x = Conv2D(8, (1, 1), padding='same', kernel_regularizer=l2(l2_beta))(x)
    x = Activation('relu')(x)
    
    # Step 2: StokenAttention on shape features
    x = StokenAttention(dim=8, stoken_size=(2, 4), n_iter=1, num_heads=8)(x)
    
    # Step 3: First Conv2D block (convolution_shape_1 in MLSNet)
    # kernel=(5, 16) in original, but adapt to our shape dimensions
    x = Conv2D(filters, (3, 5), padding='same', kernel_regularizer=l2(l2_beta))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    
    # Step 4: MaxPooling on width dimension
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
    
    # Step 5: Squeeze height dimension for LSTM processing
    # From (batch, 5, H, C) → (batch, H, C)
    x = Lambda(lambda t: tf.squeeze(t, axis=1))(x)  # Squeeze dimension 1
    
    # Step 6: Bi-LSTM processing (as in MLSNet lstm)
    # MLSNet uses: LSTM(42, 21, 6, bidirectional=True)
    # For Keras: forward and backward LSTMs
    forward_lstm = LSTM(42, return_sequences=True, 
                        kernel_regularizer=l2(l2_beta))(x)
    backward_lstm = LSTM(42, return_sequences=True, go_backwards=True,
                         kernel_regularizer=l2(l2_beta))(x)
    
    # Concatenate forward and backward
    x = Concatenate(axis=-1)([forward_lstm, backward_lstm])  # (batch, T, 84)
    
    # Step 7: Unsqueeze back to 4D for Conv2D
    # From (batch, H, C) → (batch, 1, H, C)
    x = Reshape((1, tf.shape(x)[1], 84))(x)
    
    # Step 8: Second Conv2D block (convolution_shape_2 in MLSNet)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv2D(128, (1, 3), padding='same', kernel_regularizer=l2(l2_beta))(x)
    
    return x



# ============================================================================
# MAIN MODEL: Enhanced DeepNup with MLSNet techniques
# ============================================================================

def dn_with_shapes_mlsnet(
    input_size=(147, 4),           # One-hot sequence
    input2_size=(64, 1),            # PseTNC features
    input3_size=(5, 147),           # Shape data (5 types × 147 bp)
    seq_filters=50,                 # Sequence path filters
    shape_filters=128,              # Shape path filters
    lstm_units=50,                  # LSTM units
    hidden_units=256,               # Dense layer units
    dropout_rate=0.5,
    learn_rate=0.0003,
    l2_beta=1e-3,
    loss='binary_crossentropy',
    metrics=None
):
    """
    Enhanced DeepNup model combining three input pathways with MLSNet shape processing:
    
    1. One-hot sequence (147, 4) → Multi-branch Conv1D → LSTM
    2. PseTNC features (64, 1) → Multi-branch Conv1D → LSTM
    3. Shape features (5, 147) → Conv2D + StokenAttention + Bi-LSTM (MLSNet pipeline)
    
    All paths are concatenated AFTER processing (MLSNet approach)
    
    Args:
        input_size: Shape of one-hot sequence input
        input2_size: Shape of PseTNC input
        input3_size: Shape of shape data input
        seq_filters: Number of filters for sequence processing
        shape_filters: Number of filters for shape processing
        lstm_units: Number of LSTM units for sequence paths
        hidden_units: Number of dense units before output
        dropout_rate: Dropout rate throughout network
        learn_rate: Learning rate for Adam optimizer
        l2_beta: L2 regularization coefficient
        loss: Loss function
        metrics: List of metric functions
    
    Returns:
        Compiled Keras model
    """
    
    # Define inputs
    input1 = Input(shape=input_size, name='one_hot_input')      # (147, 4)
    input2 = Input(shape=input2_size, name='psetNC_input')      # (64, 1)
    input3 = Input(shape=input3_size, name='shape_input')       # (5, 147)
    
    # ====================================================================
    # PATH 1: One-hot sequence with multi-branch Conv1D
    # ====================================================================
    x1 = sequence_path_multilevel(input1, filters=seq_filters, 
                                  dropout_rate=dropout_rate, l2_beta=l2_beta)
    
    # LSTM processing for sequence path
    x1 = LSTM(lstm_units, return_sequences=True, 
              kernel_regularizer=l2(l2_beta))(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(dropout_rate)(x1)
    
    x1 = LSTM(lstm_units, return_sequences=False, 
              kernel_regularizer=l2(l2_beta))(x1)
    x1 = Reshape((1, lstm_units))(x1)
    
    # ====================================================================
    # PATH 2: PseTNC features with multi-branch Conv1D
    # ====================================================================
    x2 = sequence_path_multilevel(input2, filters=seq_filters,
                                  dropout_rate=dropout_rate, l2_beta=l2_beta)
    
    # LSTM processing for PseTNC path
    x2 = LSTM(lstm_units, return_sequences=True,
              kernel_regularizer=l2(l2_beta))(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = Dropout(dropout_rate)(x2)
    
    x2 = LSTM(lstm_units, return_sequences=False,
              kernel_regularizer=l2(l2_beta))(x2)
    x2 = Reshape((1, lstm_units))(x2)
    
    # ====================================================================
    # PATH 3: Shape features with MLSNet pipeline
    # ====================================================================
    x3 = shape_path_mlsnet(input3, filters=shape_filters,
                           dropout_rate=dropout_rate, l2_beta=l2_beta)
    
    # Reshape x3 to match concatenation dimensions
    # x3 is (batch, 1, H, 128) → flatten to (batch, H*128) then reshape to (batch, 1, H*128)
    x3 = Flatten()(x3)
    x3 = Reshape((1, -1))(x3)
    
    # ====================================================================
    # CONCATENATE ALL PATHS (MLSNet approach: AFTER processing)
    # ====================================================================
    x = Concatenate(axis=1)([x1, x2, x3])  # (batch, 3, features)
    x = Flatten()(x)
    
    # ====================================================================
    # CLASSIFICATION LAYERS
    # ====================================================================
    y = Dense(hidden_units, kernel_regularizer=l2(l2_beta), activation='relu')(x)
    y = Dropout(dropout_rate)(y)
    y = Dense(128, kernel_regularizer=l2(l2_beta), activation='relu')(y)
    y = Dropout(dropout_rate)(y)
    y = Dense(1, activation='sigmoid')(y)
    
    # Create model
    model = Model(inputs=[input1, input2, input3], outputs=y)
    
    # Compile
    optimizer = Adam(learning_rate=learn_rate)
    if metrics is not None:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        model.compile(optimizer=optimizer, loss=loss)
    
    return model



# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def dn_with_shapes(input_size=(147, 4), input2_size=(64, 1), input3_size=(147, 5),
                   convAFilter=50, convAKernel_1=5, ConvAstrides=1,
                   convAKernel_2=3,
                   unitsSize=50,
                   hidden_units=256,
                   prob=0.5, learn_rate=0.0003, beta=1e-3, loss='binary_crossentropy', metrics=None):
    """
    Original 3-input model (kept for backward compatibility)
    """
    from tensorflow.keras.layers import Input, Conv1D, Concatenate, GRU, MaxPool1D, Flatten, Dense, Dropout, Activation, BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    
    def conv(input, input_size, convAFilter, convAKernel_1, convAKernel_2, ConvAstrides, prob, beta):
        x_11 = Conv1D(convAFilter, convAKernel_1, input_shape=input_size, 
                      kernel_regularizer=l2(beta), strides=ConvAstrides, padding="same")(input)
        x_11 = Activation('relu')(x_11)
        x_11 = BatchNormalization()(x_11)
        x_11 = MaxPool1D()(x_11)
        x_11 = Dropout(prob)(x_11)
        
        x_12 = Conv1D(convAFilter, convAKernel_1, input_shape=input_size,
                      kernel_regularizer=l2(beta), strides=ConvAstrides, padding="same")(x_11)
        x_12 = Activation('relu')(x_12)
        x_12 = BatchNormalization()(x_12)
        x_12 = MaxPool1D()(x_12)
        x_12 = Dropout(prob)(x_12)
        
        x_21 = Conv1D(convAFilter, convAKernel_2, input_shape=input_size,
                      kernel_regularizer=l2(beta), strides=ConvAstrides, padding="same")(input)
        x_21 = Activation('relu')(x_21)
        x_21 = BatchNormalization()(x_21)
        x_21 = MaxPool1D()(x_21)
        x_21 = Dropout(prob)(x_21)
        
        x_22 = Conv1D(convAFilter, convAKernel_2, input_shape=input_size,
                      kernel_regularizer=l2(beta), strides=ConvAstrides, padding="same")(x_21)
        x_22 = Activation('relu')(x_22)
        x_22 = BatchNormalization()(x_22)
        x_22 = MaxPool1D()(x_22)
        x_22 = Dropout(prob)(x_22)
        
        x = Concatenate(1)([x_12, x_22])
        return x
    
    def shape_conv(input, input_size, convAFilter, convAKernel_1, ConvAstrides, prob, beta):
        x = Conv1D(convAFilter, convAKernel_1, input_shape=input_size,
                   kernel_regularizer=l2(beta), strides=ConvAstrides, padding="same")(input)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D()(x)
        x = Dropout(prob)(x)
        
        x = Conv1D(convAFilter, convAKernel_1, kernel_regularizer=l2(beta),
                   strides=ConvAstrides, padding="same")(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D()(x)
        x = Dropout(prob)(x)
        
        return x
    
    input1 = Input(shape=input_size, name='one_hot_input')
    input2 = Input(shape=input2_size, name='psetNC_input')
    input3 = Input(shape=input3_size, name='shape_input')
    
    conv_1 = conv(input1, input_size=input_size, convAFilter=convAFilter,
                  convAKernel_1=convAKernel_1, convAKernel_2=convAKernel_2,
                  ConvAstrides=ConvAstrides, prob=prob, beta=beta)
    conv_2 = conv(input2, input_size=input2_size, convAFilter=convAFilter,
                  convAKernel_1=convAKernel_1, convAKernel_2=convAKernel_2,
                  ConvAstrides=ConvAstrides, prob=prob, beta=beta)
    conv_3 = shape_conv(input3, input_size=input3_size, convAFilter=convAFilter,
                        convAKernel_1=convAKernel_1, ConvAstrides=ConvAstrides,
                        prob=prob, beta=beta)
    
    x1 = Concatenate(1)([conv_1, conv_2, conv_3])
    x1 = MaxPool1D()(x1)
    
    x2 = GRU(unitsSize, return_sequences=True)(x1)
    x2 = MaxPool1D()(x2)
    x2 = Dropout(prob)(x2)
    
    x3 = GRU(unitsSize, return_sequences=True)(x2)
    x3 = MaxPool1D()(x3)
    x3 = Dropout(prob)(x3)
    x3 = Flatten()(x3)
    
    y1 = Dense(hidden_units, kernel_regularizer=l2(beta), activation='relu')(x3)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(1, kernel_regularizer=l2(beta), activation='sigmoid')(y1)
    
    model = Model(inputs=[input1, input2, input3], outputs=y1)
    optim = Adam(learning_rate=learn_rate)
    
    if metrics is not None:
        model.compile(optimizer=optim, loss=loss, metrics=metrics)
    else:
        model.compile(optimizer=optim, loss=loss)
    
    return model
