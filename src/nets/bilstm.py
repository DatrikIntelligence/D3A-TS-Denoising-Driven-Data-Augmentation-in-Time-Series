import tensorflow as tf

def create_model(input_shape, output, bidirectional=True, cell_type=tf.keras.layers.GRUCell,
                     nblocks=3, rnn_units=128, l1=1e-4, l2=1e-4, dropout=0.05,
                     fc1=128, fc2=64, dense_activation='relu', batch_normalization=True, output_dim=1):
    
    # model creation
    input_tensor = tf.keras.layers.Input(input_shape)
    x = input_tensor
    
    if input_shape[-1] == 1:
        x = tf.keras.layers.Reshape(input_shape[:-1])(x)
        

    rnn_units = [rnn_units] * nblocks
    for i, units in enumerate(rnn_units):
        #return_sequences = (i < len(rnn_units) - 1) or attention
        return_sequences = True
        cell = tf.keras.layers.RNN(cell_type(units=units, 
                                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2)),
                         name='rnn_cell_%d' % (i+1),
                         return_sequences=return_sequences,
                         )
        if bidirectional:
            x = tf.keras.layers.Bidirectional(cell)(x)
        else:
            x = cell(x)
            
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
            
    
    # FNN
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(fc1, 
             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Activation(dense_activation)(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(fc2, 
             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Activation(dense_activation)(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_dim, activation=output, name='predictions')(x) 
    
    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    
    return model

