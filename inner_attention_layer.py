from keras.layers.core import Layer
from keras import regularizers, constraints
from keras.initializers import glorot_uniform
from keras import backend as K
import tensorflow as tf


class InnerAttentionLayer(Layer):
    def __init__(self, topic, emb_dim=300, maxpool=False, return_sequence=False, W_regularizer=None, W_constraint=None,
                 **kwargs):
        
        self.supports_masking = True
        self.init = glorot_uniform()
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        
        self.emb_dim = emb_dim
        self.topic = topic
        self.maxpool = maxpool
        self.return_sequences = return_sequence
        
        super(InnerAttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight(shape=(self.emb_dim, self.emb_dim,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        
        self.built = True
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x, mask=None):
        
        # W * s
        s_flat = tf.reshape(x, [-1, tf.shape(x)[2]])
        W_s_topic_flat = tf.transpose(tf.matmul(self.W, tf.transpose(s_flat)))
        W_s = tf.reshape(W_s_topic_flat, [tf.shape(x)[0], tf.shape(x)[1], -1])
        
        # t * W_s
        t = tf.expand_dims(self.topic, 1)
        t_W_s = tf.matmul(t, tf.transpose(W_s, perm=[0, 2, 1]))
        t_W_s = tf.squeeze(t_W_s)
        
        # a = tf.tanh(t_W_s)
        a = tf.sigmoid(t_W_s)
        
        # softmax & masking
        # ex = K.exp(s_aq)
        
        # apply mask
        # if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            #ex *= K.cast(mask, K.floatx())
        
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number e to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        #ex /= K.cast(K.sum(ex, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        #sum = K.sum(ex, axis=[1], keepdims=True)
        #attention_weights = ex / sum
        
        # weight lstm states with alphas
        attention_weights = K.expand_dims(a)
        weighted_states = x * attention_weights
        
        if self.return_sequences:
            return weighted_states
        if self.maxpool:
            return self.get_maxpool(weighted_states)
        else:
            return K.sum(weighted_states, axis=1)

    def get_maxpool(self, item):
        return tf.reduce_max(item, [1], keep_dims=False)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[-1]

class InnerAttentionLayerKeras(Layer):
    # Same as InnerAttentionLayer() but uses Keras backend squeeze, since otherwise Keras fails to compute the output shape correctly.
    def __init__(self, topic, emb_dim=300, maxpool=False, return_sequence=False, W_regularizer=None, W_constraint=None, return_weights=False,
                 **kwargs):
        
        self.supports_masking = True
        self.init = glorot_uniform()
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        
        self.emb_dim = emb_dim
        self.topic = topic
        self.maxpool = maxpool
        self.return_sequences = return_sequence
        self.return_weights = return_weights
        
        super(InnerAttentionLayerKeras, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight(shape=(self.emb_dim, self.emb_dim,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        
        # W * s
        s_flat = tf.reshape(x, [-1, tf.shape(x)[2]])
        W_s_topic_flat = tf.transpose(tf.matmul(self.W, tf.transpose(s_flat)))
        W_s = tf.reshape(W_s_topic_flat, [tf.shape(x)[0], tf.shape(x)[1], -1])
        
        # t * W_s
        t = tf.expand_dims(self.topic, 1)
        t_W_s = tf.matmul(t, tf.transpose(W_s, perm=[0, 2, 1]))
        t_W_s = K.squeeze(t_W_s, axis=1)
        
        # a = tf.tanh(t_W_s)
        a = tf.sigmoid(t_W_s)
        
        # softmax & masking
        # ex = K.exp(s_aq)
        
        # apply mask
        # if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            #ex *= K.cast(mask, K.floatx())
        
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number e to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        #ex /= K.cast(K.sum(ex, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        #sum = K.sum(ex, axis=[1], keepdims=True)
        #attention_weights = ex / sum
        
        # weight lstm states with alphas
        attention_weights = K.expand_dims(a)
        weighted_states = x * attention_weights
        
        result = []
        
        if self.return_sequences:
            result.append(weighted_states)
        elif self.maxpool:
            result.append(self.get_maxpool(weighted_states))
        else:
            result.append(K.sum(weighted_states, axis=1))
            
        if self.return_weights == True:
            result.append(a)
            return result
        else:
            return result[0]
    
    def get_maxpool(self, item):
        return tf.reduce_max(item, [1], keep_dims=False)
    
    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            final_out = input_shape
        else:
            final_out = tuple([input_shape[0], input_shape[-1]])
            
        if self.return_weights == True:
            return [final_out, tuple([input_shape[0], input_shape[1]])]
        else:
            return final_out
    
    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        if self.return_sequences:
            final_out = input_shape
        else:
            final_out = tuple([input_shape[0], input_shape[-1]])
            
        if self.return_weights == True:
            return [final_out, tuple([input_shape[0], input_shape[1]])]
        else:
            return final_out

class InnerAttentionLayer2(Layer):
    def __init__(self, topic, emb_dim=300, maxpool=False, return_sequence=False, regularizer=None, constraint=None,
                 **kwargs):
        
        self.supports_masking = True
        self.init = glorot_uniform()
        
        self.W_regularizer = regularizers.get(regularizer)
        self.W_constraint = constraints.get(constraint)
        
        #self.cos = cos
        
        self.emb_dim = emb_dim
        self.topic = topic
        self.maxpool = maxpool
        self.return_sequences = return_sequence
        
        super(InnerAttentionLayer2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight(shape=(input_shape[-1], self.emb_dim,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        
        self.W_a = self.add_weight(shape=(input_shape[-1], self.emb_dim,),
                                 initializer=self.init,
                                 name='{}_W_a'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        
        self.W_h = self.add_weight(shape=(input_shape[-1], self.emb_dim,),
                                 initializer=self.init,
                                 name='{}_W_h'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        
        # self.W_c = self.add_weight(shape=(input_shape[-1], self.emb_dim,),
        #                         initializer=self.init,
        #                        name='{}_W_c'.format(self.name),
        #                       regularizer=self.W_regularizer,
        #                      constraint=self.W_constraint)
        
        self.built = True
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x, mask=None):
        
        # W * s
        s_flat = tf.reshape(x, [-1, tf.shape(x)[2]])
        W_s_flat = tf.transpose(tf.matmul(self.W, tf.transpose(s_flat)))
        W_s = tf.reshape(W_s_flat, [tf.shape(x)[0], tf.shape(x)[1], -1])
        
        # t * W_s
        t = tf.expand_dims(self.topic, 1)
        t_W_s = tf.matmul(t, tf.transpose(W_s, perm=[0,2,1]))
        t_W_s = tf.squeeze(t_W_s)
        
        a = tf.sigmoid(t_W_s)
        
        a_weights = tf.expand_dims(a, 2)
        a_weighted_states = a_weights * x
        a_flat = tf.reshape(a_weighted_states, [-1, tf.shape(a_weighted_states)[2]])
        W_a_a_flat = tf.transpose(tf.matmul(self.W_a, tf.transpose(a_flat)))
        W_a_a = tf.reshape(W_a_a_flat, [tf.shape(x)[0], tf.shape(x)[1], -1])
        
        # W_h * s
        s_flat = tf.reshape(x, [-1, tf.shape(x)[2]])
        W_h_s_flat = tf.transpose(tf.matmul(self.W_h, tf.transpose(s_flat)))
        W_h_s = tf.reshape(W_h_s_flat, [tf.shape(x)[0], tf.shape(x)[1], -1])
        
        
        #
        # cos_weights = tf.expand_dims(self.cos, 2)
        # cos_weighted_states = self.cos * x
        # cos_flat = tf.reshape(cos_weighted_states, [-1, tf.shape(cos_weighted_states)[2]])
        # W_c_cos_flat = tf.transpose(tf.matmul(self.W_c, tf.transpose(cos_flat)))
        # W_c_cos = tf.reshape(W_c_cos_flat, [tf.shape(x)[0], tf.shape(x)[1], -1])
        
        
        weighted_states = W_a_a + W_h_s
        
        
        
        if self.return_sequences:
            return weighted_states
        if self.maxpool:
            return self.get_maxpool(weighted_states)
        else:
            return K.sum(weighted_states, axis=1)
    
    def get_maxpool(self, item):
        return tf.reduce_max(item, [1], keep_dims=False)
    
    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[-1]
    
    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[-1]


class InnerAttentionLayer3(Layer):
    def __init__(self, topic, emb_dim=300, maxpool=False, return_sequence=False, W_regularizer=None, W_constraint=None,
                 **kwargs):
        
        self.supports_masking = True
        self.init = glorot_uniform()
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        
        self.emb_dim = emb_dim
        self.topic = topic
        self.maxpool = maxpool
        self.return_sequences = return_sequence
        
        super(InnerAttentionLayer3, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight(shape=(self.emb_dim, self.emb_dim,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        
        self.built = True
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x, mask=None):
        
        # W * s
        s_flat = tf.reshape(x, [-1, tf.shape(x)[2]])
        W_s_topic_flat = tf.transpose(tf.matmul(self.W, tf.transpose(s_flat)))
        W_s = tf.reshape(W_s_topic_flat, [tf.shape(x)[0], tf.shape(x)[1], -1])
        
        # t * W_s
        t = tf.expand_dims(self.topic, 1)
        t_W_s = tf.matmul(t, tf.transpose(W_s, perm=[0, 2, 1]))
        t_W_s = tf.squeeze(t_W_s)
        
        # a = tf.tanh(t_W_s)
        a = tf.sigmoid(t_W_s)
        
        # softmax & masking
        ex = K.exp(a)
        
        # apply mask
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ex *= K.cast(mask, K.floatx())
            
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number e to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        ex /= K.cast(K.sum(ex, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        sum = K.sum(ex, axis=[1], keepdims=True)
        attention_weights = ex / sum
        
        # weight lstm states with alphas
        attention_weights = K.expand_dims(attention_weights)
        weighted_states = x * attention_weights
        
        if self.return_sequences:
            return weighted_states
        if self.maxpool:
            return self.get_maxpool(weighted_states)
        else:
            return K.sum(weighted_states, axis=1)
    
    def get_maxpool(self, item):
        return tf.reduce_max(item, [1], keep_dims=False)
    
    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[-1]
    
    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[-1]

