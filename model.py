import tensorflow as tf
import numpy as np

def weighted_categorical_crossentropy(class_weight):
    def loss(y_true, y_pred):
        y_true = tf.dtypes.cast(y_true, tf.int32)
        one_hot = tf.one_hot(tf.reshape(y_true, [-1]), depth=len(class_weight))
        wt = tf.math.multiply(class_weight, one_hot)
        wt = tf.reduce_sum(wt, axis=-1)
        losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels = y_true, logits=y_pred, weights = wt
        )
        return losses
    return loss

class Xception(object):
    def __init__(self, input_shape, num_classes=2):
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.input= tf.keras.Input(dtype=tf.float32, shape=input_shape)
        self.model = None

    def create_model(self, weights=None, pooling="avg"):
        xception_model= tf.keras.applications.Xception(include_top=False, input_tensor=self.input, weights=weights, pooling="avg")
        xception_model.trainable = False
        xception_output = xception_model.outputs[0]
        self.y_neck = tf.keras.layers.Dense(7, activation='softmax', name='neck_pred')(xception_output)
        self.y_sleeve = tf.keras.layers.Dense(4, activation='softmax', name='sleeve_pred')(xception_output)
        self.y_pattern = tf.keras.layers.Dense(10, activation='softmax', name='pattern_pred')(xception_output)
        self.model = tf.keras.Model(inputs= self.input, outputs= [self.y_neck, self.y_sleeve, self.y_pattern])

    def compile(self, learning_rate=0.001, class_weights=None):
        self.loss_neck = weighted_categorical_crossentropy(list(class_weights["neck"].values()))
        self.loss_sleeve = weighted_categorical_crossentropy(list(class_weights["sleeve"].values()))
        self.loss_pattern = weighted_categorical_crossentropy(list(class_weights["pattern"].values()))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.mean_loss = tf.keras.metrics.Mean(name='mean_loss')
        self.acc_neck = tf.keras.metrics.SparseCategoricalAccuracy(name='acc_neck')
        self.acc_sleeve = tf.keras.metrics.SparseCategoricalAccuracy(name='acc_sleeve')
        self.acc_pattern = tf.keras.metrics.SparseCategoricalAccuracy(name='acc_pattern')


    @tf.function
    def fit(self, x_train, y_train):
        with tf.GradientTape() as tape:
            predictions = self.model(x_train, training=True)
            y_neck, y_sleeve, y_pattern = y_train
            pred_neck, pred_sleeve, pred_pattern = predictions
            loss_1= tf.reduce_mean(self.loss_neck(y_neck, pred_neck))
            loss_2= tf.reduce_mean(self.loss_sleeve(y_sleeve, pred_sleeve))
            loss_3 = tf.reduce_mean(self.loss_pattern(y_pattern, pred_pattern))
            val_1 = tf.abs(loss_1)
            val_2 = tf.abs(loss_2)
            val_3 = tf.abs(loss_3)
            val = val_1 + val_2 + val_3
            loss = [val/val_1*loss_1, val/val_2*loss_2, val/val_3*loss_3]
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.mean_loss.update_state(loss)

    @tf.function
    def test(self, x_test, y_test):
        predictions = self.model(x_test, training=False)
        y_neck, y_sleeve, y_pattern = y_test
        pred_neck, pred_sleeve, pred_pattern = predictions
        loss_1= tf.reduce_mean(self.loss_neck(y_neck, pred_neck))
        loss_2= tf.reduce_mean(self.loss_sleeve(y_sleeve, pred_sleeve))
        loss_3 = tf.reduce_mean(self.loss_pattern(y_pattern, pred_pattern))
        loss = [loss_1, loss_2, loss_3]
        self.mean_loss.update_state(loss)
        self.acc_neck.update_state(y_neck, pred_neck)
        self.acc_sleeve.update_state(y_sleeve, pred_sleeve)
        self.acc_pattern.update_state(y_pattern, pred_pattern)
        return predictions

    def predict(self, x):
        predictions = self.model.predict(x)
        return predictions

    def summary(self):
        self.model.summary()

