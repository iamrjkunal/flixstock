import os
import json
import math
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import Xception
from utils.data_gen import DataGenerator
FLAGS = tf.compat.v1.app.flags.FLAGS
from tqdm import tqdm

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.enable_eager_execution(config=config)

def class_weight_cal(df):
    wt = {}
    lab = df.value_counts()
    total_sample = lab.sum()
    n_lab = len(lab)
    for i in range(n_lab):
        wt[i] =  (1 / lab[i]) * (total_sample / float(n_lab))
    return wt



input_shape = (256,256)
num_classes = 3
n_channels = 3

def main(_):
    data_dir = FLAGS.data_dir
    image_dir = os.path.join(data_dir, 'images')
    label_path = os.path.join(data_dir, 'attributes.csv')
    label_df = pd.read_csv(label_path)
    label_df.fillna(label_df.mode().iloc[0], inplace=True)
    neck_wt = class_weight_cal(label_df['neck'])
    sleeve_wt = class_weight_cal(label_df['sleeve_length'])
    pattern_wt = class_weight_cal(label_df['pattern'])

    class_weights = {"neck": neck_wt, "sleeve": sleeve_wt, "pattern": pattern_wt}
    save_wt = os.path.join(data_dir, 'class_weights.txt')

    with open(save_wt, "w") as fp:
        json.dump(class_weights, fp)

    end_list = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(image_dir) if f.endswith(end_list)]

    X_train, X_test = train_test_split(images, test_size=0.1, random_state = FLAGS.random_state)

    train_data = os.path.join(data_dir, 'train_data.txt')
    test_data = os.path.join(data_dir, 'test_data.txt')

    with open(train_data, "w") as fp:
        json.dump(X_train, fp)
    with open(test_data, "w") as fp:
        json.dump(X_test, fp)

    labels_idx = label_df.iloc[:, 0].values
    labels= label_df.iloc[:, 1:].values

    label_dict={}
    for i, j in zip(labels_idx, labels):
        label_dict[i]=j
    params= {
        'dim': input_shape,
        'batch_size': FLAGS.batch_size,
        'n_classes': num_classes,
        'n_channels': n_channels,
        'shuffle': True,
    }

    augmentation_kwargs = { 
        'shear_range': 0.1,
        'zoom_range':0.1,
        'width_shift_range':0.1,
        'height_shift_range':0.1,
        'horizontal_flip':True
    }

    train_gen = DataGenerator(X_train, label_dict, data_dir= image_dir, data_flag='train', **params, **augmentation_kwargs)
    test_gen = DataGenerator(X_test, label_dict, data_dir= image_dir, data_flag='test', **params, **augmentation_kwargs)

    input_model= input_shape + (n_channels,) 
    xception = Xception(num_classes=num_classes, input_shape=input_model)
    if FLAGS.restore_model:
        weights = None
        xception.create_model(weights=weights)
        xception.compile(learning_rate=FLAGS.lr, class_weights=class_weights)
        saver = tf.train.Checkpoint(optimizer=xception.optimizer,
                                model=xception.model,
                                optimizer_step=tf.compat.v1.train.get_or_create_global_step())
        saver.restore(tf.train.latest_checkpoint(FLAGS.ckpt_dir)).expect_partial()
    else:
        xception.create_model(weights=FLAGS.imagenet_weights)
        xception.compile(learning_rate=FLAGS.lr, class_weights=class_weights)
    xception.summary()
    previous_best_test = -1
    previous_best_epoch = -1
    for epoch in range(FLAGS.num_epochs):
        iter_train = iter(train_gen)
        for batch in tqdm(range(len(train_gen))):
            X, y= next(iter_train)
            xception.fit(X, y)  
        train_loss = xception.mean_loss.result()
        xception.mean_loss.reset_states()
        
        iter_test = iter(test_gen)
        for batch_test in range(len(test_gen)):
            X, y = next(iter_test)
            predictions = xception.test(X,y)
        test_loss = xception.mean_loss.result()
        test_acc_neck = xception.acc_neck.result()
        test_acc_sleeve = xception.acc_sleeve.result()
        test_acc_pattern = xception.acc_pattern.result()
        test_acc = (test_acc_neck + test_acc_sleeve + test_acc_pattern)/3.0

        xception.mean_loss.reset_states()
        xception.acc_neck.reset_states()
        xception.acc_sleeve.reset_states()
        xception.acc_pattern.reset_states()

        print(f"epoch: {epoch+1}, train_loss: {train_loss}, test_loss: {test_loss}")
        print(f"test_acc_neck: {test_acc_neck:.2f}, test_acc_sleeve: {test_acc_sleeve:.2f}, test_acc_pattern: {test_acc_pattern:.2f}, test_acc: {test_acc:.2f}")
        if test_acc > previous_best_test:
            previous_best_test = test_acc
            previous_best_epoch = epoch + 1
            checkpoint_prefix = os.path.join(FLAGS.ckpt_dir, "model-{:02d}-{:.2f}".format(epoch, test_acc))
            saver = tf.train.Checkpoint(optimizer=xception.optimizer,
                                        model=xception.model,
                                        optimizer_step=tf.compat.v1.train.get_or_create_global_step())

            saver.save(checkpoint_prefix)
            if previous_best_test > FLAGS.stop_loss:
                raise Exception("Training Over due to accuracy threshold")
            

if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_string("data_dir", "data", "data directory")
    tf.compat.v1.app.flags.DEFINE_string("ckpt_dir", "ckpt", "model checkpoints directory")
    tf.compat.v1.app.flags.DEFINE_string("imagenet_weights", "pretrained_weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", "imagenet weights")
    tf.compat.v1.app.flags.DEFINE_float("test_split", 0.1, "test split ratio")
    tf.compat.v1.app.flags.DEFINE_integer("random_state", 30, "data spliting random state")
    tf.compat.v1.app.flags.DEFINE_integer("num_epochs", 3000, "number of epochs")
    tf.compat.v1.app.flags.DEFINE_integer("batch_size", 64, "Batch Size")
    tf.compat.v1.app.flags.DEFINE_float("lr", 0.0001, "Learning Rate")
    tf.compat.v1.app.flags.DEFINE_float("stop_loss", 0.6, "Minimum Overall Accuracy")
    tf.compat.v1.app.flags.DEFINE_boolean("restore_model", False, "Model Restore")
    tf.compat.v1.app.run()
