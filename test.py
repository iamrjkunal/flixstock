import tensorflow as tf
import json
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from model import Xception
from sklearn.metrics import classification_report
from utils.data_gen import DataGenerator
from train import input_shape, num_classes, n_channels
import warnings
warnings.filterwarnings('ignore') 
FLAGS = tf.compat.v1.app.flags.FLAGS


def eval_metric(y_true, y_pred, num_classes):
    labels = [i for i in range(num_classes)]
    report= classification_report(y_true, y_pred, labels=labels)
    return report

def true_pred_labels(test_gen, model):
    num_class_1 = 7
    num_class_2 = 4
    num_class_3 = 10
    y_true_1 = []
    y_pred_1 = []
    y_true_2 = []
    y_pred_2 = []
    y_true_3 = []
    y_pred_3 = []
    it = iter(test_gen)
    for i in tqdm(range(len(test_gen))):
        x, y = next(it)
        y_pred = model.predict(x)
        y_1 = np.squeeze(y[0]).astype(int)
        y_true_1 += y_1.tolist()
        y_2 = np.squeeze(y[1]).astype(int)
        y_true_2 += y_2.tolist()
        y_3 = np.squeeze(y[2]).astype(int)
        y_true_3 += y_3.tolist()
        pred_1 = np.argmax(y_pred[0], axis=-1)
        y_pred_1 += pred_1.tolist()
        pred_2 = np.argmax(y_pred[1], axis=-1)
        y_pred_2 += pred_2.tolist()
        pred_3 = np.argmax(y_pred[2], axis=-1)
        y_pred_3 += pred_3.tolist()
    
    return (y_true_1, y_pred_1, num_class_1), (y_true_2, y_pred_2, num_class_2), (y_true_3, y_pred_3, num_class_3)  
        
        

def main(_):
    data_dir = FLAGS.data_dir
    image_dir = os.path.join(data_dir, 'images')
    label_path = os.path.join(data_dir, 'attributes.csv')
    label_df = pd.read_csv(label_path)
    label_df.fillna(label_df.mode().iloc[0], inplace=True)
    
    test_data = os.path.join(data_dir, 'test_data.txt')
    wt = os.path.join(data_dir, 'class_weights.txt')

    with open(wt, "r") as fp:
        class_weights = json.load(fp)

    with open(test_data, "r") as fp:
        X_test = json.load(fp)

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

    augmentation_kwargs = {}
    
    test_gen = DataGenerator(X_test, label_dict, data_dir= image_dir, data_flag='test', **params, **augmentation_kwargs)
    
    input_model= input_shape + (n_channels,) 
    xception = Xception(num_classes=num_classes, input_shape=input_model)
    xception.create_model(weights=None)
    xception.compile(class_weights=class_weights)
    saver = tf.train.Checkpoint(optimizer=xception.optimizer,
                                model=xception.model,
                                optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    saver.restore(tf.train.latest_checkpoint(FLAGS.ckpt_dir)).expect_partial()
    
    print("Predicting Values...")
    output = true_pred_labels(test_gen, xception)

    classes_name = {0: "Neck", 1:"Sleeve Length", 2: "Pattern"}

    for i in range(3):
        print(f"### Classification Report For Class : {classes_name[i]} ###")
        print(eval_metric(output[i][0], output[i][1], output[i][2]))


if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_string("data_dir", "data", "data directory")
    tf.compat.v1.app.flags.DEFINE_string("ckpt_dir", "ckpt", "model checkpoints directory")
    tf.compat.v1.app.flags.DEFINE_integer("batch_size", 16, "Batch Size")
    tf.compat.v1.app.run()
