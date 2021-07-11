import tensorflow as tf
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from model import Xception
from utils.preprocess_utils import preprocess_input
from train import input_shape, num_classes, n_channels
os.environ['KMP_DUPLICATE_LIB_OK']='True'
FLAGS = tf.compat.v1.app.flags.FLAGS

 
def read_image(image_path, inp_shape):
    resize_dim = inp_shape[0:2]
    n_channels = inp_shape[-1]
    img = cv2.imread(image_path)
    img = cv2.resize(img, resize_dim)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (n_channels == 1):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image, v2=True)
    return image

def main(_):
    test_data_dir = FLAGS.test_data_dir
    output_dir = FLAGS.output_dir

    wt_path = 'data/class_weights.txt'
    with open(wt_path, "r") as fp:
        class_weights = json.load(fp)

    input_model= input_shape + (n_channels,) 
    xception = Xception(num_classes=num_classes, input_shape=input_model)
    xception.create_model(weights=None)
    xception.compile(class_weights=class_weights)
    saver = tf.train.Checkpoint(optimizer=xception.optimizer,
                                model=xception.model,
                                optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    saver.restore(tf.train.latest_checkpoint(FLAGS.ckpt_dir)).expect_partial()
    

    end_list = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(test_data_dir) if f.endswith(end_list)]
    output_li = []
    print("Predicting...")
    for image in tqdm(images):
        image_path = os.path.join(test_data_dir, image)
        final_img = read_image(image_path, input_model)
        y_pred = xception.predict(final_img)
        temp_li = []
        for i in range(3):
            temp_li.append(np.argmax(y_pred[i], axis=-1)[0])
        output_li.append(temp_li)
    output_li = np.array(output_li)
    output = pd.DataFrame({'filename': images, 'neck': output_li[:, 0], 'sleeve_length': output_li[:, 1], 'pattern': output_li[:, 2]})
    csv_name = os.path.join(FLAGS.output_dir, "output.csv")
    output.to_csv (csv_name, index = None) 
    print(f"Prediction done. Check csv file generated in {FLAGS.output_dir} directory")


if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_string("test_data_dir", "custom_data/test_images", "test data directory")
    tf.compat.v1.app.flags.DEFINE_string("output_dir", "custom_data", "output file saving directory")
    tf.compat.v1.app.flags.DEFINE_string("ckpt_dir", "ckpt", "model checkpoints directory")
    tf.compat.v1.app.run()
