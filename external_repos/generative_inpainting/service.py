import argparse
import os

import flask
import numpy as np
import tensorflow as tf
import neuralgym as ng

from flask import request, Response

from inpaint_model import InpaintCAModel

app = flask.Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Generative Inpainting Service')
parser.add_argument('-g', '--gpus', default=None)
parser.add_argument('-p', '--data_path', default='../../data')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
checkpoint_dir = os.path.join(args.data_path, 'external', 'generative_inpainting', 'checkpoints')


@app.route('/inpaint', methods=['POST'])
def inpaint():
    FLAGS = ng.Config('inpaint.yml')

    image = np.array(request.json['image'])
    mask = np.array(request.json['mask'])
    mask = np.dstack([mask, mask, mask])

    model = InpaintCAModel()
    assert image.shape == mask.shape, 'image shape is {} while mask shape is {}'.format(image.shape, mask.shape)

    h, w, _ = image.shape
    grid = 8
    image = image[:h // grid * grid, :w // grid * grid, :]
    mask = mask[:h // grid * grid, :w // grid * grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        return {'result': result[0][:, :, ::-1].tolist()}


app.run(host='0.0.0.0', port='9000')
