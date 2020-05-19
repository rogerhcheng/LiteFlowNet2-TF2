import math
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
from model import LiteFlowNet2
import argparse

from draw_flow import *
tf.disable_eager_execution()


def pad_image(image):
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1

    nh = int(math.ceil(h / 32.) * 32)
    nw = int(math.ceil(w / 32.) * 32)

    pad_i = np.zeros([nh, nw, c])
    pad_i[:h, :w] = image
    return pad_i

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--img1', default='images/first.png')
parser.add_argument('--img2', default='images/second.png')
parser.add_argument('--use_Sintel', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--display_flow', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--img_out', default='out.png')

args = parser.parse_args()

# Create TF session
sess = tf.Session()
model = LiteFlowNet2(isSintel=args.use_Sintel)
tens1 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
tens2 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
out = model(tens1, tens2)

# Load model
saver = tf.train.Saver()
if args.use_Sintel:
    saver.restore(sess, "./models/LiteFlowNet2_Sintel_model")
else:
    saver.restore(sess, "./models/LiteFlowNet2_Kitti_model")

# Load images
inp1 = Image.open(args.img1)
inp2 = Image.open(args.img2)

w, h = inp1.size[:2]
inp1 = np.float32(np.expand_dims(pad_image(np.asarray(inp1)[..., ::-1]), 0)) / 255.0
inp2 = np.float32(np.expand_dims(pad_image(np.asarray(inp2)[..., ::-1]), 0)) / 255.0

# input in bgr format
flow = sess.run(out, feed_dict={tens1: inp1, tens2: inp2})[0, :h, :w, :]


# visualise flow with color model as image and save
flow_color = flow_to_color(flow, convert_to_bgr=False)
flow_image = Image.fromarray(flow_color)

if args.display_flow:
    flow_image.show()
    
if args.img_out != "":
    flow_image.save(args.img_out,args.img_out[-3:])



