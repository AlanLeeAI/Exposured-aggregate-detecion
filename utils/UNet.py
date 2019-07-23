import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non
	net = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net

def attention_gate(x_l,g,filter):
	x_l_1 = slim.conv2d(x_l, filter, kernel_size=1, stride=1,padding="SAME",activation_fn=None)
	g_1 = slim.conv2d(g, filter, kernel_size=1, stride=1,padding="SAME",activation_fn=None)
	x_g_sum = tf.add(x_l_1,g_1)
	sum_relu = tf.nn.relu(x_g_sum)
	x_g_out = slim.conv2d(sum_relu, 1, kernel_size=1, stride=1,padding="SAME",activation_fn=tf.nn.sigmoid)
	fuse_out = tf.multiply(x_g_out,x_l)
	return fuse_out



def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size, stride=[2, 2], activation_fn=None)
	net = tf.nn.relu(slim.batch_norm(net))
	return net

def build_unet(inputs, num_classes):


    #####################
	# Downsampling path #
	#####################
	net = ConvBlock(inputs, 64)
	net = ConvBlock(net, 64)
	skip_1 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = ConvBlock(net, 128)
	net = ConvBlock(net, 128)
	skip_2 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = ConvBlock(net, 256)
	net = ConvBlock(net, 256)
	skip_3 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


	net = ConvBlock(net, 512)
	net = ConvBlock(net, 512)
	skip_4 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


	net = ConvBlock(net, 1024)
	net = ConvBlock(net, 1024)

	# #####################
	# # Upsampling path #
	# #####################
	net = conv_transpose_block(net, 512)
	Attention4_1 = attention_gate(net,skip_4,filter=512)
	net = tf.concat((net, Attention4_1),axis=-1)
	net = ConvBlock(net, 512)
	net = ConvBlock(net, 512)

	net = conv_transpose_block(net, 256)
	Attention3_1 = attention_gate(net,skip_3,filter=256)
	net = tf.concat((net, Attention3_1),axis=-1)
	net = ConvBlock(net, 256)
	net = ConvBlock(net, 256)

	net = conv_transpose_block(net, 128)
	Attention2_1 = attention_gate(net,skip_2,filter=128)
	net = tf.concat((net, Attention2_1),axis=-1)
	net = ConvBlock(net, 128)
	net = ConvBlock(net, 128)


	net = conv_transpose_block(net, 64)
	Attention1_1 = attention_gate(net,skip_1,filter=64)
	net = tf.concat((net, Attention1_1),axis=-1)
	net = ConvBlock(net, 64)
	net = ConvBlock(net, 64)

	net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
	return net
