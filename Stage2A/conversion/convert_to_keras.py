#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:24:09 2020

@author: jky
"""
from caffe.proto import caffe_pb2
from PIL import Image
import google.protobuf.text_format
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import sys,argparse
from tensorflow import keras
from utils import *
import convert_weight


class Model_Convertor():
	'''
	Class which creates a Keras model from a pre-trained Caffe model.
	'''
	
	def __init__(self,prototxt_pathfile, caffemodel_pathfile,is_weight_converted,model_name,debug):
		'''
		Class constructor.
		Parameters:
				prototxt_pathfile : the path to the file "my_modele.prototxt"
				caffemodel_pathfile : the path to the file "my_modele.caffemodel"
				is_weight_converted : a boolean to see if the weight of the model 
									 has already been converted
				model_name : the name of the model to convert
				debug : a boolean to print or not information while conversion
		'''
		self.proto_path = prototxt_pathfile
		self.caffemodel_path = caffemodel_pathfile
		self.model_name = model_name
		self.keras_layers_list = []
		self.net = caffe_pb2.NetParameter()
		f = open(self.proto_path,'r')
		self.net = google.protobuf.text_format.Merge(str(f.read()), self.net)
		f.close()
		
		if len(self.net.layers) != 0:
			raise Exception("Prototxt files V1 are not supported.")
			layers = self.net.layers[:]  # prototext V1
		elif len(self.net.layer) != 0:
			layers = self.net.layer[:]  # prototext V2
		else:
			raise Exception('could not load any layers from prototext')
				
		input_dim = []
		print(self.net.input_shape)
		print(self.net.input_shape[0])
		print(self.net.input_shape[0].dim[:])
		if len(self.net.input_shape[0].dim[:]) == 0:
			print(layers[0])
			#print(layers[0].memory_data_param)
			input_dim.append(int(layers[0].memory_data_param.batch_size))
			input_dim.append(int(layers[0].memory_data_param.height))
			input_dim.append(int(layers[0].memory_data_param.width))
			input_dim.append(int(layers[0].memory_data_param.channels))
			#input_dim.append(int(layers[0].input_param.shape[0].dim[0]))
			#input_dim.append(int(layers[0].input_param.shape[0].dim[2]))
			#input_dim.append(int(layers[0].input_param.shape[0].dim[3]))
			#input_dim.append(int(layers[0].input_param.shape[0].dim[1]))

		else:
			input_dim = tuple(self.net.input_shape[0].dim[:])
			input_dim = (input_dim[0],input_dim[2],input_dim[3],input_dim[1])
		print(input_dim)
		print("CREATING MODEL")
		
		if not(is_weight_converted): # Convert the weight of the model.
			print("Converting the weights....") 
			self.convert_weight()
			print("Weight of the model successfully converted !")
			
		print("Converting the layers....")
		self.model = self.create_model(layers,tuple(input_dim[1:]), debug)
		self.model.summary()
		print("Layers of the model successfully converted !")
		#batch = self.model.get_layer('conv5_3_pool1_conv/bn')
		#print("Batch shape",np.shape(batch.get_weights()[0]))
		

			
		# Load the weights in the model and save the model
		self.model.load_weights(f"weights_{model_name}.h5",by_name = True)
		#tf.saved_model.save(self.model,'/home/jky/')
		self.model.save(f"{model_name}.h5")
		print(f"Model converted and save to the file {self.model_name}.h5 in the current folder.")
			
	def convert_weight(self):
		'''
		Call the fonction from 'convert_weight.py' to convert the weights of the model
		from Caffe to Keras.
		Make python3 convert_weight.py -h to see more.
		'''
		convert_weight.convert_caffemodel_to_keras(f"weights_{self.model_name}",self.proto_path,self.caffemodel_path,False,True,'tf',False)
		

	def createKerasConvLayer(self,layer,input_layers,debug):
		'''
		Create a Keras Convolutional layer from a convolutional layer
		 definition in Caffe (prototxt).
		 Parameters:
				layer : convolutional layer in Caffe format
				input_layers : the preceding layer
		'''
		name = layer.name
		has_bias = layer.convolution_param.bias_term
		nb_filter = layer.convolution_param.num_output
		nb_col = (layer.convolution_param.kernel_size or [layer.convolution_param.kernel_h])[0]
		nb_row = (layer.convolution_param.kernel_size or [layer.convolution_param.kernel_w])[0]
		stride_h = (layer.convolution_param.stride or [layer.convolution_param.stride_h])[0] or 1
		stride_w = (layer.convolution_param.stride or [layer.convolution_param.stride_w])[0] or 1
		pad_h = (layer.convolution_param.pad or [layer.convolution_param.pad_h])[0] 
		pad_w = (layer.convolution_param.pad or [layer.convolution_param.pad_w])[0]
		

		if debug:
			print("\t kernel: " + str(nb_filter) + 'x' + str(nb_col) + 'x' + str(nb_row))
			print("\t stride: " + str(stride_h))
			print("\t pad_h: " + str(pad_h))
			print("\t pad_w:" + str(pad_w))
			print("\t inputs", input_layers)
			print("\t has bias", + bool(has_bias))
			#print("\t inputs", input_layers.shape)
		#if pad_h + pad_w > 0:
			#input_layers = keras.layers.ZeroPadding2D(padding=(int(pad_h), int(pad_w)), name=name + '_zeropadding')(input_layers)
		if (layer.convolution_param.dilation or [layer.convolution_param.dilation])[0]:
			dilation = layer.convolution_param.dilation[0]
			print("\t dilation" + str(dilation))
			return keras.layers.Conv2D(nb_filter, (int(nb_row), int(nb_col)), use_bias=bool(has_bias),
											strides=(stride_h, stride_w), name=name, padding='same', dilation_rate=(int(dilation),int(dilation)))(input_layers)
		else:
			return keras.layers.Conv2D(nb_filter, (int(nb_row), int(nb_col)), use_bias=bool(has_bias),
											strides=(stride_h, stride_w), name=name, padding='same')(input_layers)

		
		
	def createKerasReluLayer(self,layer,input_layers):
		'''
		Create a Keras ReLu layer from a convolutional layer
		 definition in Caffe (prototxt).
		 Parameters:
				layer : ReLu layer in Caffe format
				input_layers : the preceding layers
		'''
		name = layer.name
		return keras.layers.Activation('relu',name=name)(input_layers)
		#return keras.layers.Lambda(lambda x : x,name=name)(input_layers)
		
	def createKerasInterpLayer(self,layer,input_layers,debug):
		'''
		Create a Keras interpretation of Caffe Interp layer.
		Parameters:
				layer: Interp layer in Caffe format
				input_layers: the preceding layer
		'''
		name = layer.name
		new_height = layer.interp_param.height
		new_width = layer.interp_param.width
		
		if debug:
			print("\t height: " + str(new_height))
			print("\t width:" + str(new_width))
			print("\t inputs", input_layers)
			
		#upsample_factor_heigh = new_height/input_layers.shape[1]
		#upsample_factor_width = new_width/input_layers.shape[2]
		
		#return keras.layers.UpSampling2D(size=(upsample_factor_heigh,upsample_factor_width),name=name)(input_layers)
		return keras.layers.Lambda(lambda image: tf.image.resize(image,(new_height,new_width)),name=name) (input_layers)
			
		#return keras.layers.Lambda(lambda image: tf.image.resize(image,(new_height,new_width),preserve_aspect_ratio=True,
    #antialias=True),name=name) (input_layers)
		#return keras.layers.Lambda(lambda image : Image.resize(image))
			
		
		
	def createKerasConcatLayer(self,layer,input_layers,debug):
		'''
		Create a Keras Concat layer from a concat layer
		 definition in Caffe (prototxt).
		 Parameters:
				layer : Concat layer in Caffe format
				input_layers : the preceding layers
		'''
		
		axis = layer.concat_param.axis
		name = layer.name
		if debug:
			print("\t axis ",axis)
			for i in range (len(input_layers)):
				print(f"\t inputs :{i}, shape ={input_layers[i].shape}")
		return keras.layers.concatenate(input_layers,name=name)
		
		
	def createKerasDropoutLayer(self,layer,input_layers,debug):
		'''
		Create a Keras Dropout layer from a dropout layer
		 definition in Caffe (prototxt).
		 Parameters:
				layer : dropout layer in Caffe format
				input_layers : the preceding layer
		'''
		name = layer.name
		prob = layer.dropout_param.dropout_ratio
		if debug :
			print("\t rate ",prob)
		return keras.layers.Dropout(prob, name=name)(input_layers)
		
		
	def createKerasPoolingLayer(self,layer,input_layers,debug):
		'''
		Create a Keras Pooling layer from a Pooling layer
		 definition in Caffe (prototxt).
		 Parameters:
				layer : Pooling layer in Caffe format
				input_layers : the preceding layer
		'''
		name = layer.name
		kernel_h = layer.pooling_param.kernel_size or layer.pooling_param.kernel_h
		kernel_w = layer.pooling_param.kernel_size or layer.pooling_param.kernel_w

		# caffe defaults to 1, hence both of the params can be zero. 'or 1'
		stride_h = layer.pooling_param.stride or layer.pooling_param.stride_h or 1
		stride_w = layer.pooling_param.stride or layer.pooling_param.stride_w or 1

		pad_h = layer.pooling_param.pad or layer.pooling_param.pad_h or 0
		pad_w = layer.pooling_param.pad or layer.pooling_param.pad_w or 0

		if debug:
			print("\t kernel: " + str(kernel_h) + 'x' + str(kernel_w))
			print("\t stride: " + str(stride_h))
			print("\t pad_h: " + str(pad_h))
			print("\t pad_w:" + str(pad_w))
			print("\t inout shape" + str(input_layers.shape))
			
		_,h,w,_ = input_layers.shape

		if h % 2 != 0 or w % 2 != 0:
			#paddings = tf.constant([[0,0],[pad_h,pad_h],[pad_w,pad_w],[0,0]])
			input_layers = keras.layers.Lambda(lambda x : tf.pad(x,[[0,0],[0,pad_h],[0,pad_w],[0,0]]))(input_layers)
		else:
			input_layers = keras.layers.Lambda(lambda x : tf.pad(x,[[0,0],[pad_h,pad_h],[pad_w,pad_w],[0,0]]))(input_layers)
			#input_layers = keras.layers.ZeroPadding2D(padding=(int(pad_h), int(pad_w)), name=name + '_zeropadding')(input_layers)
			#input_layer_name = name + '_zeropadding'
		if layer.pooling_param.pool == 0:  # MAX pooling
			border_mode = 'same'
			#border_mode = 'valid'
			return keras.layers.MaxPooling2D(pool_size=(kernel_h, kernel_w), strides=(stride_h, stride_w),
											  padding=border_mode, name=name)(input_layers)
			if debug:
				print("\t MAX pooling")
		elif layer.pooling_param.pool == 1:  # AVE pooling
			border_mode = 'same'
			return keras.layers.AveragePooling2D(pool_size=(kernel_h, kernel_w), strides=(stride_h, stride_w),padding=border_mode,
												  name=name)(input_layers)
			if debug:
				print("\t AVE pooling")
		else:  # STOCHASTIC?
			raise Exception("This pooling layer is unknown ! \n Only MAX and AVE pooling are implemented in keras!")
		
	def createKerasSoftmaxLayer(self,layer,first_layer,input_layers):
		'''
		Create a Keras SoftMax layer from a SoftMax layer
		 definition in Caffe (prototxt).
		 Parameters:
				layer : SoftMax layer in Caffe format
				first_layer : the first layer of the model
				input_layers : the precedents layers
		'''
		name = layer.name

		# check output shape
		semi_model = keras.Model(inputs=first_layer, outputs=input_layers)
		op_shape = semi_model.layers[-1].output_shape
		#op_shape = input_layers.shape
		del semi_model
		#print("op_shape",op_shape)

		if len(op_shape) == 4:  # for img segmentation - i/p to softmax is (None, height, width,channels)
			print("op_shape",op_shape)
			#interm_layer = keras.layers.Flatten()(input_layers)
			#interm_layer = keras.layers.Reshape((op_shape[3],op_shape[1] * op_shape[2]))(input_layers)
			#print("interm_layer")
			#input_layers = keras.layers.Permute((2, 1))(interm_layer)  # reshaped to (None, height*width, channels)
			#interm_layer = keras.layers.Dense(op_shape[1]*op_shape[2]*op_shape[3])(input_layers)
			#soft_layer = keras.layers.Reshape((op_shape[1] * op_shape[2],op_shape[3]))(interm_layer)
			soft_layer = keras.layers.Activation('softmax',name=name)(input_layers)
			#soft_layer = keras.layers.Reshape((op_shape[1],op_shape[2],op_shape[3]))(soft_layer)
			#soft_layer = keras.layers.Conv2D(op_shape[3],,activation='softmax',use_bias=False)(input_layers)

		return soft_layer

		
	def createKerasBatchnormLayer(self,layer,scale_layer,input_layers,debug):
		'''
		Create a Keras BatchNorm layer from a BatchNorm layer and a Scale layer
		 definition in Caffe (prototxt).
		 In Keras BatchNorm layer applies the norm and the scale operation in one layer
		 while in Caffe this is made with 2 layers.
		 We need to process the two layers at the same moment.
		 Parameters:
				layer : BatchNorm layer in Caffe format
				scale_layer : Scale layer in Caffe format
				input_layers: the preceding layer
				debug : Boolean to print or not the information during the process
		'''
		name = layer.name
		if layer_type(scale_layer) != 'scale':
			raise Exception(f"The BatchNorm layer {name} is not followed by a Scale layer")
		else:
			
			axis = scale_layer.scale_param.axis
			bias_term = scale_layer.scale_param.bias_term
			epsilon = layer.batch_norm_param.eps
			moving_average = layer.batch_norm_param.moving_average_fraction  # unused
			batch = keras.layers.BatchNormalization(epsilon=epsilon, name=name)(input_layers)

			if debug:
				print('\t -- Scale')
				print('\t axis: ' + str(axis))
				#for var in batch.variables:
					#print(f"\t name {var.name} shape {var.shape}")
				#[(var.name, var.trainable,var.shape) for var in bn1.variables]

			return batch

	
	def createKerasScaleLayer(self,layer,batch_layer_type,input_layers):
		'''
		Create a Keras Scale layer from a Scale layer
		 definition in Caffe (prototxt).
		 The conversion has been made with the BatchNorm layer, this function ensures
		 that the scale layer was preced by a BatchNorm layer and just call an Identity layer 
		 Parameters:
				layer : Scale layer in Caffe format
				batch_layer_type : the type of the layer that preceds the scale layer
				input_layers : the preceding layers
				
		'''
		name = layer.name
		if batch_layer_type != 'batchnorm':
			raise Exception(f"The Scale layer {name} was not preced by a BatchNorm layer")
		
		return keras.layers.Lambda(lambda x : x,name=name)(input_layers)
		#return input_layers

	
	def create_model(self,layers,input_dim, debug=False):
		'''
			layers:
				a list of all the layers in the model
			phase:
				parameter to specify which network to extract: training or test
			input_dim:
				`input dimensions of the configuration (if in model is in deploy mode)
		'''
		if input_dim == ():
			in_deploy_mode = False
		else:
			in_deploy_mode = True
		
		# obtain the nodes that make up the graph
		# returned in linked list (not matrix) representation (dictionary here)
		phase = 1 # 0 train 1 test 
		network = parse_network(layers, phase)
		if len(network) == 0:
			raise Exception('failed to construct network from the prototext')
			
		# inputs of the network - 'in-order' is zero
		inputs = get_inputs(network)
		# outputs of the network - 'out-order' is zero
		network_outputs = get_outputs(network)
		
		# path from input to loss layers (label) removed
		network = remove_label_paths(layers, network, inputs, network_outputs)

		# while network contains what nodes follow a particular node.
		# we need to know what feeds a given node, hence reverse it.
		inputs_to = reverse(network)

		# create all net nodes without link
		net_node = [None] * (max(network) + 1)
		#net_node[0] = keras.Input(shape=input_dim)
		for n_layer, layer_nb in enumerate(network):
			layer = layers[layer_nb]
			name = layer.name
			type_of_layer = layer_type(layer)
			#print("numero layer",n_layer)
			#print("Layer",type_of_layer)

			# case of inputs
			#if False:
			if layer_nb in inputs:
				if in_deploy_mode:
					dim = input_dim
				else:
					# raise Exception("You must define the 'input_dim' of your network at the start of your .prototxt file.")
					dim = get_data_dim(layers[0])
				#print("Layers dim",dim)
				#sys.exit(2)
				net_node[layer_nb] = keras.Input(shape=dim, name=name)
				print("Layer nb = ",layer_nb)

			# other cases
			else:
				input_layers = [None] * (len(inputs_to[layer_nb]))
				for l in range(0, len(inputs_to[layer_nb])):
					input_layers[l] = net_node[inputs_to[layer_nb][l]]
					print("input layers = ",input_layers[l])

				# input_layers = net_node[inputs_to[layer_nb]]
				input_layer_names = []
				for input_layer in inputs_to[layer_nb]:
					input_layer_names.append(layers[input_layer].name)

				if debug:
					print("Layer", str(n_layer) + ":", name)
					print('\t input shape: ' + str(input_layers[0].shape))

				if type(input_layers) is list and len(input_layers) == 1:
					input_layers = input_layers[0]
				
				if type_of_layer == 'memorydata':
					print('memorydata',input_layers)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
				elif type_of_layer == 'convolution':
					net_node[layer_nb] = self.createKerasConvLayer(layer,input_layers,debug)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
				elif type_of_layer == 'relu':
					net_node[layer_nb] = self.createKerasReluLayer(layer,input_layers)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
				elif type_of_layer == 'pooling':
					net_node[layer_nb] = self.createKerasPoolingLayer(layer,input_layers,debug)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
				elif type_of_layer == 'interp':
					net_node[layer_nb] = self.createKerasInterpLayer(layer,input_layers,debug)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
				elif type_of_layer == 'batchnorm':
					scale_layer = layers[layer_nb+1]
					net_node[layer_nb] = self.createKerasBatchnormLayer(layer,scale_layer,input_layers,debug)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
				elif type_of_layer == 'scale':
					# The scale layer has been processed with the bachnorm layer
					# So we will just put an Identity layer 
					batch_layer_type = layer_type(layers[layer_nb -1])
					net_node[layer_nb] = self.createKerasScaleLayer(layer,batch_layer_type,input_layers)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
				elif type_of_layer == 'concat':
					net_node[layer_nb] = self.createKerasConcatLayer(layer,input_layers,debug)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
				elif type_of_layer == 'dropout':
					net_node[layer_nb] = self.createKerasDropoutLayer(layer,input_layers,debug)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
				elif type_of_layer == 'softmax':
					net_node[layer_nb] = self.createKerasSoftmaxLayer(layer,net_node[0],input_layers)
					print(f"The {type_of_layer} layer {n_layer} : has been processed.")
									

				else:
					raise Exception('layer type', type_of_layer, 'used in this model is not currently supported')

		input_l = [None] * (len(inputs))
		output_l = [None] * (len(network_outputs))

		for i in range(0, len(inputs)):
			input_l[i] = net_node[inputs[i]]
		for i in range(0, len(network_outputs)):
			output_l[i] = net_node[network_outputs[i]]
			
		print(input_l)
		print(output_l)

		model = keras.Model(inputs=input_l, outputs=output_l)
		return model

def main(argv):
	'''
	Convert the model with the arguments of command line.
	'''
	my_modele = Model_Convertor(argv.prototxt,argv.caffemodel,False,argv.model_name,argv.verbose)
	
	
if __name__=='__main__':
	parser = argparse.ArgumentParser(description=('Convert a PSPNet model from the Caffe format to Tensorflow/Keras format'))
	parser.add_argument('prototxt',action='store',type=str,help='the path to the prototxt file')
	parser.add_argument('caffemodel',action='store',type=str,help='the path to the caffemodel file')
	parser.add_argument('model_name',action='store',type=str,help='the name of the model. It will be use to create the file where the model will be save')
	parser.add_argument('-v', '--verbose',action='store_true',default='False',help='Print the information while conversion')
	#parser.add_argument('')
	args = parser.parse_args()
	main(args)
	
	
