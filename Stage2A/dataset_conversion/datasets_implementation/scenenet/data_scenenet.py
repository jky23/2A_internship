import numpy as np
import os, sys
from PIL import Image
from tqdm import tqdm
import scenenet_pb2 as sn
import json, argparse
import data_generic

class Data_SceneNet(data_generic.DataSetGeneric):
	'''
	   Create the dataset's labels for the dataset SceneNet
	'''
	def __init__(self,
				path,
				protobuf_path,
				train_rep,
				create_color=False):
		'''
		Parameters:
				path: the path to the dataset
				protobuf_path: the path to the protobuf file
				train_rep: the number of the train folder and the folder to preprocess.
				create_color: create the colored labels or not
		'''
		label_dataset = {'Unknown':0,'Bed':1,'Books':2,'Ceiling':3,'Chair':4,'Floor':5,
		'Furniture':6,'Objects':7,'Picture':8,'Sofa':9,'Table':10,'TV':11,'Wall':12,'Window':13}
		path_images = os.path.join(path, 'photo')
		path_labels = os.path.join(path, 'instance')
		super().__init__(path,path_images,path_labels,"SceneNet","Images_RGB",label_dataset)
		self.path_proto = protobuf_path # Path to the protobuf file
		self.train_rep = train_rep # Sub-repository
		self.create_color = create_color
		
		
		with open('NYU_WNID_TO_CLASS.json') as f :
			self.trad_value_label = json.load(f)
		
	def load_label_gray_image(self,image_path,gray_path,color_path,mapping):
		'''
		Create the gray and color image associate to an image of the dataset.
		Parameters:
				image_path: the path to the image to preprocess
				gray_path: the path to the gray image to create
				color_path: the path to the color image to create
				mapping: a dict file for mapping instance and label
		'''
		# Create the new image
		Im = np.asarray(Image.open(image_path,'r'))
		h,w = Im.shape
		im_gris = np.zeros((h,w))
		im_color = np.zeros((h,w,3),dtype=np.uint8)
		
		red = im_color[:,:,0]
		green = im_color[:,:,1]
		blue = im_color[:,:,2]
		#Im = Im.convert('L')
		for instance, semantic_class in mapping.items():
			#print("semantic",semantic_class)
			meta_label = self.equiv_label[semantic_class]
			#print(self.color_meta[meta_label],"meta")
			im_gris[Im == instance] = meta_label
			if self.create_color:
				red[Im==instance] = np.uint8(self.color_meta[meta_label][0])
				green[Im==instance] = np.uint8(self.color_meta[meta_label][1])
				blue[Im==instance] = np.uint8(self.color_meta[meta_label][2])

		if self.create_color:	
			im_color[:,:,0] = red
			im_color[:,:,1] = green
			im_color[:,:,2] = blue
		
		# Save the image in their folders.
		im_gris = Image.fromarray(np.uint8(im_gris))
		im_gris.save(gray_path)

		if self.create_color:
			im_color = Image.fromarray(im_color)
			im_color.save(color_path)
		#print(image_path)
		
	def load_label_total(self):
		'''
		Preprocess the whole dataset.
		'''
		trajectories = sn.Trajectories()
		try:
			with open(self.path_proto,'rb') as f:
				trajectories.ParseFromString(f.read())
		except IOError:
			print("The scenenet.proto file is not find at the following path:{0}".format(self.path_proto))
			print('Please check that you have copied the file in the righ repository')
			sys.exit(1)
		
		traj = [t for t in trajectories.trajectories if t.render_path == self.train_rep][0]
		instance_class_map = {}
		for instance in traj.instances:
			instance_type = sn.Instance.InstanceType.Name(instance.instance_type)
			
			if instance.instance_type != sn.Instance.BACKGROUND:
				instance_class_map[instance.instance_id] = self.trad_value_label[instance.semantic_wordnet_id]
				
		for view in tqdm(traj.views, "Processing images"):
			lab_path = f"{view.frame_num}"
			gray_path = self.getPathGray(f"{lab_path}.png")
			color_path = self.getPathColor(f"{lab_path}.png")
			instance_path = self.getPathLabel(f"{lab_path}.png")
			#instance_path = instance_path_from_view(traj.render_path,view)
			print('Converting into a gray image the label {0}'.format(lab_path))
			
			self.load_label_gray_image(instance_path,gray_path,color_path,instance_class_map)

		print('Repository {0} converted'.format(self.path))

def main(args):
	'''
	Parameters:
			args: arguments from command line
	'''
	# Create the file for saving the images
	try:
		seg_path = os.path.join(args.data_path,'mask_gray')
		os.mkdir(seg_path)

		if args.create_color:
			color_path = os.path.join(args.data_path,'mask_color')
			os.mkdir(color_path)
	except FileExistsError:
		print("Warning : the file already exists ")
	
	data_scene = Data_SceneNet(args.data_path,args.protobuf_path,args.train_rep,args.create_color)
	data_scene.load_label_total()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Build the SceneNet dataset label")
	parser.add_argument("data_path", help="the path of the dataset")
	parser.add_argument("protobuf_path", help="the path of the protobuf file")
	parser.add_argument("train_rep", help="the path of the subfile where the data is.\n For example if you are preprocessing rep train5 and the subfile 10, put 5/10")
	parser.add_argument("--create-color", type=bool, default=False, help="create the color mask or not")
	args = parser.parse_args()
	main(args)
