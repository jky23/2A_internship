import numpy as np
import os, sys
from PIL import Image
import json

class DataSetGeneric():
	def __init__(self,
				path,
				path_images, 
				path_labels, 
				name,format_label,
				label_dataset,
				meta_file='meta-classes.json',
				dict_labels='dict_labels.json'):
		'''
		Parameters:
				path: the path to the dataset (images and labels)
				path_images: the path to the images of the dataset
				path_labels: the path to the labels of the dataset
				name: the name of the dataset
				format_label:
				label_dataset: the label of the dataset
				meta_file: the json file with all the meta-classes
				dict_labels: the json file with NYU-SUNRGBD classes
		'''
		self.path = path
		self.name = name # Name of the dataset
		self.path_images = path_images  # Path to the images of the dataset
		self.path_labels = path_labels  # Path to the labels of the dataset
		self.format_label = format_label # Dataset's labels format
		self.meta_file = meta_file # The file where that describes each possible class and its meta-class
		self.dict_labels = dict_labels # The file of the main classes encounter in NYU, SceneNet datasets
		self.label_dataset = label_dataset # Value associated to each class of the dataset
		self.color_meta = self.ColorMetaClasseDefinition() # Color for each meta-class
		self.equiv_class_meta = self.ProcessMetaJson() # Equivalence between each class and meta-class
		self.meta_labels = self.createLabelMetaClasse() # Value associated to each meta-class of the dataset
		self.equiv_label = self.createEquivLabel()  # Equivalence between the values of the class and those of the meta-class
		
		
	def getPathImage(self,im_path):
		'''
		Return the whole path of an image given its name.
		Args:
			im_path : name of the image
		'''
		# image.png
		return os.path.join(self.path_images,im_path)
	
	def getPathLabel(self,lab_path):
		'''
		Return the whole path of a label given its name.
		Args:
			lab_path : name of the label
		'''
		# label.png
		#lab_path = f"{lab_path}.png"
		return os.path.join(self.path_labels,lab_path)
		
	def getPathGray(self,gray_path):
		'''
		Return the whole path of a grayscale label given its name.
		Args:
			gray_path : name of the label
		'''
		# label.png
		#gris_path = f"{gris_path}_gris.png"
		#gris_path = os.path.join(gris_path,'_gris.png')
		return os.path.join(self.path,'mask_gray',gray_path)
		
	def getPathColor(self,col_path):
		'''
		Return the whole path of a colored label given its name.
		Args:
			col_path : name of the label
		'''
		return os.path.join(self.path,'mask_color',col_path)
		
	def ProcessMetaJson(self) :
		''' 
		Get the equivalences between class and meta-class by reading the json file
		'''
		# Open the meta json file
		with open(self.meta_file) as f :
			meta_classes = json.load(f)
			
		# Open the dict labels file
		with open(self.dict_labels) as f:
			dict_labels = json.load(f)
		
		# Create the dictionnary that will contains the equivalences
		equiv_class = dict()
		for classe in dict_labels['dictionnaire']:
			equiv_class[classe] = meta_classes["classes"][classe]["metaclasse"]
			
		# Writing the equivalence in a file "equiv.json"
		equiv_json = open('equiv.json','w')
		equiv_json.write(json.dumps(equiv_class, indent=4, sort_keys=True))
		equiv_json.close()
		return equiv_class
		
	def createEquivLabel(self):
		''' 
		Create a dictionnary for the equivalences between the labels
		of the class and those of the meta-class.
		If a class does not have a meta-class equivalent, the class is add in 
		a txt file and will receive the meta-label of 'Accessoire'.
		'''
		equiv_label = dict()
		for classe, label in self.label_dataset.items():
			#print(label,classe)
			try:
				equiv_label[label] = self.meta_labels[self.equiv_class_meta[classe.lower()]]
			except KeyError: # If the class is unknown add it in the txt file
				with open("unknowclass.txt",'a') as f:
					f.write(f"{classe}\n")
					f.close()
					equiv_label[label] = 8 # Accessoire
					print(f"Warning : The class {classe} doesn't have an equivalent in the metaclass file..")
				
		return equiv_label
			
		
	def createLabelMetaClasse(self):
		''' 
		Create the labels for each meta-class
		'''
		# Ouverture du fichier json
		with open('meta-classes.json') as f:
			meta_classes = json.load(f)
		
		# Create the dictionnary for the meta-class and their labels
		meta_labels = dict()
		 #i = 0
		for i in range(len(meta_classes["dictionnaire"])):
			meta_labels[meta_classes["dictionnaire"][i]] = i
		
		# Write the equivalences in the file "equiv.json"
		meta_json = open('meta-labels.json','w')
		meta_json.write(json.dumps(meta_labels, indent=4, sort_keys=True))
		meta_json.close()
		
		return meta_labels
		
	def ColorMetaClasseDefinition(self) :
		''' 
		Create the color for each meta-class (RGB Code)
		"sol" : (255,0,0), # Red
		"mur" : (255,255,255), # White
		"plafond" : (0,255,0), # Lime
		"fenetre" : (255,255,0), # Yellow
		"porte" : (0,255,255), # Cyan
		"prise" : (128,0,128), # Purple
		"architecture" : (192,192,192), # Silver
		"meuble" : (128,0,0), # Maroon
		"accessoire" : (255,0,255), # Magenta'''
		
		color_meta = {
		0 : (255,0,0), # Red Sol
		1 : (255,255,255), # White Mur
		2 : (0,255,0), # Lime Plafond
		3 : (255,255,0), # Yellow Fenetre
		4 : (0,255,255), # Cyan Porte
		5 : (128,0,128), # Purple Prise
		6 : (192,192,192), # Silver Architecture
		7 : (128,0,0), # Maroon Meuble
		8 : (255,0,255), # Magenta Accessoire
		}
		
		# Write the equivalence in the file "color-meta.json"
		color_json = open('color-meta.json','w')
		color_json.write(json.dumps(color_meta, indent=4, sort_keys=True))
		color_json.close()
		
		return color_meta
