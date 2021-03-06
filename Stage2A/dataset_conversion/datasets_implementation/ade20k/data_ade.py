import numpy as np
import os, sys
from PIL import Image
from tqdm import tqdm
import json, argparse
import pandas as pd
import data_generic

class Data_ADE(data_generic.DataSetGeneric):
    '''
    Create the dataset labels for Nyu, Sun-RGBD, Uiuc datasets.
    '''

    def __init__(self,
                path,
                create_color=False):
        '''
        Args:
            path: the path of the ADE20K dataset
            create_color: whether the colored label must be created or not (default= False)
        '''
        image_path = os.path.join(path, 'images')
        label_path = os.path.join(path, 'annotations')
        label_dataset = self.get_dataset_label()

        super().__init__(path,image_path,label_path,"ADE20K","RGB",label_dataset)
        
        self.image_names = list(sorted(os.listdir(image_path)))
        self.label_names = list(sorted(os.listdir(label_path)))
        self.create_color = create_color
        #self.image_names_path = [os.path.join(image_path,filename) for filename in self.image_names]
        #self.label_names_path = [os.path.join(label_path,filename) for filename in self.label_names]
        
        # Check if the images folder and labels folder has the same
        assert len(self.image_names) == len(self.label_names)

    def get_dataset_label(self,filename='objectInfo150.csv'):
        '''
        Get the label of each class of the dataset with names.txt files.
        
        Args:
            filename: the csv file that contains each class in the ADE20K Dataset
                        and theirs labels.
                        (default = 'objectInfo150.csv')
        '''
        # Check if the txt file exists
        if not (os.path.isfile(filename)):
            raise Exception('No csv file with the class names !')
        else:
            data = pd.read_csv(filename,names=['Idx','Ratio','Train','Val','Stuff','Name'])
            class_names = data.Name.tolist()
            class_names = [class_names[i].split(';') for i in range(len(class_names))]
            class_names.pop(0)
        
        #print(class_names)
        # Create the dictionnary for equivalence between labels and class names
        dict_labels = {}
        for i in range(len(class_names)):
            if (len(class_names[i]) == 1 ):
                dict_labels[class_names[i][0]] = i+1

            else:
                for j in range(len(class_names[i])):
                    if class_names[i][j] in dict_labels.keys():
                        continue
                    else:
                        dict_labels[class_names[i][j]] = i+1
                #dict_labels[class_names[i][0]] = i+1
        # Add 0 label
        dict_labels['UNKNOWN'] = 0
        print(dict_labels)
        return dict_labels

        #sys.exit(0)


    def load_label_gray_image(self,
                            image_path,
                            gray_path,
                            color_path=None):
        '''
        Create the gray and color image associate to an image of the dataset.
        Parameters:
                image_path: the path to the image to preprocess
                gray_path: the path to the gray image to create
                color_path: the path to the color image to create
                            (default = None)
        '''

        Im = Image.open(image_path,'r')
        # Get the differents label value in the image
        label_values = list(set(Im.getdata()))

        # Create the new image
        Im = np.asarray(Im)
        h,w = Im.shape
        im_gris = np.zeros((h,w))

        if self.create_color:
            im_color = np.zeros((h,w,3),dtype=np.uint8)		
            red = im_color[:,:,0]
            green = im_color[:,:,1]
            blue = im_color[:,:,2]

        for value in label_values:
            meta_label = self.equiv_label[value]

            im_gris[Im == value] = meta_label
            if self.create_color:
                red[Im==value] = np.uint8(self.color_meta[meta_label][0])
                green[Im==value] = np.uint8(self.color_meta[meta_label][1])
                blue[Im==value] = np.uint8(self.color_meta[meta_label][2])

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
        for name in  tqdm(self.label_names, "Processing images"):
            # Get the current relative label filename
            label_filename = name
            print('Converting into a gray image the label {0}'.format(label_filename))
            # Get the absolute filename of color and gray image to create
            gray_filename = self.getPathGray(label_filename)

            color_filename = self.getPathColor(label_filename) if self.create_color else None

            # Get the absolute label_filename
            label_filename = self.getPathLabel(label_filename)

            # Create and save the gray and color image
            self.load_label_gray_image(label_filename,gray_filename,color_path=color_filename)
        print(f'Repository {self.name} converted')

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
    
    data_ade = Data_ADE(args.data_path,args.create_color)
    data_ade.load_label_total()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build the ADE dataset label")
    parser.add_argument("data_path", help="the path of the dataset")
    parser.add_argument("--create-color", type=bool, default=False, help="create the color mask or not \n (default=None)")
    args = parser.parse_args()
    main(args)

