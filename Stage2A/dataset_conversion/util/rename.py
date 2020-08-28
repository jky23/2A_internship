import os, argparse


def rename(folder_path, folder_name):
    '''
    '''
    if os.path.isdir(folder_path):
        files_path = os.listdir(folder_path)
        for files in files_path:
            os.rename(os.path.join(folder_path, files), os.path.join(folder_path, folder_name+'_'+files))
        print('Renaming of folder {} complete'.format(folder_name))
    else:
        raise RuntimeError('This folder {} does not exists'.format(folder_path+folder_name))


def main(args):
    '''
    '''
    # Check if the folder exists
    if os.path.isdir(args.folder_path):
        folder_name = args.folder_path.split('/')[-2]
        # Rename Image folder
        rename(args.folder_path,folder_name)

        # Rename Annotation folder
        #rename(os.path.join(args.folder_path,'mask_gray'),folder_name )
    else:
        raise RuntimeError('This folder does not exists')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename all the files of a folder.')
    parser.add_argument("folder_path", type=str, help='the path of the folder')
    args = parser.parse_args()
    main(args)