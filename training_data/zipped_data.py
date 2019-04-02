import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path')
parser.add_argument('--mask_path', default=None, type=str, help='mask path')
parser.add_argument('--train_filename', default='./data_flist/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation_shuffled.flist', type=str,
                    help='The validation filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled')
parser.add_argument('--train_subdir', default='training', type=str, help='The training subdirectory name')
parser.add_argument('--validation_subdir', default='validation', type=str, help='The validation subdirectory name')
parser.add_argument('--split', default=None, type=float, help='percentage of the dataset to use for validation')
parser.add_argument('--suffix', default='', type=str, help='suffix of files to choose for dataset')

if __name__ == "__main__":

    args = parser.parse_args()

    # make 2 lists to save file paths
    training_file_names = []
    validation_file_names = []
    
    if args.split is None:
        # get the list of directories and separate them into 2 types: training and validation
        training_dirs = os.listdir(os.path.join(args.folder_path, args.train_subdir))
        validation_dirs = os.listdir(os.path.join(args.folder_path, args.validation_subdir))

        # append all files into 2 lists
        for training_dir in training_dirs:
            # append each file into the list file names
            training_folder = os.listdir(os.path.join(args.folder_path, args.train_subdir, training_dir))
            for training_item in training_folder:
                # modify to full path -> directory
                if training_item[-3:] != 'xml' and training_item[-3:] != 'mip':
                    training_item = os.path.join(args.folder_path, args.train_subdir, training_dir, training_item)
                    training_file_names.append(training_item)

                    # append all files into 2 lists
        for validation_dir in validation_dirs:
            # append each file into the list file names
            validation_folder = os.listdir(os.path.join(args.folder_path, args.validation_subdir, validation_dir))
            for validation_item in validation_folder:
                # modify to full path -> directory
                if validation_item[-3:] != 'xml' and validation_item[-3:] != 'mip':
                    validation_item = os.path.join(args.folder_path, args.validation_subdir, validation_dir, validation_item)
                    validation_file_names.append(validation_item)
    else:
        folder = [item.name for item in os.scandir(args.folder_path) if item.name.endswith(args.suffix)]
        masks = [mask.name for mask in os.scandir(args.mask_path)]
        ids = {file.split('_')[0] for file in folder}.intersection({mask.split('_')[0] for mask in masks})
        images = [d.name for d in os.scandir(args.folder_path) if d.name.endswith(args.suffix) and d.name.split('_')[0] in ids]
        masks = [d.name for d in os.scandir(args.mask_path) if d.name.split('_')[0] in ids]
        images.sort()
        masks.sort()
        

        n = len(folder)
        split = args.split * n
        i=0
        for item, mask in zip(images, masks):
            item = os.path.join(args.folder_path, item)
            mask = os.path.join(args.mask_path, mask)
            if i < split:
                validation_file_names.append((item, mask))
            else:
                training_file_names.append((item, mask))
            i+=1
        

    # print all file paths
    for i in training_file_names:
        print(i)
    for i in validation_file_names:
        print(i)

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    for f in training_file_names:
        fo.write(f[0] + ' ' + f[1])
        fo.write('\n')
    fo.close()

    fo = open(args.validation_filename, "w")
    for f in validation_file_names:
        fo.write(f[0] + ' ' + f[1])
        fo.write('\n')
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)
