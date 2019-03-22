import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data_flist/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation_shuffled.flist', type=str,
                    help='The validation filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled')
parser.add_argument('--train_subdir', default='training', type=str, help='The training subdirectory name')
parser.add_argument('--validation_subdir', default='validation', type=str, help='The validation subdirectory name')
parser.add_argument('--split', default=None, type=float, help='percentage of the dataset to use for validation')
parser.add_argument('--suffix', default=None, type=str, help='suffix of files to choose for dataset')
parser.add_argument('--prefix', default=None, type=str, help='prefix of files to choose for dataset')

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
        folder = os.listdir(args.folder_path)
        if args.suffix is not None:
            folder = [item for item in folder if item.endswith(args.suffix)]
        if args.prefix is not None:
            folder = [item for item in folder if item.startswith(args.prefix)]
        n = len(folder)
        split = args.split * n
        for i, item in enumerate(folder):
            item = os.path.join(args.folder_path, item)
            if i < split:
                validation_file_names.append(item)
            else:
                training_file_names.append(item)
        

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
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)
