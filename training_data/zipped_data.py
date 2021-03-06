import argparse
import os
from random import shuffle
import functools
from data_util import get_valid_ids

parser = argparse.ArgumentParser()
parser.add_argument('folder_paths', nargs='+', type=str, help='The folder path')
parser.add_argument('--name', default='', type=str, help='output name')
parser.add_argument('--outdir', default='', type=str, help='output path')
parser.add_argument('--shuffle', action='store_true', help='shuffle the dataset')
parser.add_argument('--split', default=0.05, type=float, help='percentage of the dataset to use for validation')
parser.add_argument('--suffixes', nargs='+', type=str, help='suffix of files to choose for dataset')
parser.add_argument('--split_textures', nargs='?', const=0, type=int, help='ensure that validation and training set use separate texture based on filenames (with optional index specifying which set of filenames to inspect for texture names)')

if __name__ == "__main__":

    args = parser.parse_args()

    # make 2 lists to save file paths

    fnamelists = []

    suffixes = ['']*len(args.folder_paths) if args.suffixes is None else args.suffixes
    while len(suffixes) < len(args.folder_paths):
        suffixes.append('')

    fnamelists = [[item.name for item in os.scandir(path) if item.name.endswith(suffix)] for path, suffix in zip(args.folder_paths, suffixes)]
    for path, fnamelist in zip(args.folder_paths, fnamelists):
        print('original length of %s:' % path, len(fnamelist))

    id_sets = [{fname.split('_')[0] for fname in fnames} for fnames in fnamelists]
    ids = functools.reduce(lambda x,y: x.intersection(y), id_sets)
    
    fnamelists = [[item.name for item in os.scandir(path) if item.name.endswith(suffix) and item.name.split('_')[0] in ids] for path, suffix in zip(args.folder_paths, suffixes)]
    for path, fnamelist in zip(args.folder_paths, fnamelists):
        print('length of %s:' % path, len(fnamelist))
        fnamelist.sort()

    #get the ids of filenames satisfying the validation set constraints
    if args.split_textures is not None:
        valid_ids = get_valid_ids(fnamelists[args.split_textures], args.split)


    n = len(fnamelists[0])
    split = args.split * n
    validation_file_names = []
    training_file_names = []
    for i, items in enumerate(zip(*fnamelists)):
        nids = []
        paths = []
        for path, item in zip(args.folder_paths, items):
            nid = item.split('_')[0]
            nids.append(nid)
            item = os.path.join(path, item)
            paths.append(item)
        #assert all IDS match across pairs
        for nid in nids[1:]:
            if nid != nids[0]:
                print('invalid pairing')
                exit()
        if args.split_textures is None:
            if i < split:
                validation_file_names.append(paths)
            else:
                training_file_names.append(paths)
        else:
            if nids[0] in valid_ids:
                validation_file_names.append(paths)
            else:
                training_file_names.append(paths)
        

    # print all file paths
    #for i in training_file_names:
    #    print(i)
    #for i in validation_file_names:
    #    print(i)

    # shuffle file names if set
    if args.shuffle:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    train_filename = os.path.join(args.outdir, 'train_' + args.name + ('_splittex' if args.split_textures is not None else '') + ('_shuffled' if args.shuffle else '') + '.flist')
    validation_filename = os.path.join(args.outdir, 'validation_' + args.name + ('_splittex' if args.split_textures is not None else '') + ('_shuffled' if args.shuffle else '') + '.flist')
    
    if not os.path.exists(train_filename):
        os.mknod(train_filename)

    if not os.path.exists(validation_filename):
        os.mknod(validation_filename)

    # write to file
    fo = open(train_filename, "w")
    for f in training_file_names:
        for fi in f:
            fo.write(fi + ' ')
        fo.write('\n')
    fo.close()

    fo = open(validation_filename, "w")
    for f in validation_file_names:
        for fi in f:
            fo.write(fi + ' ')
        fo.write('\n')
    fo.close()

    # print process
    print("Written files: %s (%d) and %s (%d)" % (train_filename, len(training_file_names), validation_filename, len(validation_file_names)), ", shuffle: ", args.shuffle)
