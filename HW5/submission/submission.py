# #### INSTRUCTIONS
#
# argument: filename of hdf5 file where the predicted sequence
#           is stored in the key 'v_seq_test'
# output: save the Kaggle formatted output as filename+'.csv'

import numpy as np
import h5py
import argparse
import sys


# Formating the output
def get_kaggle_output(pred_seq, input_matrix, index2tag):
    # #### First loop: build the list of results by line
    kaggle_output = []
    current_line = []
    for r_input, r_pred in zip(input_matrix, pred_seq):
        # Start of a new line
        if (r_input[1] == 1):
            kaggle_output.append(current_line)
            current_line = []
        else:
            if (r_pred > 1) and (r_pred < 8):
                current_line.append((r_input[1] - 1, r_pred))
    kaggle_output.append(current_line)
    # Remove first element
    kaggle_output = kaggle_output[1:]

    # #### Second loop: Format the result of each line

    kaggle_output_new = []
    for i, k in enumerate(kaggle_output):
        if len(k):
            current_seq = ''
            seq = ''
            prev_tag = ''
            prev_ind = -1
            for u in k:
                ind, tag_ind = u
                tag = index2tag[tag_ind][2:]
                # Growing the current tag sequence
                if tag == prev_tag and ind == prev_ind+1:
                    seq = seq+'-'+str(ind)
                # New tag
                else:
                    if len(seq):
                        current_seq += seq+' '
                    seq = tag+'-'+str(ind)
                    prev_tag = tag
                prev_ind = ind
            # adding remaining element
            current_seq += seq
            kaggle_output_new.append((i+1, current_seq))
        else:
            kaggle_output_new.append((i+1, ''))

    # Output: list of tuple (line_index, line_output)
    return kaggle_output_new


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('pred', type=str, help='Filename with pred in h5')
    args = parser.parse_args(arguments)
    filename = args.pred

    # Loading the prediction
    with h5py.File(filename, "r") as f:
        # Model
        v_seq = np.array(f.get('v_seq_test'), dtype=int)

    # Loading the input data (to have the line recording)
    filename_input = '../data/words_feature.hdf5'
    with h5py.File(filename_input, "r") as f:
        # Input
        input_matrix_test = np.array(f.get('input_matrix_test'), dtype=int)

    # Loading and formating the tag
    tag2index = {}
    with open('../data/tags.txt', 'r') as f:
        for line in f:
            line_split = line[:-1].split(' ')
            tag2index[line_split[0]] = int(line_split[1])

    index2tag = {v: k for k, v in tag2index.iteritems()}

    output = get_kaggle_output(v_seq, input_matrix_test, index2tag)

    # Write to file
    with open(filename + '.csv', 'w') as f:
        f.write('ID,Labels\n')
        for line in output:
            f.write(str(line[0]) + ',' + line[1] + '\n')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
