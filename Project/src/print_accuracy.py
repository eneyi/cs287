import numpy as np
import h5py
import argparse
import sys

# Script to print accuracy saved by mem2_pe.lua
# Example of use:
# $ python print_accuracy.py -acc task123_3hops_1adjacent_pe1_1.acc_by_task.hdf5


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-acc', type=str, help='Filename with accuracy in h5')
    args = parser.parse_args(arguments)
    filename = args.acc

    with h5py.File("accuracies/"+filename, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        accuracy_by_task_train = hf.get('train')
        accuracy_by_task_train = np.array(accuracy_by_task_train, dtype=float)
        accuracy_by_task_test = hf.get('test')
        accuracy_by_task_test = np.array(accuracy_by_task_test, dtype=float)

    print('Train accuracy TOTAL ' + str(np.mean(accuracy_by_task_train[:, 1])))
    print('Train accuracy by task')
    print(accuracy_by_task_train)
    print('\n')
    print('***************************************************')

    print('Test accuracy TOTAL ' + str(np.mean(accuracy_by_task_test[:, 1])))
    print('Test accuracy by task')
    print(accuracy_by_task_test)
    print('\n')
    print('***************************************************')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
