import numpy as np
import h5py
import argparse
import sys


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('pred', type=str, help='Filename with pred in h5')
    parser.add_argument('--f', default='new_submission', type=str, help='Filename with predictions in csv')
    args = parser.parse_args(arguments)
    filename = args.pred

    with h5py.File("submission/"+filename, 'r') as hf:
        data = hf.get('num_spaces')
        np_data = np.array(data, dtype=int)

    np.savetxt("submission/{}.csv".format(args.f), np_data, delimiter=",",
               fmt='%1d', header='ID,Count', comments='')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
