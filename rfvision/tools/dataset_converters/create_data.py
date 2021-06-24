import argparse
from tools.data_converter import indoor_converter as indoor


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)




parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='sunrgbd', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/sunrgbd',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for sunrgbd')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/sunrgbd',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='sunrgbd')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
