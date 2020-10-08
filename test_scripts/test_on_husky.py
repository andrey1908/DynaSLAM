import argparse
import os
import os.path as osp
import subprocess


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data-fld', '--husky-data-folder', type=str, default='/media/cds-jetson-host/data/Husky-NKBVS')
    parser.add_argument('-out-fld', '--out-folder', type=str, default=None)
    parser.add_argument('-net-set', '--network-settings-file', type=str, default=None)
    parser.add_argument('-seq', '--sequences', type=int, nargs='+')
    parser.add_argument('--dyna-slam-folder', type=str, default='./')
    return parser


program_to_run = 'Examples/Stereo/stereo_kitti'

husky_calib_files = ['camera_calibration/Husky_2020-03-16.yaml',
                     'camera_calibration/Husky_2020-03-17.yaml',
                     'camera_calibration/Husky_2020-04-24.yaml']

husky_calib_dates = ['2020-03-16',
                     '2020-03-17',
                     '2020-04-24']


def find_calibration_file(data_folder):
    idx = -1
    for i, calib_date in enumerate(husky_calib_dates):
        pos = data_folder.find(calib_date)
        if pos != -1:
            assert idx == -1
            assert data_folder.find(calib_date, pos + 1) == -1
            idx = i
    if idx == -1:
        return None
    else:
        return husky_calib_files[idx]


def test_on_husky(husky_data_folder, out_folder=None, network_settings_file=None, sequences=None, dyna_slam_folder='./'):
    items = os.listdir(husky_data_folder)
    data_folders = list()
    calib_files = list()
    for item in items:
        data_folder = osp.join(husky_data_folder, item)
        if not osp.isdir(data_folder):
            continue
        if not osp.isdir(osp.join(data_folder, 'stereo_images/gray')):
            continue
        calib_file = find_calibration_file(item)
        if not calib_file:
            continue
        data_folders.append(osp.join(data_folder))
        calib_files.append(osp.join(dyna_slam_folder, calib_file))
    print("Found {} sequences\n".format(len(data_folders)))

    if sequences:
        for i in range(len(data_folders)-1, -1, -1):
            idx = int(osp.basename(osp.normpath(data_folders[i]))[:2])
            if idx not in sequences:
                del data_folders[i]
                del calib_files[i]
        assert len(data_folders) == len(sequences)
        print("Use {} sequences\n".format(len(data_folders)))
    else:
        print("Use all sequences\n")

    if out_folder:
        if not osp.exists(out_folder):
            os.mkdir(out_folder)
    for data_folder, calib_file in zip(data_folders, calib_files):
        print("Sequence {}\nCalibration {}\n".format(data_folder, calib_file))
        if network_settings_file:
            res = subprocess.call([osp.join(dyna_slam_folder, program_to_run), 'Vocabulary/ORBvoc.txt', calib_file,
                                  osp.join(data_folder, 'stereo_images/gray'), network_settings_file])
        else:
            res = subprocess.call([osp.join(dyna_slam_folder, program_to_run), 'Vocabulary/ORBvoc.txt', calib_file,
                                  osp.join(data_folder, 'stereo_images/gray')])
        assert res == 0
        if out_folder:
            os.rename(osp.join(dyna_slam_folder, 'CameraTrajectory.txt'),
                      osp.join(out_folder, osp.basename(osp.normpath(data_folder))[:2]+'.txt'))


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    test_on_husky(**vars(args))
