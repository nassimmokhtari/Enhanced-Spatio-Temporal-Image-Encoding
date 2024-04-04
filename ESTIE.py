############################################################################################################
# Copyright (c) 2022, Nassim Mokhtari, Vincent Fer, Alexis Nédélec, Marlène Gilles and Pierre De Loor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
############################################################################################################
import numpy as np
from os.path import join
import argparse
import os

# List of actions_list from the Online Action Detection Dataset
actions_list = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping',
                'drinking', 'opening microwave oven', 'Throwing trash']

parser = argparse.ArgumentParser(description='Enhanced Spatio-Temporal Image Encoding')
parser.add_argument('--data_path', default='data', type=str, help='path to the dataset. Default is "data"')
parser.add_argument('--output_dir', default='Out', type=str,
                    help='directory used to store the encoded dataset. Default is "Out"')
parser.add_argument('--one_hot_encoding', default='True', type=str, help='True to use one hot encoding of the labels,'
                                                                         ' False to use class Index. Default is True')
parser.add_argument('--order', default='foot_to_foot', type=str,
                    help='order used to organise the skeleton : foot_to_foot, human (head_to_feet) and no order. '
                         'Default is "foot_to_foot"')
parser.add_argument('--window_length', default=40, type=int, help="Sliding window's length. Default is 40")
parser.add_argument('--rho', default=1, type=float, help="Impact of the motion energy between 0 and 1. Default is 1")


def create_dir(path):
    """
    Checks if the "path" exists, and create it if not.
    :param path: directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_data_file(file, order):
    """
    Reads a .txt file containing the skeleton data and re-order it according to specified order
    :param file: file path
    :param order: re-ordering type : foot_to_foot, human (head_to_feet) and no order

    :return:
    """
    try:
        f = open(file, 'r').read().split()
        data = [float(x) for x in f]
        if order == 'no_order':
            data = np.asarray(data)
            data = data.reshape((25, 3))
        else:
            spine_base = data[0:3]
            spine_mid = data[3:6]
            neck = data[6:9]
            head = data[9:12]
            shoulder_left = data[12:15]
            elbow_left = data[15:18]
            wrist_left = data[18:21]
            hand_left = data[21:24]
            shoulder_right = data[24:27]
            elbow_right = data[27:30]
            wrist_right = data[30:33]
            hand_right = data[33:36]
            hip_left = data[36:39]
            knee_left = data[39:42]
            ankle_left = data[42:45]
            foot_left = data[45:48]
            hip_right = data[48:51]
            knee_right = data[51:54]
            ankle_right = data[54:57]
            foot_right = data[57:60]
            spine_shoulder = data[60:63]
            hand_tip_left = data[63:66]
            thumb_left = data[66:69]
            hand_tip_right = data[69:72]
            thumb_right = data[72:75]

            if order == 'human':
                data = np.stack((head, neck, spine_shoulder, shoulder_left, shoulder_right, elbow_left, elbow_right,
                                 wrist_left, wrist_right, thumb_left, thumb_right, hand_left, hand_right, hand_tip_left,
                                 hand_tip_right, spine_mid, spine_base, hip_left, hip_right, knee_left, knee_right,
                                 ankle_left, ankle_right, foot_left, foot_right))
            else:
                data = np.stack((foot_left, ankle_left, knee_left, hip_left, spine_base, hand_tip_left, thumb_left,
                                 hand_left, wrist_left, elbow_left, shoulder_left, spine_shoulder, head, neck,
                                 shoulder_right, elbow_right, wrist_right, hand_right, thumb_right, hand_tip_right,
                                 spine_mid, hip_right, knee_right, ankle_right, foot_right))
        return data
    except:
        return None



def normalize(array):
    """
    Normalizes an array between 0 and 1 using the Min-Max normalization
    :param array: data to normalize
    :return: normalized data
    """
    min_ = np.min(array, 0)
    max_ = np.max(array, 0)
    return (array - min_) / (max_ - min_)


def get_sequence_energy(sequence, rho):
    """
    Computes the energy of each joint in a sequence
    :param sequence: data
    :param rho: impact of the motion energy
    :return: normalized energy of each joint, array of shape (len_sequence,25)
    """
    energy = np.zeros((len(sequence), 25))
    for i in range(len(sequence)):
        for k in range(25):
            if i == 0:
                energy[i][k] = np.linalg.norm(sequence[i][k] - sequence[i + 1][k])
            elif i == len(sequence) - 1:
                energy[i][k] = np.linalg.norm(sequence[i][k] - sequence[i - 1][k])
            else:
                energy[i][k] = (np.linalg.norm(sequence[i][k] - sequence[i + 1][k]) + np.linalg.norm(
                    sequence[i][k] - sequence[i - 1][k])) / 2
    E = normalize(energy)
    w = rho * E + (1 - rho)
    return w


def get_labels(file):
    """
    Reads a .txt file and return the information related to actions_list (start,end,labels)
    :param file: path to file
    :return: (start,end,label) of the performed actions_list
    """
    labels = open(file, 'r').read().splitlines()
    prev_action = None
    start = []
    end = []
    actions = []
    for line in labels:
        if line.replace(' ', '').isalpha():
            prev_action = line.strip()
        else:
            tab = line.split(' ')
            start.append(int(tab[0]))
            end.append(int(tab[1]))
            actions.append(prev_action)
    return (start, end, actions)


def get_image_label(start, end, labels):
    """
    Return the label of the action performed at the middle of the sequence
    :param start: frame index, referring to the begging of the sequence
    :param end: frame index, referring to the end of the sequence
    :param labels: (start,end,label) of the performed actions_list
    :return: actual action
    """
    index = (start + end) // 2
    for s, e, a in set(zip(labels[0], labels[1], labels[2])):
        if s <= index <= e:
            return a
    return None


def ESTIE(image, weights):
    """
    Transforms a sequence of skeleton joints into the Enhanced Spatio-Temporal Image Encoding
    :param image: sequence of skeleton joints
    :param weights: energy of each joint
    :return: Image representing the input sequence
    """
    RGB = image
    from copy import deepcopy
    height = image.shape[1]
    width = image.shape[0]
    X = np.arange(height)
    Y = np.arange(width)
    RGB = np.squeeze(RGB)
    En = deepcopy(RGB)
    if len(weights.shape) == 1:
        weights = np.expand_dims(weights, 0)
    white = np.ones((width, height)) * 255
    for i in range(3):
        RGB[:, :, i] = np.floor(
            255 * (RGB[:, :, i] - np.min(RGB[:, :, i])) / (np.max(RGB[:, :, i]) - np.min(RGB[:, :, i])))
        En[:, :, i] = RGB[:, :, i] * weights + (1 - weights) * white

    img = np.zeros((2 * height, width, 3), dtype=np.uint8)
    for i in X:
        for j in Y:
            img[i, j] = RGB[j, i]
    for i in X:
        for j in Y:
            img[i + 25, j] = En[j, i]
    return img


def split_into_sequences(data_path, labels, window_length, order, rho):
    """
    Splits a sequence of skeleton data into several samples using a sliding window
    :param data_path: sequence path
    :param labels: labels of the sequence
    :param window_length: sliding window length
    :param order: re-ordering type, used for reading the skeleton data
    :param rho: impact of the motion energy
    :return: Samples, Labels, Energy of each sample
    """
    start_frame = min(labels[0]) - window_length // 2
    end_frame = max(labels[1]) + window_length // 2
    data = []
    for i in range(start_frame, end_frame + 1):
        data.append(load_data_file(data_path + '/' + str(i) + '.txt', order))
    images = [data[i:i + window_length] for i in range(len(data) - window_length + 1)]
    lab = [get_image_label(i, i + window_length, labels) for i in range(start_frame, end_frame - window_length + 2)]

    # Removing the sequences without label (label = None)
    i = 0
    while i < len(lab):
        if lab[i] is None:
            del lab[i]
            del images[i]
        else:
            i += 1
    # Removing the sequences with missing data (some joints coordinates are None)
    i = 0
    while i < len(images):
        jumped = False
        for x in images[i]:
            if x is None or not x.shape == (25, 3):
                del lab[i]
                del images[i]
                jumped = True
                break
        if not jumped:
            i += 1
    return np.asarray(images), np.asarray(lab), [get_sequence_energy(x, rho) for x in images]


def ESTIE_sequence(data_path, label_path, window_length, order, rho):
    """
    Trensform a sequence of skeleton data to a sequence of images
    :param data_path: sequence path
    :param label_path: label path
    :param window_length: sliding window's length
    :param order: re-ordering type, used for reading the skeleton data
    :param rho: impact of the motion energy
    :return: encoded images, labels
    """
    images, labels, weights = split_into_sequences(data_path, get_labels(label_path), window_length, order, rho)
    data = []
    lab = []
    for i in range(len(images)):
        data.append(ESTIE(images[i], weights[i]))
        lab.append(actions_list.index(labels[i]))

    data = np.asarray(data)
    labels = np.asarray(lab)
    return data, labels


def main(data_path, output_dir, one_hot_encoding, window_length, order, rho):
    """
    Encode the data from "data_path" and store them into the "output_dir" directory
    :param data_path: location of the dataset
    :param output_dir: output directory
    :param one_hot_encoding: True to use one hot encoding of the labels, False to used class index
    :param window_length: sliding window's length
    :param order: re-ordering type, used for reading the skeleton data
    :param rho: impact of the motion energy
    """

    # Train and Test sub from the ReadMe of the OAD dataset
    train_sub = [1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 32, 33, 34, 35, 37, 38, 39, 49, 50, 51,
                 54, 57, 58]
    test_sub = [0, 10, 13, 17, 21, 26, 27, 28, 29, 36, 40, 41, 42, 43, 44, 45, 52, 53, 55, 56]
    train = None
    train_label = None
    test = None
    test_label = None
    for i in range(59):
        path = join(data_path, str(i))
        label_path = join(path, 'label', 'label.txt')
        image_path = join(path, 'skeleton')
        print('Processing sequence num ===========>', i)
        data, label = ESTIE_sequence(image_path, label_path, window_length, order, rho)
        if i in train_sub:
            if train_sub.index(i) == 0:
                train = data
                train_label = label
            else:
                train = np.concatenate([train, data])
                train_label = np.concatenate([train_label, label])
        elif i in test_sub:
            if test_sub.index(i) == 0:
                test = data
                test_label = label
            else:
                test = np.concatenate([test, data])
                test_label = np.concatenate([test_label, label])

    if one_hot_encoding:
        from keras.utils.np_utils import to_categorical
        test_label = to_categorical(test_label)
        train_label = to_categorical(train_label)

    create_dir(output_dir)
    np.save(f'{output_dir}/ESTIE_train_x.npy', train)
    np.save(f'{output_dir}/ESTIE_test_x.npy', test)
    np.save(f'{output_dir}/ESTIE_train_y.npy', train_label)
    np.save(f'{output_dir}/ESTIE_test_y.npy', test_label)
    print('Train: data shape', train.shape, 'label shape', train_label.shape)
    print('Test: data shape', test.shape, 'label shape', test_label.shape)


if __name__ == "__main__":
    args = parser.parse_args()
    main(data_path=args.data_path,
         output_dir=args.output_dir,
         one_hot_encoding=args.one_hot_encoding.lower() == 'true',
         order=args.order,
         window_length=args.window_length,
         rho=args.rho)
