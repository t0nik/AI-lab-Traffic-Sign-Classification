import xml.etree.ElementTree as ET
import cv2
import glob
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier


def parse_data(path):
    # Parse data, put it into a custom structure

    data = []
    # Loop through all .xml files in path
    for filename in glob.glob(path + '*.xml'):
        # Parse and receive the data as a tree, get root of the tree
        tree = ET.parse(filename)
        root = tree.getroot()

        sample = {}

        # Name, full image, image size

        # Find tag 'filename' and iterate through all occurrences
        for name in root.iter('filename'):
            sample['name'] = name.text

        # Etc.
        for folder in root.iter('folder'):
            image_path = filename.replace('annotations', folder.text)
            image_path = image_path.replace('.xml', sample['name'][-4::])  # .png .jpg
            image = cv2.imread(image_path)
            sample['image'] = image

        size_data = {}
        for size in root.iter('size'):
            for width in size.iter('width'):
                size_data['width'] = int(width.text)
            for height in size.iter('height'):
                size_data['height'] = int(height.text)
        sample['size'] = size_data

        # Traffic signs in the image
        object_list = []

        for obj in root.iter('object'):
            object_data = {}

            for name_type in obj.iter('name'):
                if name_type.text == 'crosswalk':
                    object_data['type'] = 4  # the goal is to detect crosswalks
                elif name_type.text == 'stop':
                    object_data['type'] = 3
                elif name_type.text == 'trafficlight':
                    object_data['type'] = 2
                elif name_type.text == 'speedlimit':
                    object_data['type'] = 1

            bounds_data = {}
            for bounds in obj.iter('bndbox'):
                for xmin in bounds.iter('xmin'):
                    bounds_data['xmin'] = int(xmin.text)
                for ymin in bounds.iter('ymin'):
                    bounds_data['ymin'] = int(ymin.text)
                for xmax in bounds.iter('xmax'):
                    bounds_data['xmax'] = int(xmax.text)
                for ymax in bounds.iter('ymax'):
                    bounds_data['ymax'] = int(ymax.text)
            object_data['bounds'] = bounds_data

            crop = sample['image'][bounds_data['ymin']:bounds_data['ymax'], bounds_data['xmin']:bounds_data['xmax']]
            object_data['obj_image'] = crop

            object_list.append(object_data)

        sample['object'] = object_list

        data.append(sample)

    return data

def balance_dataset(data, ratio):
    # Reduce elements of dataset, randomly
    # Ratio interval: from 0.0 to 1.0

    sampled_data = random.sample(data, int(ratio * len(data)))

    return sampled_data

def learn_bovw(data):
    # Create a dictionary of visual words from cropped traffic light images

    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        for object_data in sample['object']:
            key_points = sift.detect(object_data['obj_image'], None)
            key_points, desc = sift.compute(object_data['obj_image'], key_points)

            if desc is not None:
                bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)

    return


def extract_features(data):
    # Create key points in cropped images and determine if they match the vocabulary in the dictionary

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        for object_data in sample['object']:
            # compute descriptor for each object and add it as "desc" entry in object_data
            key_points = sift.detect(object_data['obj_image'], None)
            desc = bow.compute(object_data['obj_image'], key_points)
            object_data['desc'] = desc

    return data


def train(data):
    # Train random forest model and return it from function.

    descs = []
    labels = []
    for sample in data:
        for object_data in sample['object']:
            if object_data['desc'] is not None:
                descs.append(object_data['desc'].squeeze(0))
                labels.append(object_data['type'])

    rf = RandomForestClassifier()

    rf.fit(descs, labels)

    return rf

def predict(rf, data):
    # Perform prediction using trained model and add results as "type_pred" (int) entry in object_data

    for sample in data:
        for object_data in sample['object']:
            object_data['type_pred'] = 0  # default value for type_pred is 0 - nothing predicted
            if object_data['desc'] is not None:
                pred = rf.predict(object_data['desc'])
                object_data['type_pred'] = int(pred)

    return data


def check_size(sample, objects):
    # Returns True if size of object is greater or equal to 10% of the image size

    w = sample['size']['width']
    h = sample['size']['height']
    w_ob = objects['bounds']['xmax'] - objects['bounds']['xmin']
    h_ob = objects['bounds']['ymax'] - objects['bounds']['ymin']

    if w_ob >= 0.1*w and h_ob >= 0.1*h:
        return True

    return False


def print_predicted(data):
    # Print detected crosswalks

    for sample in data:
        count = 0
        found = False
        for objects in sample['object']:
            if objects['type_pred'] == 4 and check_size(sample, objects):
                found = True
                count += 1

        if found:
            print(sample['name'])
            print(count)
            for objects in sample['object']:
                if objects['type_pred'] == 4 and check_size(sample, objects):
                    print(objects['bounds']['xmin'],
                          objects['bounds']['ymin'],
                          objects['bounds']['xmax'],
                          objects['bounds']['ymax'])
    return


def test_print(data):
    # Function used for debug, skip
    for properties in data:
        # print(properties)
        print(properties['name'])
        # print(properties['count'])
        for objects in properties['object']:
            print('predicted:', objects['type_pred'], 'real_type:', objects['type'])
            print(objects['bounds']['xmin'],
                  objects['bounds']['ymin'],
                  objects['bounds']['xmax'],
                  objects['bounds']['ymax'])


def main():

    print('Parsing data...')
    train_data = parse_data('train\\annotations\\')
    test_data = parse_data('test\\annotations\\')

    # print('Balancing datasets...')
    # train_data = balance_dataset(train_data, 0.3)
    # test_data = balance_dataset(test_data, 0.3)

    # comment both lines after dictionary (voc.npy) is saved to disk
    # print('Learning BoVW...')
    # learn_bovw(train_data)

    print('Extracting training features...')
    train_data = extract_features(train_data)

    print('Training...')
    rf = train(train_data)

    print('Extracting test features...')
    test_data = extract_features(test_data)

    print('Predicting...')
    test_data = predict(rf, test_data)

    print('Printing predicted results...')
    print_predicted(test_data)

    # test_print(train_data)

if __name__ == '__main__':
    main()