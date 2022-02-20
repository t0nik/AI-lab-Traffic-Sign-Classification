import xml.etree.ElementTree as ET
import cv2
import glob
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier


def parse_data(path):
    data = []
    # Loop through all .xml files in path
    for filename in glob.glob(path + '*.xml'):
        # Parse and receive the data as a tree, get root of the tree
        tree = ET.parse(filename)
        root = tree.getroot()

        sample = {}

        # Name, folder, full image

        for name in root.iter('filename'):
            sample['name'] = name.text

        for folder in root.iter('folder'):
            image_path = filename.replace('annotations', folder.text)
            image_path = image_path.replace('.xml', sample['name'][-4::])  # .png .jpg
            image = cv2.imread(image_path)
            sample['image'] = image

        size_data = {}
        for size in root.iter('size'):
            for width in size.iter('width'):
                size_data['width'] = width.text
            for height in size.iter('height'):
                size_data['height'] = height.text
            for depth in size.iter('depth'):
                size_data['depth'] = depth.text
        sample['size'] = size_data

        # Traffic signs in the image

        sample['count'] = 0
        object_list = []

        for obj in root.iter('object'):
            object_data = {}

            for name_type in obj.iter('name'):
                if name_type.text == 'crosswalk':
                    object_data['type'] = 1  # the goal is to detect crosswalks
                else:
                    object_data['type'] = 0

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

            sample['count'] = sample['count'] + 1

        sample['object'] = object_list

        data.append(sample)

    return data

def balance_dataset(data, ratio):
    sampled_data = random.sample(data, int(ratio * len(data)))

    return sampled_data

def learn_bovw(data):
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        key_points = sift.detect(sample['image'], None)
        key_points, desc = sift.compute(sample['image'], key_points)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)

    return


def extract_features(data):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        # compute descriptor and add it as "desc" entry in object_data
        for object_data in sample['object']:
            key_points = sift.detect(object_data['obj_image'], None)
            desc = bow.compute(object_data['obj_image'], key_points)
            object_data['desc'] = desc

    return data


def train(data):
    # train random forest model and return it from function.
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
    # perform prediction using trained model and add results as "type_pred" (int) entry in sample
    for sample in data:
        for object_data in sample['object']:
            object_data['type_pred'] = 0 # default value for type_pred is 0 - crosswalk not detected
            if object_data['desc'] is not None:
                pred = rf.predict(object_data['desc'])
                object_data['type_pred'] = int(pred)

    return data


def print_predicted(data):
    # Print detected crosswalks
    for sample in data:
        count = 0
        found = False
        for objects in sample['object']:
            if objects['type_pred'] == 1:
                found = True
                count += 1

        if found:
            print(sample['name'])
            print(count)
            for objects in sample['object']:
                if objects['type_pred'] == 1:
                    print(objects['bounds']['xmin'],
                          objects['bounds']['ymin'],
                          objects['bounds']['xmax'],
                          objects['bounds']['ymax'])
    return


def test_print(data):
    # debug - testing print
    for properties in data:
        # print(properties)
        print(properties['name'])
        print(properties['count'])
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
    # print('Learning BoVW')
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