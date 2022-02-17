import os
import xml.etree.ElementTree as ET
import cv2
import glob


def parse_data(path):
    data = []
    # Loop through all .xml files in path
    for filename in glob.glob(path + '*.xml'):
        # Parse and receive the data as a tree, get root of the tree
        tree = ET.parse(filename)
        root = tree.getroot()

        dataframe = {}

        # Name, folder, size of image

        for name in root.iter('filename'):
            dataframe['name'] = name.text

        for folder in root.iter('folder'):
            image_path = filename.replace('annotations', folder.text)
            image_path = image_path.replace('.xml', dataframe['name'][-4::])  # .png .jpg
            image = cv2.imread(image_path)
            dataframe['image'] = image

        size_data = {}
        for size in root.iter('size'):
            for width in size.iter('width'):
                size_data['width'] = width.text
            for height in size.iter('height'):
                size_data['height'] = height.text
            for depth in size.iter('depth'):
                size_data['depth'] = depth.text
        dataframe['size'] = size_data

        # Traffic signs in the image

        dataframe['count'] = 0
        object_list = []

        for obj in root.iter('object'):
            object_data = {}

            for name_type in obj.iter('name'):
                if name_type.text == 'crosswalk':
                    object_data['type'] = 1  # main goal is to find crosswalks
                else:
                    object_data['type'] = 0

            bounds_data = {}
            for bounds in obj.iter('bndbox'):
                for xmin in bounds.iter('xmin'):
                    bounds_data['xmin'] = xmin.text
                for ymin in bounds.iter('ymin'):
                    bounds_data['ymin'] = ymin.text
                for xmax in bounds.iter('xmax'):
                    bounds_data['xmax'] = xmax.text
                for ymax in bounds.iter('ymax'):
                    bounds_data['ymax'] = ymax.text
            object_data['bounds'] = bounds_data

            object_list.append(object_data)

            dataframe['count'] = dataframe['count'] + 1

        dataframe['object'] = object_list

        data.append(dataframe)

    return data


def main():

    print('Parsing data...')
    train_data = parse_data('train\\annotations\\')





    # debug
    for properties in train_data:
        print(properties['name'])
        print(properties['count'])
        # print(properties['object'])
        for objects in properties['object']:
            print(objects['bounds']['xmin'],
                  objects['bounds']['ymin'],
                  objects['bounds']['xmax'],
                  objects['bounds']['ymax'])


if __name__ == '__main__':
    main()