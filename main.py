import os
import xml.etree.ElementTree as ET
import pandas
import cv2
import glob


def parse_data(path):
    data = []
    # loop through all .xml files in path
    for filename in glob.glob(path + '*.xml'):
        # parse and receive the data as a tree, get root of the tree
        tree = ET.parse(filename)
        root = tree.getroot()

        # for bnd in root.iter('bndbox'):
        #     for dir in bnd:
        #         print(dir.tag + ": " + dir.text)

        for name in root.iter('filename'):
            data.append([name.text])

    return data


def main():

    train_data = parse_data('train\\annotations\\')
    print(train_data)


if __name__ == '__main__':
    main()