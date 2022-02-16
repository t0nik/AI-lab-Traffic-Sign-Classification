import xml.etree.ElementTree as ET

def main():

    # parse and receive the data as a tree, get root of the tree
    tree = ET.parse('test\\annotations\\road0.xml')
    root = tree.getroot()

    for bnd in root.iter('bndbox'):
        for dir in bnd:
            print(dir.tag + ": " + dir.text)

if __name__ == '__main__':
    main()