from xml.dom import minidom

def main():
    # parse an xml file by name
    file = minidom.parse("annotations\\road0.xml")

    width = file.getElementsByTagName("width")

    # print data
    print('\nWidth:')
    for children in width:
        print(children.firstChild.data)


if __name__ == '__main__':
    main()