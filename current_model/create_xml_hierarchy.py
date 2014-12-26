
from optparse import OptionParser
import xmlutils

from xml.dom import minidom
from collections import Mapping

import experimental_materials


def dict2element(root, structure, doc):
    """
    Gets a dictionary like structure and converts its
    content into xml elements. After that appends
    resulted elements to root element. If root element
    is a string object creates a new elements with the
    given string and use that element as root.

    This function returns a xml element object.

    """
    assert isinstance(structure, Mapping), \
        'Structure must be a mapping object such as dict'

    # if root is a string make it a element
    if isinstance(root, str):
        root = doc.createElement(root)

    for key, value in structure.iteritems():
        el = doc.createElement("" if key is None else unicode(key))
        if isinstance(value, Mapping):
            dict2element(el, value, doc)
        else:
            el.appendChild(doc.createTextNode("" if value is None
                                              else unicode(value)))
        root.appendChild(el)

    return root


def dict2xml(structure, tostring=False):
    """
    Gets a dict like object as a structure and returns a corresponding minidom
    document object.

    If str is needed instead of minidom, tostring parameter can be used

    Restrictions:
    Structure must only have one root.
    Structure must consist of str or dict objects (other types will
    converted into string)

    Sample structure object would be
    {'root':{'elementwithtextnode':'text content',
             'innerelements':{'innerinnerelements':'inner element content'}}}
    result for this structure would be
    '<?xml version="1.0" ?>
    <root>
      <innerelements>
      <innerinnerelements>inner element content</innerinnerelements>
      </innerelements>
      <elementwithtextnode>text content</elementwithtextnode>
    </root>'
    """
    # This is main function call. which will return a document
    assert len(structure) == 1, 'Structure must have only one root element'
    assert isinstance(structure, Mapping), \
        'Structure must be a mapping object such as dict'

    root_element_name, value = next(structure.iteritems())
    impl = minidom.getDOMImplementation()
    doc = impl.createDocument(None, unicode(root_element_name), None)
    dict2element(doc.documentElement, value, doc)
    return doc.toxml() if tostring else doc

#TODO
def parse_cmd_line():
    optparser = OptionParser()
    optparser.add_option('-f', '--file',
        action='store', dest='config', type='string', default=None,
        help="The file to generate a hierarchy from.")

    options, args = optparser.parse_args()
    return options, args

def generate_from_command_line():
    hiea




if __name__ == '__main__':
    options, args = parse_cmd_line()
    generate_from_command_line()
