#!/usr/bin/python


from __future__ import print_function, division


from argparse import ArgumentParser
import json
import logging
import re
import os
import sys


def generate_simple_data(data_path):

    save_dir = os.path.join(data_path, 'simple')

    # Create the necessary directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Categorize the feature groups as {super., basic, subord., instance} level
    feature_group_to_level_map = {}
    feature_group_to_level_map['instance'] = 'instance'
    feature_group_to_level_map['subordinate'] = 'subordinate'
    feature_group_to_level_map['basic-level'] = 'basic-level'
    feature_group_to_level_map['superordinate'] = 'superordinate'

    json.dump(feature_group_to_level_map,
              open(os.path.join(save_dir, 'feature_group_to_level_map.json'),
                   'w'), indent=4, separators=(',', ': '))

    # Create the objects in the training set
    training_set = {}

    training_set['one example'] = {}
    training_set['three subordinate examples'] = {}
    training_set['three basic-level examples'] = {}
    training_set['three superordinate examples'] = {}

    training_set['one example']['subord_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    training_set['three subordinate examples']['subord_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['three subordinate examples']['subord_02'] = ['instance_feature_02', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['three subordinate examples']['subord_03'] = ['instance_feature_03', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    training_set['three basic-level examples']['basic_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['three basic-level examples']['basic_02'] = ['instance_feature_04', 'subord_feature_02', 'basic_feature_01', 'super_feature_01']
    training_set['three basic-level examples']['basic_03'] = ['instance_feature_05', 'subord_feature_03', 'basic_feature_01', 'super_feature_01']

    training_set['three superordinate examples']['super_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['three superordinate examples']['super_02'] = ['instance_feature_06', 'subord_feature_04', 'basic_feature_02', 'super_feature_01']
    training_set['three superordinate examples']['super_03'] = ['instance_feature_07', 'subord_feature_05', 'basic_feature_03', 'super_feature_01']

    # Create the objects in the test set
    test_set = {}

    test_set['subordinate matches'] = {}
    test_set['basic-level matches'] = {}
    test_set['superordinate matches'] = {}

    test_set['subordinate matches']['subord_01'] = ['instance_feature_08', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    test_set['subordinate matches']['subord_02'] = ['instance_feature_09', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    test_set['basic-level matches']['basic_01'] = ['instance_feature_10', 'subord_feature_06', 'basic_feature_01', 'super_feature_01']
    test_set['basic-level matches']['basic_02'] = ['instance_feature_11', 'subord_feature_07', 'basic_feature_01', 'super_feature_01']

    test_set['superordinate matches']['super_01'] = ['instance_feature_12', 'subord_feature_08', 'basic_feature_04', 'super_feature_01']
    test_set['superordinate matches']['super_02'] = ['instance_feature_13', 'subord_feature_09', 'basic_feature_05', 'super_feature_01']
    test_set['superordinate matches']['super_03'] = ['instance_feature_14', 'subord_feature_10', 'basic_feature_06', 'super_feature_01']
    test_set['superordinate matches']['super_04'] = ['instance_feature_15', 'subord_feature_11', 'basic_feature_07', 'super_feature_01']

    simple = {}
    simple['training set'] = training_set
    simple['test set'] = test_set

    # Create the "unseen" object
    simple['unseen object features'] = ['instance_feature_00', 'subord_feature_00', 'basic_feature_00', 'super_feature_00']

    json.dump(simple, open(os.path.join(save_dir, 'stimuli.json'), 'w'),
              indent=4, separators=(',', ': '))

    # Create the mapping of feature -> feature group
    feature_to_feature_group_map = {}

    for feature in ['instance_feature_{num:02d}'.format(num=i) for i in range(0, 16)]:
        feature_to_feature_group_map[feature] = 'instance'
    for feature in ['subord_feature_{num:02d}'.format(num=i) for i in range(0, 12)]:
        feature_to_feature_group_map[feature] = 'subordinate'
    for feature in ['basic_feature_{num:02d}'.format(num=i) for i in range(0, 8)]:
        feature_to_feature_group_map[feature] = 'basic-level'
    for feature in ['super_feature_{num:02d}'.format(num=i) for i in range(0, 2)]:
        feature_to_feature_group_map[feature] = 'superordinate'

    json.dump(feature_to_feature_group_map,
              open(os.path.join(save_dir, 'feature_to_feature_group_map.json'),
                   'w'), indent=4, separators=(',', ': '), sort_keys=True)


def generate_grid_simple_data(data_path):

    save_dir = os.path.join(data_path, 'grid_simple')

    # Create the necessary directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Categorize the feature groups as {super., basic, subord., instance} level
    feature_group_to_level_map = {}
    feature_group_to_level_map['instance'] = 'instance'
    feature_group_to_level_map['subordinate'] = 'subordinate'
    feature_group_to_level_map['basic-level'] = 'basic-level'
    feature_group_to_level_map['superordinate'] = 'superordinate'

    json.dump(feature_group_to_level_map,
              open(os.path.join(save_dir, 'feature_group_to_level_map.json'),
                   'w'), indent=4, separators=(',', ': '))

    # Create the objects in the training set
    training_set = {}

    training_set['one example'] = {}

    training_set['two subordinate examples'] = {}
    training_set['two basic-level examples'] = {}
    training_set['two superordinate examples'] = {}

    training_set['three subordinate examples'] = {}
    training_set['three basic-level examples'] = {}
    training_set['three superordinate examples'] = {}

    training_set['four subordinate examples'] = {}
    training_set['four basic-level examples'] = {}
    training_set['four superordinate examples'] = {}

    training_set['one example']['subord_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    training_set['two subordinate examples']['subord_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['two subordinate examples']['subord_02'] = ['instance_feature_02', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    training_set['three subordinate examples']['subord_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['three subordinate examples']['subord_02'] = ['instance_feature_02', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['three subordinate examples']['subord_03'] = ['instance_feature_03', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    training_set['four subordinate examples']['subord_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['four subordinate examples']['subord_02'] = ['instance_feature_02', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['four subordinate examples']['subord_03'] = ['instance_feature_03', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['four subordinate examples']['subord_04'] = ['instance_feature_04', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    training_set['two basic-level examples']['basic_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['two basic-level examples']['basic_02'] = ['instance_feature_05', 'subord_feature_02', 'basic_feature_01', 'super_feature_01']

    training_set['three basic-level examples']['basic_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['three basic-level examples']['basic_02'] = ['instance_feature_05', 'subord_feature_02', 'basic_feature_01', 'super_feature_01']
    training_set['three basic-level examples']['basic_03'] = ['instance_feature_06', 'subord_feature_03', 'basic_feature_01', 'super_feature_01']

    training_set['four basic-level examples']['basic_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['four basic-level examples']['basic_02'] = ['instance_feature_05', 'subord_feature_02', 'basic_feature_01', 'super_feature_01']
    training_set['four basic-level examples']['basic_03'] = ['instance_feature_06', 'subord_feature_03', 'basic_feature_01', 'super_feature_01']
    training_set['four basic-level examples']['basic_04'] = ['instance_feature_07', 'subord_feature_04', 'basic_feature_01', 'super_feature_01']

    training_set['two superordinate examples']['super_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['two superordinate examples']['super_02'] = ['instance_feature_08', 'subord_feature_05', 'basic_feature_02', 'super_feature_01']

    training_set['three superordinate examples']['super_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['three superordinate examples']['super_02'] = ['instance_feature_08', 'subord_feature_05', 'basic_feature_02', 'super_feature_01']
    training_set['three superordinate examples']['super_03'] = ['instance_feature_09', 'subord_feature_06', 'basic_feature_03', 'super_feature_01']

    training_set['four superordinate examples']['super_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    training_set['four superordinate examples']['super_02'] = ['instance_feature_08', 'subord_feature_05', 'basic_feature_02', 'super_feature_01']
    training_set['four superordinate examples']['super_03'] = ['instance_feature_09', 'subord_feature_06', 'basic_feature_03', 'super_feature_01']
    training_set['four superordinate examples']['super_04'] = ['instance_feature_10', 'subord_feature_07', 'basic_feature_04', 'super_feature_01']

    # Create the objects in the test set
    test_set = {}

    test_set['subordinate matches'] = {}
    test_set['basic-level matches'] = {}
    test_set['superordinate matches'] = {}

    test_set['subordinate matches']['subord_01'] = ['instance_feature_11', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    test_set['subordinate matches']['subord_02'] = ['instance_feature_12', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    test_set['basic-level matches']['basic_01'] = ['instance_feature_13', 'subord_feature_08', 'basic_feature_01', 'super_feature_01']
    test_set['basic-level matches']['basic_02'] = ['instance_feature_14', 'subord_feature_09', 'basic_feature_01', 'super_feature_01']

    test_set['superordinate matches']['super_01'] = ['instance_feature_15', 'subord_feature_10', 'basic_feature_05', 'super_feature_01']
    test_set['superordinate matches']['super_02'] = ['instance_feature_16', 'subord_feature_11', 'basic_feature_06', 'super_feature_01']
    test_set['superordinate matches']['super_03'] = ['instance_feature_17', 'subord_feature_12', 'basic_feature_07', 'super_feature_01']
    test_set['superordinate matches']['super_04'] = ['instance_feature_18', 'subord_feature_13', 'basic_feature_08', 'super_feature_01']

    simple = {}
    simple['training set'] = training_set
    simple['test set'] = test_set

    # Create the "unseen" object
    simple['unseen object features'] = ['instance_feature_00', 'subord_feature_00', 'basic_feature_00', 'super_feature_00']

    json.dump(simple, open(os.path.join(save_dir, 'stimuli.json'), 'w'),
              indent=4, separators=(',', ': '))

    # Create the mapping of feature -> feature group
    feature_to_feature_group_map = {}

    for feature in ['instance_feature_{num:02d}'.format(num=i) for i in range(0, 19)]:
        feature_to_feature_group_map[feature] = 'instance'
    for feature in ['subord_feature_{num:02d}'.format(num=i) for i in range(0, 14)]:
        feature_to_feature_group_map[feature] = 'subordinate'
    for feature in ['basic_feature_{num:02d}'.format(num=i) for i in range(0, 9)]:
        feature_to_feature_group_map[feature] = 'basic-level'
    for feature in ['super_feature_{num:02d}'.format(num=i) for i in range(0, 2)]:
        feature_to_feature_group_map[feature] = 'superordinate'

    json.dump(feature_to_feature_group_map,
              open(os.path.join(save_dir, 'feature_to_feature_group_map.json'),
                   'w'), indent=4, separators=(',', ': '), sort_keys=True)


def generate_category_data(data_path):

    ###############################################################################
    # "CONTAINERS" SET
    ###############################################################################

    save_dir = os.path.join(data_path, 'containers')

    # Create the necessary directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Categorize the feature groups as {super., basic, subord., instance} level
    feature_group_to_level_map = {}
    feature_group_to_level_map['instance_A'] = 'instance'
    feature_group_to_level_map['instance_B'] = 'instance'
    feature_group_to_level_map['instance_C'] = 'instance'
    feature_group_to_level_map['subordinate_A'] = 'subordinate'
    feature_group_to_level_map['subordinate_B'] = 'subordinate'
    feature_group_to_level_map['subordinate_C'] = 'subordinate'
    feature_group_to_level_map['basic-level_A'] = 'basic-level'
    feature_group_to_level_map['basic-level_B'] = 'basic-level'
    feature_group_to_level_map['basic-level_C'] = 'basic-level'
    feature_group_to_level_map['superordinate_A'] = 'superordinate'
    feature_group_to_level_map['superordinate_B'] = 'superordinate'
    feature_group_to_level_map['superordinate_C'] = 'superordinate'

    json.dump(feature_group_to_level_map,
              open(os.path.join(save_dir, 'feature_group_to_level_map.json'),
                   'w'), indent=4, separators=(',', ': '))

    containers_training_set = {}

    containers_training_set['one example'] = {}
    containers_training_set['three subordinate examples'] = {}
    containers_training_set['three basic-level examples'] = {}
    containers_training_set['three superordinate examples'] = {}

    containers_training_set['one example']['subord_01'] = ['instance_feature_A01', 'instance_feature_B01', 'instance_feature_C01', 'subord_feature_A01', 'subord_feature_B01', 'subord_feature_C01', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']

    containers_training_set['three subordinate examples']['subord_01'] = ['instance_feature_A01', 'instance_feature_B01', 'instance_feature_C01', 'subord_feature_A01', 'subord_feature_B01', 'subord_feature_C01', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_training_set['three subordinate examples']['subord_02'] = ['instance_feature_A02', 'instance_feature_B02', 'instance_feature_C02', 'subord_feature_A01', 'subord_feature_B01', 'subord_feature_C01', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_training_set['three subordinate examples']['subord_03'] = ['instance_feature_A03', 'instance_feature_B03', 'instance_feature_C03', 'subord_feature_A01', 'subord_feature_B01', 'subord_feature_C01', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']

    containers_training_set['three basic-level examples']['basic_01'] = ['instance_feature_A01', 'instance_feature_B01', 'instance_feature_C01', 'subord_feature_A01', 'subord_feature_B01', 'subord_feature_C01', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_training_set['three basic-level examples']['basic_02'] = ['instance_feature_A04', 'instance_feature_B04', 'instance_feature_C04', 'subord_feature_A02', 'subord_feature_B02', 'subord_feature_C02', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_training_set['three basic-level examples']['basic_03'] = ['instance_feature_A05', 'instance_feature_B05', 'instance_feature_C05', 'subord_feature_A03', 'subord_feature_B03', 'subord_feature_C03', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']

    containers_training_set['three superordinate examples']['super_01'] = ['instance_feature_A01', 'instance_feature_B01', 'instance_feature_C01', 'subord_feature_A01', 'subord_feature_B01', 'subord_feature_C01', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_training_set['three superordinate examples']['super_02'] = ['instance_feature_A06', 'instance_feature_B06', 'instance_feature_C06', 'subord_feature_A04', 'subord_feature_B04', 'subord_feature_C04', 'basic_feature_A02', 'basic_feature_B02', 'basic_feature_C02', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_training_set['three superordinate examples']['super_03'] = ['instance_feature_A07', 'instance_feature_B07', 'instance_feature_C07', 'subord_feature_A05', 'subord_feature_B05', 'subord_feature_C05', 'basic_feature_A03', 'basic_feature_B03', 'basic_feature_C03', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']

    containers_test_set = {}

    containers_test_set['subordinate matches'] = {}
    containers_test_set['basic-level matches'] = {}
    containers_test_set['superordinate matches'] = {}

    containers_test_set['subordinate matches']['subord_01'] = ['instance_feature_A08', 'instance_feature_B08', 'instance_feature_C08', 'subord_feature_A01', 'subord_feature_B01', 'subord_feature_C01', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_test_set['subordinate matches']['subord_02'] = ['instance_feature_A09', 'instance_feature_B09', 'instance_feature_C09', 'subord_feature_A01', 'subord_feature_B01', 'subord_feature_C01', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']

    containers_test_set['basic-level matches']['basic_01'] = ['instance_feature_A10', 'instance_feature_B10', 'instance_feature_C10', 'subord_feature_A06', 'subord_feature_B06', 'subord_feature_C06', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_test_set['basic-level matches']['basic_02'] = ['instance_feature_A11', 'instance_feature_B11', 'instance_feature_C11', 'subord_feature_A07', 'subord_feature_B07', 'subord_feature_C07', 'basic_feature_A01', 'basic_feature_B01', 'basic_feature_C01', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']

    containers_test_set['superordinate matches']['super_01'] = ['instance_feature_A12', 'instance_feature_B12', 'instance_feature_C12', 'subord_feature_A08', 'subord_feature_B08', 'subord_feature_C08', 'basic_feature_A04', 'basic_feature_B04', 'basic_feature_C04', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_test_set['superordinate matches']['super_02'] = ['instance_feature_A13', 'instance_feature_B13', 'instance_feature_C13', 'subord_feature_A09', 'subord_feature_B09', 'subord_feature_C09', 'basic_feature_A05', 'basic_feature_B05', 'basic_feature_C05', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_test_set['superordinate matches']['super_03'] = ['instance_feature_A14', 'instance_feature_B14', 'instance_feature_C14', 'subord_feature_A10', 'subord_feature_B10', 'subord_feature_C10', 'basic_feature_A06', 'basic_feature_B06', 'basic_feature_C06', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']
    containers_test_set['superordinate matches']['super_04'] = ['instance_feature_A15', 'instance_feature_B15', 'instance_feature_C15', 'subord_feature_A11', 'subord_feature_B11', 'subord_feature_C11', 'basic_feature_A07', 'basic_feature_B07', 'basic_feature_C07', 'super_feature_A01', 'super_feature_B01', 'super_feature_C01']

    containers = {}
    containers['training set'] = containers_training_set
    containers['test set'] = containers_test_set

    containers['unseen object features'] = ['instance_feature_A00', 'instance_feature_B00', 'instance_feature_C00', 'subord_feature_A00', 'subord_feature_B00', 'subord_feature_C00', 'basic_feature_A00', 'basic_feature_B00', 'basic_feature_C00', 'super_feature_A00', 'super_feature_B00', 'super_feature_C00']

    json.dump(containers, open(os.path.join(save_dir, 'stimuli.json'), 'w'),
              indent=4, separators=(',', ': '))

    container_features = {}

    for letter in ['A', 'B', 'C']:
        for feature in ['instance_feature_' + letter + '{num:02d}'.format(num=i) for i in range(0, 16)]:
            container_features[feature] = 'instance_' + letter
        for feature in ['subord_feature_' + letter + '{num:02d}'.format(num=i) for i in range(0, 12)]:
            container_features[feature] = 'subordinate_' + letter
        for feature in ['basic_feature_' + letter + '{num:02d}'.format(num=i) for i in range(0, 8)]:
            container_features[feature] = 'basic-level_' + letter
        for feature in ['super_feature_' + letter + '{num:02d}'.format(num=i) for i in range(0, 2)]:
            container_features[feature] = 'superordinate_' + letter

    json.dump(container_features,
              open(os.path.join(save_dir, 'feature_to_feature_group_map.json'),
                   'w'), indent=4, separators=(',', ': '), sort_keys=True)


    ###############################################################################
    # "CLOTHING" SET
    ###############################################################################

    save_dir = os.path.join(data_path, 'clothing')

    # Create the necessary directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Categorize the feature groups as {super., basic, subord., instance} level
    feature_group_to_level_map = {}
    feature_group_to_level_map['instance_A'] = 'instance'
    feature_group_to_level_map['instance_B'] = 'instance'
    feature_group_to_level_map['subordinate_A'] = 'subordinate'
    feature_group_to_level_map['subordinate_B'] = 'subordinate'
    feature_group_to_level_map['basic-level_A'] = 'basic-level'
    feature_group_to_level_map['basic-level_B'] = 'basic-level'
    feature_group_to_level_map['superordinate_A'] = 'superordinate'
    feature_group_to_level_map['superordinate_B'] = 'superordinate'

    json.dump(feature_group_to_level_map,
              open(os.path.join(save_dir, 'feature_group_to_level_map.json'),
                   'w'), indent=4, separators=(',', ': '))

    clothing_training_set = {}

    clothing_training_set['one example'] = {}
    clothing_training_set['three subordinate examples'] = {}
    clothing_training_set['three basic-level examples'] = {}
    clothing_training_set['three superordinate examples'] = {}

    clothing_training_set['one example']['subord_01'] = ['instance_feature_A01', 'instance_feature_B01', 'subord_feature_A01', 'subord_feature_B01', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']

    clothing_training_set['three subordinate examples']['subord_01'] = ['instance_feature_A01', 'instance_feature_B01', 'subord_feature_A01', 'subord_feature_B01', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']
    clothing_training_set['three subordinate examples']['subord_02'] = ['instance_feature_A02', 'instance_feature_B02', 'subord_feature_A01', 'subord_feature_B01', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']
    clothing_training_set['three subordinate examples']['subord_03'] = ['instance_feature_A03', 'instance_feature_B03', 'subord_feature_A01', 'subord_feature_B01', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']

    clothing_training_set['three basic-level examples']['basic_01'] = ['instance_feature_A01', 'instance_feature_B01', 'subord_feature_A01', 'subord_feature_B01', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']
    clothing_training_set['three basic-level examples']['basic_02'] = ['instance_feature_A04', 'instance_feature_B04', 'subord_feature_A02', 'subord_feature_B02', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']
    clothing_training_set['three basic-level examples']['basic_03'] = ['instance_feature_A05', 'instance_feature_B05', 'subord_feature_A03', 'subord_feature_B03', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']

    clothing_training_set['three superordinate examples']['super_01'] = ['instance_feature_A01', 'instance_feature_B01', 'subord_feature_A01', 'subord_feature_B01', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']
    clothing_training_set['three superordinate examples']['super_02'] = ['instance_feature_A06', 'instance_feature_B06', 'subord_feature_A04', 'subord_feature_B04', 'basic_feature_A02', 'basic_feature_B02', 'super_feature_A01', 'super_feature_B01']
    clothing_training_set['three superordinate examples']['super_03'] = ['instance_feature_A07', 'instance_feature_B07', 'subord_feature_A05', 'subord_feature_B05', 'basic_feature_A03', 'basic_feature_B03', 'super_feature_A01', 'super_feature_B01']

    clothing_test_set = {}

    clothing_test_set['subordinate matches'] = {}
    clothing_test_set['basic-level matches'] = {}
    clothing_test_set['superordinate matches'] = {}

    clothing_test_set['subordinate matches']['subord_01'] = ['instance_feature_A08', 'instance_feature_B08', 'subord_feature_A01', 'subord_feature_B01', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']
    clothing_test_set['subordinate matches']['subord_02'] = ['instance_feature_A09', 'instance_feature_B09', 'subord_feature_A01', 'subord_feature_B01', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']

    clothing_test_set['basic-level matches']['basic_01'] = ['instance_feature_A10', 'instance_feature_B10', 'subord_feature_A06', 'subord_feature_B06', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']
    clothing_test_set['basic-level matches']['basic_02'] = ['instance_feature_A11', 'instance_feature_B11', 'subord_feature_A07', 'subord_feature_B07', 'basic_feature_A01', 'basic_feature_B01', 'super_feature_A01', 'super_feature_B01']

    clothing_test_set['superordinate matches']['super_01'] = ['instance_feature_A12', 'instance_feature_B12', 'subord_feature_A08', 'subord_feature_B08', 'basic_feature_A04', 'basic_feature_B04', 'super_feature_A01', 'super_feature_B01']
    clothing_test_set['superordinate matches']['super_02'] = ['instance_feature_A13', 'instance_feature_B13', 'subord_feature_A09', 'subord_feature_B09', 'basic_feature_A05', 'basic_feature_B05', 'super_feature_A01', 'super_feature_B01']
    clothing_test_set['superordinate matches']['super_03'] = ['instance_feature_A14', 'instance_feature_B14', 'subord_feature_A10', 'subord_feature_B10', 'basic_feature_A06', 'basic_feature_B06', 'super_feature_A01', 'super_feature_B01']
    clothing_test_set['superordinate matches']['super_04'] = ['instance_feature_A15', 'instance_feature_B15', 'subord_feature_A11', 'subord_feature_B11', 'basic_feature_A07', 'basic_feature_B07', 'super_feature_A01', 'super_feature_B01']

    clothing = {}
    clothing['training set'] = clothing_training_set
    clothing['test set'] = clothing_test_set

    clothing['unseen object features'] = ['instance_feature_A00', 'instance_feature_B00', 'subord_feature_A00', 'subord_feature_B00', 'basic_feature_A00', 'basic_feature_B00', 'super_feature_A00', 'super_feature_B00']

    json.dump(clothing, open(os.path.join(save_dir, 'stimuli.json'), 'w'),
              indent=4, separators=(',', ': '))


    clothing_features = {}

    for letter in ['A', 'B']:
        for feature in ['instance_feature_' + letter + '{num:02d}'.format(num=i) for i in range(0, 16)]:
            clothing_features[feature] = 'instance_' + letter
        for feature in ['subord_feature_' + letter + '{num:02d}'.format(num=i) for i in range(0, 12)]:
            clothing_features[feature] = 'subordinate_' + letter
        for feature in ['basic_feature_' + letter + '{num:02d}'.format(num=i) for i in range(0, 8)]:
            clothing_features[feature] = 'basic-level_' + letter
        for feature in ['super_feature_' + letter + '{num:02d}'.format(num=i) for i in range(0, 2)]:
            clothing_features[feature] = 'superordinate_' + letter

    json.dump(clothing_features,
              open(os.path.join(save_dir, 'feature_to_feature_group_map.json'),
                   'w'), indent=4, separators=(',', ': '), sort_keys=True)

    ###############################################################################
    # "SEATS" SET
    ###############################################################################

    save_dir = os.path.join(data_path, 'seats')

    # Create the necessary directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Categorize the feature groups as {super., basic, subord., instance} level
    feature_group_to_level_map = {}
    feature_group_to_level_map['instance'] = 'instance'
    feature_group_to_level_map['subordinate'] = 'subordinate'
    feature_group_to_level_map['basic-level'] = 'basic-level'
    feature_group_to_level_map['superordinate'] = 'superordinate'

    json.dump(feature_group_to_level_map,
              open(os.path.join(save_dir, 'feature_group_to_level_map.json'),
                   'w'), indent=4, separators=(',', ': '))

    seats_training_set = {}

    seats_training_set['one example'] = {}
    seats_training_set['three subordinate examples'] = {}
    seats_training_set['three basic-level examples'] = {}
    seats_training_set['three superordinate examples'] = {}

    seats_training_set['one example']['subord_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    seats_training_set['three subordinate examples']['subord_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    seats_training_set['three subordinate examples']['subord_02'] = ['instance_feature_02', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    seats_training_set['three subordinate examples']['subord_03'] = ['instance_feature_03', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    seats_training_set['three basic-level examples']['basic_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    seats_training_set['three basic-level examples']['basic_02'] = ['instance_feature_04', 'subord_feature_02', 'basic_feature_01', 'super_feature_01']
    seats_training_set['three basic-level examples']['basic_03'] = ['instance_feature_05', 'subord_feature_03', 'basic_feature_01', 'super_feature_01']

    seats_training_set['three superordinate examples']['super_01'] = ['instance_feature_01', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    seats_training_set['three superordinate examples']['super_02'] = ['instance_feature_06', 'subord_feature_04', 'basic_feature_02', 'super_feature_01']
    seats_training_set['three superordinate examples']['super_03'] = ['instance_feature_07', 'subord_feature_05', 'basic_feature_03', 'super_feature_01']

    seats_test_set = {}

    seats_test_set['subordinate matches'] = {}
    seats_test_set['basic-level matches'] = {}
    seats_test_set['superordinate matches'] = {}

    seats_test_set['subordinate matches']['subord_01'] = ['instance_feature_08', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']
    seats_test_set['subordinate matches']['subord_02'] = ['instance_feature_09', 'subord_feature_01', 'basic_feature_01', 'super_feature_01']

    seats_test_set['basic-level matches']['basic_01'] = ['instance_feature_10', 'subord_feature_06', 'basic_feature_01', 'super_feature_01']
    seats_test_set['basic-level matches']['basic_02'] = ['instance_feature_11', 'subord_feature_07', 'basic_feature_01', 'super_feature_01']

    seats_test_set['superordinate matches']['super_01'] = ['instance_feature_12', 'subord_feature_08', 'basic_feature_04', 'super_feature_01']
    seats_test_set['superordinate matches']['super_02'] = ['instance_feature_13', 'subord_feature_09', 'basic_feature_05', 'super_feature_01']
    seats_test_set['superordinate matches']['super_03'] = ['instance_feature_14', 'subord_feature_10', 'basic_feature_06', 'super_feature_01']
    seats_test_set['superordinate matches']['super_04'] = ['instance_feature_15', 'subord_feature_11', 'basic_feature_07', 'super_feature_01']

    seats = {}
    seats['training set'] = seats_training_set
    seats['test set'] = seats_test_set

    seats['unseen object features'] = ['instance_feature_00', 'subord_feature_00', 'basic_feature_00', 'super_feature_00']

    json.dump(seats, open(os.path.join(save_dir, 'stimuli.json'), 'w'),
              indent=4, separators=(',', ': '))


    seat_features = {}

    for feature in ['instance_feature_{num:02d}'.format(num=i) for i in range(0, 16)]:
        seat_features[feature] = 'instance'
    for feature in ['subord_feature_{num:02d}'.format(num=i) for i in range(0, 12)]:
        seat_features[feature] = 'subordinate'
    for feature in ['basic_feature_{num:02d}'.format(num=i) for i in range(0, 8)]:
        seat_features[feature] = 'basic-level'
    for feature in ['super_feature_{num:02d}'.format(num=i) for i in range(0, 2)]:
        seat_features[feature] = 'superordinate'

    json.dump(seat_features,
              open(os.path.join(save_dir, 'feature_to_feature_group_map.json'),
                   'w'), indent=4, separators=(',', ': '), sort_keys=True)


def generate_xt_data(data_path):

    ###############################################################################
    # "XT HIERARCHY: VEGETABLES" SET
    ###############################################################################

    save_dir = os.path.join(data_path, 'xt_vegetables')

    # Create the necessary directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Categorize the feature groups as {super., basic, subord., instance} level
    feature_group_to_level_map = {}
    feature_group_to_level_map['UNIQ'] = 'instance'
    for level in range(1, 35):
        if level in range(31, 35):
            feature_group_to_level_map[level] = 'instance'
        elif level in range(21, 31):
            feature_group_to_level_map[level] = 'subordinate'
        elif level in range(3, 21):
            feature_group_to_level_map[level] = 'basic-level'
        elif level in range(1, 3):
            feature_group_to_level_map[level] = 'superordinate'

    json.dump(feature_group_to_level_map,
              open(os.path.join(save_dir, 'feature_group_to_level_map.json'),
                   'w'), indent=4, separators=(',', ': '))

    xt_vegetables = {}

    to_parse = '''
    21;EE11,EE12,BB13,BB14,BB15,BB16,W17,W18,W19,W110,W111,W112,W113,W114,W115,W116,W117,W118,W119,W120,J121,I122,H123,H124,H125,H126,H127,H128,H129,H130,H131,H132,H133,H134,UNIQ21
    23;EE11,EE12,BB13,BB14,BB15,BB16,W17,W18,W19,W110,W111,W112,W113,W114,W115,W116,W117,W118,W119,W120,J121,I122,H223,H224,H225,H226,H227,H228,H229,H230,H231,H232,H233,H234,UNIQ23
    24;EE11,EE12,BB13,BB14,BB15,BB16,W17,W18,W19,W110,W111,W112,W113,W114,W115,W116,W117,W118,W119,W120,J121,I222,I223,I224,I225,I226,I227,I228,I229,I230,I231,I232,I233,I234,UNIQ24
    22;EE11,EE12,BB13,BB14,BB15,BB16,W17,W18,W19,W110,W111,W112,W113,W114,W115,W116,W117,W118,W119,W120,J221,J222,J223,J224,J225,J226,F127,F128,F129,F130,F131,F132,F133,F134,UNIQ22
    16;EE11,EE12,BB13,BB14,BB15,BB16,W17,W18,W19,W110,W111,W112,W113,W114,W115,W116,W117,W118,W119,W120,J221,J222,J223,J224,J225,J226,F227,F228,F229,F230,B131,B132,B133,B134,UNIQ16
    17;EE11,EE12,BB13,BB14,BB15,BB16,W17,W18,W19,W110,W111,W112,W113,W114,W115,W116,W117,W118,W119,W120,J221,J222,J223,J224,J225,J226,F227,F228,F229,F230,B131,B132,B133,B134,UNIQ17
    18;EE11,EE12,BB13,BB14,BB15,BB16,W17,W18,W19,W110,W111,W112,W113,W114,W115,W116,W117,W118,W119,W120,J221,J222,J223,J224,J225,J226,F227,F228,F229,F230,B131,B132,B133,B134,UNIQ18
    19;EE11,EE12,BB13,BB14,BB15,BB16,W17,W18,W19,W110,W111,W112,W113,W114,W115,W116,W117,W118,W119,W120,J221,J222,J223,J224,J225,J226,F227,F228,F229,F230,B231,B232,B233,B234,UNIQ19
    20;EE11,EE12,BB13,BB14,BB15,BB16,W17,W18,W19,W110,W111,W112,W113,W114,W115,W116,W117,W118,W119,W120,J221,J222,J223,J224,J225,J226,F227,F228,F229,F230,C31,C32,C33,C34,UNIQ20
    28;EE11,EE12,BB13,BB14,BB15,BB16,W27,W28,W29,W210,W211,W212,W213,W214,W215,W216,W217,W218,W219,W220,W221,W222,W223,W224,W225,W226,W227,W228,W229,W230,W231,W232,W233,W234,UNIQ28
    25;EE11,EE12,BB23,BB24,Y15,Y16,X17,X18,X19,X110,X111,X112,X113,X114,X115,X116,X117,X118,X119,X120,X121,X122,X123,X124,X125,X126,X127,X128,X129,X130,X131,X132,X133,X134,UNIQ25
    29;EE11,EE12,BB23,BB24,Y15,Y16,X27,X28,X29,X210,X211,X212,X213,X214,X215,X216,X217,X218,X219,X220,X221,X222,X223,X224,X225,X226,X227,X228,X229,X230,X231,X232,X233,X234,UNIQ29
    26;EE11,EE12,BB23,BB24,Y25,Y26,Y27,Y28,Y29,Y210,S111,S112,S113,S114,S115,S116,S117,S118,S119,S120,S121,S122,S123,S124,S125,S126,S127,S128,S129,S130,S131,S132,S133,S134,UNIQ26
    30;EE11,EE12,BB23,BB24,Y25,Y26,Y27,Y28,Y29,Y210,S211,S212,S213,S214,S215,S216,S217,S218,S219,S220,S221,S222,S223,S224,S225,S226,S227,S228,S229,S230,S231,S232,S233,S234,UNIQ30
    27;EE21,EE22,EE23,EE24,EE25,EE26,EE27,EE28,EE29,EE210,EE211,EE212,EE213,EE214,EE215,EE216,EE217,EE218,EE219,EE220,EE221,EE222,EE223,EE224,EE225,EE226,EE227,EE228,EE229,EE230,EE231,EE232,EE233,EE234,UNIQ27
    '''

    for item in to_parse.split():
        xt_vegetables[item.split(';')[0]] = item.split(';')[1].split(',')

    xt_vegetables_training_sets = {}
    xt_vegetables_training_sets['one example'] = {}
    xt_vegetables_training_sets['three subordinate examples'] = {}
    xt_vegetables_training_sets['three basic-level examples'] = {}
    xt_vegetables_training_sets['three superordinate examples'] = {}

    xt_vegetables_training_sets['one example']['16'] = xt_vegetables['16']

    xt_vegetables_training_sets['three subordinate examples']['16'] = xt_vegetables['16']
    xt_vegetables_training_sets['three subordinate examples']['17'] = xt_vegetables['17']
    xt_vegetables_training_sets['three subordinate examples']['18'] = xt_vegetables['18']

    xt_vegetables_training_sets['three basic-level examples']['16'] = xt_vegetables['16']
    xt_vegetables_training_sets['three basic-level examples']['21'] = xt_vegetables['21']
    xt_vegetables_training_sets['three basic-level examples']['22'] = xt_vegetables['22']

    xt_vegetables_training_sets['three superordinate examples']['16'] = xt_vegetables['16']
    xt_vegetables_training_sets['three superordinate examples']['25'] = xt_vegetables['25']
    xt_vegetables_training_sets['three superordinate examples']['26'] = xt_vegetables['26']

    xt_vegetables_test_sets = {}
    xt_vegetables_test_sets['subordinate matches'] = {}
    xt_vegetables_test_sets['basic-level matches'] = {}
    xt_vegetables_test_sets['superordinate matches'] = {}

    xt_vegetables_test_sets['subordinate matches']['19'] = xt_vegetables['19']
    xt_vegetables_test_sets['subordinate matches']['20'] = xt_vegetables['20']

    xt_vegetables_test_sets['basic-level matches']['23'] = xt_vegetables['23']
    xt_vegetables_test_sets['basic-level matches']['24'] = xt_vegetables['24']

    xt_vegetables_test_sets['superordinate matches']['27'] = xt_vegetables['27']
    xt_vegetables_test_sets['superordinate matches']['28'] = xt_vegetables['28']
    xt_vegetables_test_sets['superordinate matches']['29'] = xt_vegetables['29']
    xt_vegetables_test_sets['superordinate matches']['30'] = xt_vegetables['30']

    xt_vegetables_sets = {}
    xt_vegetables_sets['training set'] = xt_vegetables_training_sets
    xt_vegetables_sets['test set'] = xt_vegetables_test_sets
    xt_vegetables_sets['test set'] = xt_vegetables_test_sets

    xt_vegetables_sets['unseen object features'] = ['UNSEEN1{num:01d}'.format(num=i) for i in range(1, 35)]
    xt_vegetables_sets['unseen object features'] += ['UNIQ00']

    json.dump(xt_vegetables_sets, open(os.path.join(save_dir, 'stimuli.json'), 'w'),
              indent=4, separators=(',', ': '))


    xt_vegetable_features = {}

    for obj in xt_vegetables:
        for feature in xt_vegetables[obj]:
            if not feature.startswith('UNIQ'):
                level = re.match('[A-Z]*[0-9]([0-9]?[0-9])', feature).group(1)
                xt_vegetable_features[feature] = level
            else:
                xt_vegetable_features[feature] = 'UNIQ'

    for feature in xt_vegetables_sets['unseen object features']:
        if not feature.startswith('UNIQ'):
            level = re.match('[A-Z]*[0-9]([0-9]?[0-9])', feature).group(1)
            xt_vegetable_features[feature] = level
        else:
            xt_vegetable_features[feature] = 'UNIQ'

    json.dump(xt_vegetable_features,
              open(os.path.join(save_dir, 'feature_to_feature_group_map.json'),
                   'w'), indent=4, separators=(',', ': '), sort_keys=True)


    ###############################################################################
    # "XT HIERARCHY: VEHICLES" SET
    ###############################################################################

    save_dir = os.path.join(data_path, 'xt_vehicles')

    # Create the necessary directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Categorize the feature groups as {super., basic, subord., instance} level
    feature_group_to_level_map = {}
    feature_group_to_level_map['UNIQ'] = 'instance'
    for level in range(1, 36):
        if level in range(26, 36):
            feature_group_to_level_map[level] = 'instance'
        elif level in range(12, 26):
            feature_group_to_level_map[level] = 'subordinate'
        elif level in range(2, 12):
            feature_group_to_level_map[level] = 'basic-level'
        elif level in range(1, 2):
            feature_group_to_level_map[level] = 'superordinate'

    json.dump(feature_group_to_level_map,
              open(os.path.join(save_dir, 'feature_group_to_level_map.json'),
                   'w'), indent=4, separators=(',', ': '))


    xt_vehicles = {}

    to_parse = '''
    36;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V110,V111,T112,T113,T114,T115,Q116,Q117,N118,N119,N120,N121,N122,N123,N124,N125,N126,N127,N128,N129,N130,N131,N132,N133,N134,N135,UNIQ36
    39;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V110,V111,T112,T113,T114,T115,Q116,Q117,N218,N219,N220,N221,N222,N223,N224,N225,N226,N227,N228,N229,N230,N231,N232,N233,N234,N235,UNIQ39
    31;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V110,V111,T112,T113,T114,T115,Q216,Q217,Q218,L119,L120,L121,L122,L123,L124,L125,G126,G127,G128,E129,E130,E131,E132,E133,E134,E135,UNIQ31
    32;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V110,V111,T112,T113,T114,T115,Q216,Q217,Q218,L119,L120,L121,L122,L123,L124,L125,G126,G127,G128,E129,E130,E131,E132,E133,E134,E135,UNIQ32
    33;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V110,V111,T112,T113,T114,T115,Q216,Q217,Q218,L119,L120,L121,L122,L123,L124,L125,G126,G127,G128,E129,E130,E131,E132,E133,E134,E135,UNIQ33
    34;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V110,V111,T112,T113,T114,T115,Q216,Q217,Q218,L119,L120,L121,L122,L123,L124,L125,G126,G127,G128,E229,E230,E231,E232,E233,E234,E235,UNIQ34
    35;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V110,V111,T112,T113,T114,T115,Q216,Q217,Q218,L119,L120,L121,L122,L123,L124,L125,G226,G227,G228,G229,G230,G231,G232,G233,G234,G235,UNIQ35
    38;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V110,V111,T112,T113,T114,T115,Q216,Q217,Q218,L219,L220,L221,L222,L223,L224,L225,L226,L227,L228,L229,L230,L231,L232,L233,L234,L235,UNIQ38
    37;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V110,V111,T212,T213,T214,T215,T216,T217,T218,T219,T220,T221,T222,T223,T224,T225,T226,T227,T228,T229,T230,T231,T232,T233,T234,T235,UNIQ37
    45;HH11,HH12,HH13,CC14,CC15,CC16,CC17,CC18,CC19,V210,V211,V212,V213,V214,V215,V216,V217,V218,V219,V220,V221,V222,V223,V224,V225,V226,V227,V228,V229,V230,V231,V232,V233,V234,V235,UNIQ45
    42;HH11,HH12,HH13,CC24,CC25,CC26,CC27,CC28,CC29,CC210,CC211,CC212,CC213,CC214,CC215,CC216,CC217,CC218,CC219,CC220,CC221,CC222,CC223,CC224,CC225,CC226,CC227,CC228,CC229,CC230,CC231,CC232,CC233,CC234,CC234,UNIQ42
    40;HH21,FF12,FF13,FF14,FF15,FF16,FF17,FF18,FF19,FF110,FF111,FF112,FF113,FF114,FF115,FF116,FF117,FF118,FF119,FF120,FF121,FF122,FF123,FF124,FF125,FF126,FF127,FF128,FF129,FF130,FF131,FF132,FF133,FF134,FF134,UNIQ40
    41;HH21,FF22,FF23,FF24,FF25,FF26,FF27,FF28,FF29,FF210,FF211,FF212,U113,U114,U115,U116,U117,U118,O119,O120,O121,O122,O123,O124,O125,O126,O127,O128,O129,O130,O131,O132,O133,O134,O135,UNIQ41
    44;HH21,FF22,FF23,FF24,FF25,FF26,FF27,FF28,FF29,FF210,FF211,FF212,U113,U114,U115,U116,U117,U118,O219,O220,O221,O222,O223,O224,O225,O226,O227,O228,O229,O230,O231,O232,O233,O234,O235,UNIQ44
    43;HH21,FF22,FF23,FF24,FF25,FF26,FF27,FF28,FF29,FF210,FF211,FF212,U213,U214,U215,U216,U217,U218,U219,U220,U221,U222,U223,U224,U225,U226,U227,U228,U229,U230,U231,U232,U233,U234,U235,UNIQ43
    '''

    for item in to_parse.split():
        xt_vehicles[item.split(';')[0]] = item.split(';')[1].split(',')

    xt_vehicles_training_sets = {}
    xt_vehicles_training_sets['one example'] = {}
    xt_vehicles_training_sets['three subordinate examples'] = {}
    xt_vehicles_training_sets['three basic-level examples'] = {}
    xt_vehicles_training_sets['three superordinate examples'] = {}

    xt_vehicles_training_sets['one example']['31'] = xt_vehicles['31']

    xt_vehicles_training_sets['three subordinate examples']['31'] = xt_vehicles['31']
    xt_vehicles_training_sets['three subordinate examples']['32'] = xt_vehicles['32']
    xt_vehicles_training_sets['three subordinate examples']['33'] = xt_vehicles['33']

    xt_vehicles_training_sets['three basic-level examples']['31'] = xt_vehicles['31']
    xt_vehicles_training_sets['three basic-level examples']['36'] = xt_vehicles['36']
    xt_vehicles_training_sets['three basic-level examples']['37'] = xt_vehicles['37']

    xt_vehicles_training_sets['three superordinate examples']['31'] = xt_vehicles['31']
    xt_vehicles_training_sets['three superordinate examples']['40'] = xt_vehicles['40']
    xt_vehicles_training_sets['three superordinate examples']['41'] = xt_vehicles['41']

    xt_vehicles_test_sets = {}
    xt_vehicles_test_sets['subordinate matches'] = {}
    xt_vehicles_test_sets['basic-level matches'] = {}
    xt_vehicles_test_sets['superordinate matches'] = {}

    xt_vehicles_test_sets['subordinate matches']['34'] = xt_vehicles['34']
    xt_vehicles_test_sets['subordinate matches']['35'] = xt_vehicles['35']

    xt_vehicles_test_sets['basic-level matches']['38'] = xt_vehicles['38']
    xt_vehicles_test_sets['basic-level matches']['39'] = xt_vehicles['39']

    xt_vehicles_test_sets['superordinate matches']['42'] = xt_vehicles['42']
    xt_vehicles_test_sets['superordinate matches']['43'] = xt_vehicles['43']
    xt_vehicles_test_sets['superordinate matches']['44'] = xt_vehicles['44']
    xt_vehicles_test_sets['superordinate matches']['45'] = xt_vehicles['45']

    xt_vehicles_sets = {}
    xt_vehicles_sets['training set'] = xt_vehicles_training_sets
    xt_vehicles_sets['test set'] = xt_vehicles_test_sets

    xt_vehicles_sets['unseen object features'] = ['UNSEEN1{num:01d}'.format(num=i) for i in range(1, 36)]
    xt_vehicles_sets['unseen object features'] += ['UNIQ00']

    json.dump(xt_vehicles_sets, open(os.path.join(save_dir, 'stimuli.json'), 'w'),
              indent=4, separators=(',', ': '))


    xt_vehicle_features = {}

    for feature in ['UNIQ{num:02d}'.format(num=i) for i in range(31, 46)]:
        xt_vehicle_features[feature] = 'instance'
    xt_vehicle_features['UNIQ00'] = 'instance'

    for obj in xt_vehicles:
        for feature in xt_vehicles[obj]:
            if not feature.startswith('UNIQ'):
                level = re.match('[A-Z]*[0-9]([0-9]?[0-9])', feature).group(1)
                xt_vehicle_features[feature] = level
            else:
                xt_vehicle_features[feature] = 'UNIQ'

    for feature in xt_vehicles_sets['unseen object features']:
        if not feature.startswith('UNIQ'):
            level = re.match('[A-Z]*[0-9]([0-9]?[0-9])', feature).group(1)
            xt_vehicle_features[feature] = level
        else:
            xt_vehicle_features[feature] = 'UNIQ'

    json.dump(xt_vehicle_features,
              open(os.path.join(save_dir, 'feature_to_feature_group_map.json'),
                   'w'), indent=4, separators=(',', ': '), sort_keys=True)


    ###############################################################################
    # "XT HIERARCHY: ANIMALS" SET
    ###############################################################################

    save_dir = os.path.join(data_path, 'xt_animals')

    # Create the necessary directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Categorize the feature groups as {super., basic, subord., instance} level
    feature_group_to_level_map = {}
    feature_group_to_level_map['UNIQ'] = 'instance'
    for level in range(1, 41):
        if level in range(36, 41):
            feature_group_to_level_map[level] = 'instance'
        elif level in range(21, 36):
            feature_group_to_level_map[level] = 'subordinate'
        elif level in range(3, 21):
            feature_group_to_level_map[level] = 'basic-level'
        elif level in range(1, 3):
            feature_group_to_level_map[level] = 'superordinate'

    json.dump(feature_group_to_level_map,
              open(os.path.join(save_dir, 'feature_group_to_level_map.json'),
                   'w'), indent=4, separators=(',', ': '))


    xt_animals = {}

    to_parse = '''
    1;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,R121,R122,P123,K124,K125,K126,K127,K128,K129,K130,K131,K132,K133,K134,K135,D136,A137,A138,A139,A140,UNIQ1
    2;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,R121,R122,P123,K124,K125,K126,K127,K128,K129,K130,K131,K132,K133,K134,K135,D136,A137,A138,A139,A140,UNIQ2
    3;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,R121,R122,P123,K124,K125,K126,K127,K128,K129,K130,K131,K132,K133,K134,K135,D136,A137,A138,A139,A140,UNIQ3
    4;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,R121,R122,P123,K124,K125,K126,K127,K128,K129,K130,K131,K132,K133,K134,K135,D136,A237,A238,A239,A240,UNIQ4
    5;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,R121,R122,P123,K124,K125,K126,K127,K128,K129,K130,K131,K132,K133,K134,K135,D236,D237,D238,D239,D240,UNIQ5
    6;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,R121,R122,P123,K224,K225,K226,K227,K228,K229,K230,K231,K232,K233,K234,K235,K236,K237,K238,K239,K240,UNIQ6
    8;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,R121,R122,P223,P224,P225,P226,P227,P228,P229,P230,P231,P232,P233,P234,P235,P236,P237,P238,P239,P240,UNIQ8
    7;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,R221,R222,R223,M124,M125,M126,M127,M128,M129,M130,M131,M132,M133,M134,M135,M136,M137,M138,M139,M140,UNIQ7
    9;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,R221,R222,R223,M224,M225,M226,M227,M228,M229,M230,M231,M232,M233,M234,M235,M236,M237,M238,M239,M240,UNIQ9
    12;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA110,Z211,Z212,Z213,Z214,Z215,Z216,Z217,Z218,Z219,Z220,Z221,Z222,Z223,Z224,Z225,Z226,Z227,Z228,Z229,Z230,Z231,Z232,Z233,Z234,Z235,Z236,Z237,Z238,Z239,Z240,UNIQ12
    15;JJ11,JJ12,II13,II14,II15,II16,DD17,DD18,DD19,AA210,AA211,AA212,AA213,AA214,AA215,AA216,AA217,AA218,AA219,AA220,AA221,AA222,AA223,AA224,AA225,AA226,AA227,AA228,AA229,AA230,AA231,AA232,AA233,AA234,AA234,AA236,AA237,AA238,AA239,AA240,UNIQ15
    14;JJ11,JJ12,II13,II14,II15,II16,DD27,DD28,DD29,DD210,DD211,DD212,DD213,DD214,DD215,DD216,DD217,DD218,DD219,DD220,DD221,DD222,DD223,DD224,DD225,DD226,DD227,DD228,DD229,DD230,DD231,DD232,DD233,DD234,DD234,DD236,DD237,DD238,DD239,DD240,UNIQ14
    11;JJ11,JJ12,II23,II24,II25,II26,II27,II28,II29,II210,II211,II212,II213,II214,II215,II216,II217,II218,II219,II220,II221,II222,II223,II224,II225,II226,II227,II228,II229,II230,II231,II232,II233,II234,II234,II236,II237,II238,II239,II240,UNIQ11
    10;JJ21,JJ22,JJ23,JJ24,GG15,GG16,GG17,GG18,GG19,GG110,GG111,GG112,GG113,GG114,GG115,GG116,GG117,GG118,GG119,GG120,GG121,GG122,GG123,GG124,GG125,GG126,GG127,GG128,GG129,GG130,GG131,GG132,GG133,GG134,GG134,GG136,GG137,GG138,GG139,GG140,UNIQ10
    13;JJ21,JJ22,JJ23,JJ24,GG25,GG26,GG27,GG28,GG29,GG210,GG211,GG212,GG213,GG214,GG215,GG216,GG217,GG218,GG219,GG220,GG221,GG222,GG223,GG224,GG225,GG226,GG227,GG228,GG229,GG230,GG231,GG232,GG233,GG234,GG234,GG236,GG237,GG238,GG239,GG240,UNIQ13
    '''

    for item in to_parse.split():
        xt_animals[item.split(';')[0]] = item.split(';')[1].split(',')

    xt_animals_training_sets = {}
    xt_animals_training_sets['one example'] = {}
    xt_animals_training_sets['three subordinate examples'] = {}
    xt_animals_training_sets['three basic-level examples'] = {}
    xt_animals_training_sets['three superordinate examples'] = {}

    xt_animals_training_sets['one example']['1'] = xt_animals['1']

    xt_animals_training_sets['three subordinate examples']['1'] = xt_animals['1']
    xt_animals_training_sets['three subordinate examples']['2'] = xt_animals['2']
    xt_animals_training_sets['three subordinate examples']['3'] = xt_animals['3']

    xt_animals_training_sets['three basic-level examples']['1'] = xt_animals['1']
    xt_animals_training_sets['three basic-level examples']['6'] = xt_animals['6']
    xt_animals_training_sets['three basic-level examples']['7'] = xt_animals['7']

    xt_animals_training_sets['three superordinate examples']['1'] = xt_animals['1']
    xt_animals_training_sets['three superordinate examples']['10'] = xt_animals['10']
    xt_animals_training_sets['three superordinate examples']['11'] = xt_animals['11']

    xt_animals_test_sets = {}
    xt_animals_test_sets['subordinate matches'] = {}
    xt_animals_test_sets['basic-level matches'] = {}
    xt_animals_test_sets['superordinate matches'] = {}

    xt_animals_test_sets['subordinate matches']['4'] = xt_animals['4']
    xt_animals_test_sets['subordinate matches']['5'] = xt_animals['5']

    xt_animals_test_sets['basic-level matches']['8'] = xt_animals['8']
    xt_animals_test_sets['basic-level matches']['9'] = xt_animals['9']

    xt_animals_test_sets['superordinate matches']['12'] = xt_animals['12']
    xt_animals_test_sets['superordinate matches']['13'] = xt_animals['13']
    xt_animals_test_sets['superordinate matches']['14'] = xt_animals['14']
    xt_animals_test_sets['superordinate matches']['15'] = xt_animals['15']

    xt_animals_sets = {}
    xt_animals_sets['training set'] = xt_animals_training_sets
    xt_animals_sets['test set'] = xt_animals_test_sets

    xt_animals_sets['unseen object features'] = ['UNSEEN1{num:01d}'.format(num=i) for i in range(1, 41)]
    xt_animals_sets['unseen object features'] += ['UNIQ00']

    json.dump(xt_animals_sets, open(os.path.join(save_dir, 'stimuli.json'), 'w'),
              indent=4, separators=(',', ': '))


    xt_animal_features = {}

    for obj in xt_animals:
        for feature in xt_animals[obj]:
            if not feature.startswith('UNIQ'):
                level = re.match('[A-Z]*[0-9]([0-9]?[0-9])', feature).group(1)
                xt_animal_features[feature] = level
            else:
                xt_animal_features[feature] = 'UNIQ'

    for feature in xt_animals_sets['unseen object features']:
        if not feature.startswith('UNIQ'):
            level = re.match('[A-Z]*[0-9]([0-9]?[0-9])', feature).group(1)
            xt_animal_features[feature] = level
        else:
            xt_animal_features[feature] = 'UNIQ'

    json.dump(xt_animal_features,
              open(os.path.join(save_dir, 'feature_to_feature_group_map.json'),
                   'w'), indent=4, separators=(',', ': '), sort_keys=True)


def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument('--logging', type=str, default='INFO',
                        metavar='logging', choices=['DEBUG', 'INFO', 'WARNING',
                                                    'ERROR', 'CRITICAL'],
                        help='Logging level')

    parser.add_argument('--data_path', '-d', metavar='data_path',
                        type=str,
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help='The path to which to write the data')

    return parser.parse_args(args)


def main(args=sys.argv[1:]):

    args = parse_args(args)

    logging.basicConfig(level=args.logging)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    generate_simple_data(args.data_path)
    generate_grid_simple_data(args.data_path)
    generate_category_data(args.data_path)
    generate_xt_data(args.data_path)


if __name__ == '__main__':
    sys.exit(main())
