import logging as lg

import dataprocessing

lg.basicConfig(level=lg.INFO)
import yaml

prefs = {}
with open('resources/preferences.yaml') as f:
    prefs = yaml.load(f, Loader=yaml.FullLoader)

test_set = dataprocessing.MultiSet(prefs)
for key, value in test_set.datedict.items():
    print(key, len(value))
