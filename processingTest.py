import logging as lg

import multiset

lg.basicConfig(level=lg.DEBUG)
import yaml

prefs = {}
with open('resources/preferences.yaml') as f:
    prefs = yaml.load(f, Loader=yaml.FullLoader)

test_set = multiset.MultiSet(prefs, contig_resp=True)
