import logging as lg

import dataprocessing

lg.basicConfig(level=lg.DEBUG)
preembed_folder = '/data/leuven/332/vsc33219/data/test_embed/'
json_folder = '/data/leuven/332/vsc33219/data/aapl.json'

test_set = dataprocessing.MultiSet(preembed_folder, json_folder)
