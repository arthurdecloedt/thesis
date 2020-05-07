import logging as lg

import multiset_plus

if __name__ == '__main__':
    logFormatter = lg.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = lg.getLogger()
    rootLogger.setLevel(lg.INFO)

    fileHandler = lg.FileHandler("train_binned.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = lg.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    import yaml

    with open('../resources/preferences.yaml') as f:
        prefs = yaml.load(f, Loader=yaml.FullLoader)

    test_set = multiset_plus.MultiSetCombined(prefs, contig_resp=True)
