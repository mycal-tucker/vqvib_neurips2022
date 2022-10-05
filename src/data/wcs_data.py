import string
import numpy as np
import pandas as pd
import os
import pickle
from src.utils.files import ensure_dir, ensure_file
from bunch import Bunch


N_CHIPS = 330
N_COLS = 41
N_ROWS = 10
SPACE = 0.1
ROWS = [string.ascii_uppercase[i] for i in range(10)]

# download data files
curr_path = os.path.dirname(os.path.abspath(__file__))
WCS_DATA_DIR = curr_path + '/../../data/wcs/'
ensure_dir(curr_path + '/../../data/')
ensure_dir(WCS_DATA_DIR)
ensure_file(WCS_DATA_DIR + 'cnum-vhcm-lab-new.txt', 'http://www1.icsi.berkeley.edu/wcs/data/cnum-maps/cnum-vhcm-lab-new.txt')
ensure_file(WCS_DATA_DIR + 'chip.txt', 'http://www1.icsi.berkeley.edu/wcs/data/20021219/txt/chip.txt')
ensure_file(WCS_DATA_DIR + 'term.txt', 'http://www1.icsi.berkeley.edu/wcs/data/20021219/txt/term.txt')
ensure_file(WCS_DATA_DIR + 'langs_info.txt', 'https://www1.icsi.berkeley.edu/wcs/data/20021219/txt/lang.txt')

# read data
TERMS_DF = pd.read_csv(WCS_DATA_DIR + 'term.txt', delimiter='\t', header=None, keep_default_na=False, na_values=['NaN'])
CHIPS_DF = pd.read_csv(WCS_DATA_DIR + 'cnum-vhcm-lab-new.txt', delimiter='\t').sort_values(by='#cnum')
CNUMS_DF = pd.read_csv(WCS_DATA_DIR + 'chip.txt', delimiter='\t', header=None).values

WCS_CHIPS = CHIPS_DF[['L*', 'a*', 'b*']].values
LANGS = pd.read_csv(WCS_DATA_DIR + 'langs_info.txt', delimiter='\t', header=None, encoding='utf8')

CNUMS_WCS_COR = dict(
    zip(CNUMS_DF[:, 0], [(ROWS.index(CNUMS_DF[cnum - 1, 1]), CNUMS_DF[cnum - 1, 2]) for cnum in CNUMS_DF[:, 0]]))
_WCS_COR_CNUMS = dict(zip(CNUMS_DF[:, 3], CNUMS_DF[:, 0]))


class WCSDataset:

    def __init__(self, data_file=None):
        self.chips = WCS_CHIPS
        self.data_file = data_file if data_file is not None else WCS_DATA_DIR + 'wcs.pkl'
        if os.path.isfile(self.data_file):
            print('initializing WCS data from file %s' % self.data_file)
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.__dict__.update(data.__dict__)
        else:
            print('preprocessed data file not found! loading WCS raw data...')
            self._langs = list(LANGS[0])
            self._lang_names = dict(zip(LANGS[0], LANGS[1]))
            self._names2ind = dict(zip(LANGS[1], LANGS[0]))
            self._lang_countries = dict(zip(LANGS[0], LANGS[2]))
            self._naming_data = process_wcs_naming_data()
            self.save()

    def __contains__(self, lang_name):
        return lang_name in self._lang_names.values()

    def get_features(self, cnum):
        return self.chips[cnum - 1]

    def all_langs(self):
        return self._langs.copy()

    def n_langs(self):
        return len(self._langs)

    def n_colors(self):
        return self.chips.shape[0]

    def lang_name(self, lang_id, short=False):
        if short:
            return self._lang_names[lang_id].split(' (')[0]
        return self._lang_names[lang_id]

    def lang_country(self, lang_id):
        return self._lang_countries[lang_id].split(' (')[0]

    def lang_name_to_id(self, name):
        return self._names2ind[name]

    def get_naming_data(self, lang_ids=None):
        if lang_ids is None:
            lang_ids = self._langs
        return {l: self._naming_data[l] for l in lang_ids if l in self._naming_data}

    def lang_nCW(self, lang_id):
        return self._naming_data[lang_id].nCW

    def lang_lex(self, lang_id):
        return self._naming_data[lang_id].lex

    def lang_pW_C(self, lang_id):
        return self._naming_data[lang_id].pW_C

    def save(self):
        with open(self.data_file, 'wb') as f:
            pickle.dump(self, f)


def get_lex(terms_l):
    lex = sorted(list(terms_l[3].unique()))
    if '*' in lex:
        lex.remove('*')
    return lex


def code2cnum(code):
    """
    convert WCS palette code to chip number
    Example: code2cnum('C22') returns 100
    :param code: string
    :return:
    """
    if code[0] == 'A':
        return _WCS_COR_CNUMS['A0']
    if code[0] == 'J':
        return _WCS_COR_CNUMS['J0']
    return _WCS_COR_CNUMS[code]


def rotate(P, rotation):
    perm = np.zeros(N_CHIPS, dtype=int)
    r = rotation if rotation > 0 else 40 + rotation
    for cnum, (row, col) in CNUMS_WCS_COR.items():
        if col > 0:
            col = col + r
            if col > 40:
                col -= 40
        perm[code2cnum('%s%d' % (ROWS[row], col)) - 1] = cnum - 1
    return P[perm]


def process_wcs_naming_data():
    naming_data = dict()
    langs = list(LANGS[0])
    for lang in langs:
        lang_data = Bunch()
        terms_l = TERMS_DF[TERMS_DF[0] == lang]
        lang_data.lex = get_lex(terms_l)
        K = len(lang_data.lex)
        lang_data.pW_C = np.zeros((N_CHIPS, K))
        lang_data.nCW = np.zeros((N_CHIPS, K))
        for i, w in enumerate(lang_data.lex):
            f = terms_l[terms_l[3] == w][2].values
            chips_w, counts = np.unique(f, return_counts=True)
            lang_data.nCW[chips_w - 1, i] = counts
        nC = lang_data.nCW.sum(axis=1)[:, None]
        if (lang_data.nCW.sum(axis=1) == 0).any():
            lang_data.pW_C = lang_data.nCW / (nC + 1e-20)
        else:
            lang_data.pW_C = lang_data.nCW / nC
        naming_data[lang] = lang_data
    return naming_data


if __name__ == '__main__':
    wcs = WCSDataset()
    print(wcs.get_features(1))
    print(wcs.lang_name_to_id('Agarabi'))
