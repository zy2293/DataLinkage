import json
import csv
import pandas as pd
import numpy as np
import unicodedata
import recordlinkage
from sklearn.ensemble import RandomForestClassifier
import usaddress
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import warnings

warnings.filterwarnings("ignore")

"""
This assignment can be done in groups of 3 students. Everyone must submit individually.

Write down the UNIs of your group (if applicable)

Name : Zihan Ye
Uni  : zy2293

Member 2: Jiachen Xu
Uni    2: jx2318

Member 3: name, uni
"""



def load_data(foursquare_path, locu_path):
    foursquare = pd.read_json(foursquare_path)
    locu = pd.read_json(locu_path)
    return foursquare, locu


# # validate pairs by other attributes

def validate(predictions, address_locu, address_foursquare, attr):
    locu_y = address_locu["id"]
    foursquare_y = address_foursquare["id"]
    count = 0
    matched = {}
    Nmatched = {}
    locu = []
    fq = []

    for key, value in predictions.items():
        locu_key = address_locu.index[locu_y == key].tolist()
        fs_key = address_foursquare.index[foursquare_y == value].tolist()

        if (address_locu[attr][locu_key].tolist()) == (address_foursquare[attr][fs_key].tolist()):
            matched[key] = value
            count += 1
            locu.append(key)
            fq.append(value)
        else:
            locu_value = address_locu["phone"][locu_key].tolist()
            fs_value = address_foursquare["phone"][fs_key].tolist()
            if locu_value == fs_value:
                matched[key] = value
                count += 1
                locu.append(key)
                fq.append(value)

            else:
                locu_zipcode = address_locu["latitude"][locu_key].tolist()
                fq_zipcode = address_foursquare["latitude"][fs_key].tolist()
                if locu_zipcode == fq_zipcode:

                    matched[key] = value
                    count += 1
                    locu.append(key)
                    fq.append(value)
                else:
                    Nmatched[key] = value

    # print ("validated ", count)

    return matched, Nmatched, locu, fq


# # parse address


def parseAddress(df, attrs):
    df1 = df[df[attrs].apply(lambda x: len(x) != 0)]
    index = df1.index.tolist()
    for ind in index:
        try:
            tags = dict(usaddress.tag(df[attrs][ind])[0])
            key = list(tags.keys())
            if 'AddressNumber' in key:
                df['AddressNumber'][ind] = tags['AddressNumber']

            if 'StreetName' in key:
                df['StreetName'][ind] = tags['StreetName']

            if 'StreetNamePreDirectional' in key:
                df['StreetNamePreDirectional'][ind] = tags['StreetNamePreDirectional']

        except:
            pass
    return df


# # address, phone normalization

def normalize(df):
    df["phone"].replace(np.nan, '', inplace=True)
    df["phone"] = df["phone"].apply(lambda x: x.replace('-', '').replace(')', '').replace('(', '').replace(' ', ''))
    df['StreetNamePreDirectional'] = df['StreetNamePreDirectional'].apply(
        lambda x: x.replace('W.', 'West').replace('E.', 'East').replace('N.', 'North').replace('S.', 'South'))

    return df


# # URL parse

from urllib.parse import urlparse


def parseUrl(df, attrs):
    df1 = df[df[attrs].apply(lambda x: len(x) != 0)]
    index = df1.index.tolist()
    for ind in index:
        o = urlparse(df[attrs][ind])
        df["url"][ind] = o.netloc
    return df


def preprocess(testdf):
    # parse street address to AddressNumber, StreetName, StreetNamePreDirectional
    testdf['AddressNumber'] = ''
    testdf['StreetName'] = ''
    testdf['StreetNamePreDirectional'] = ''
    testdf = parseAddress(testdf, "street_address")

    # Normalization
    testdf = normalize(testdf)

    # Urlparse
    testdf["url"] = ''
    testdf = parseUrl(testdf, "website")

    return testdf


# # train models, applying svc, rf, knn
# # No record duplicate in each file

def trainModels(model, address_foursquare, address_locu):
    select_col = ["latitude", "longitude"]
    foursquareTrain = address_foursquare[select_col]
    locuTrain = address_locu[select_col]

    foursquare_y = address_foursquare["id"]
    locu_y = address_locu["id"]

    foursquareTrain.fillna(0, inplace=True)

    locuTrain.fillna(0, inplace=True)

    if model == "svc":
        clf = svm.SVC(decision_function_shape='ovo')
    elif model == "rf":
        clf = RandomForestClassifier(max_leaf_nodes=None, random_state=0)
    elif model == "knn":
        clf = KNeighborsClassifier(n_neighbors=1)

    clf.fit(foursquareTrain, foursquare_y)
    locu_predict = clf.predict(locuTrain)

    predictions = dict(zip(locu_y, locu_predict))
    knn_predictions, Nvalidated, locu, fq = validate(predictions, address_locu, address_foursquare, "name")

    # delete ones already validated matched
    train_foursquare = address_foursquare[~address_foursquare["id"].isin(fq)]
    train_locu = address_locu[~address_locu["id"].isin(locu)]

    return knn_predictions, Nvalidated, train_foursquare, train_locu


# # Compare Address

def compareString(address_foursquare, address_locu, attrs):
    foursquare_Address = address_foursquare[address_foursquare[attrs].apply(lambda x: len(x) != 0)]

    foursquareAddress = foursquare_Address[attrs]
    foursquareIndex = foursquareAddress.index.tolist()

    locu_Address = address_locu[address_locu[attrs].apply(lambda x: len(x) != 0)]
    locuAddress = locu_Address[attrs]
    locuIndex = locuAddress.index.tolist()

    candidate_links = pd.MultiIndex.from_product([locuIndex, foursquareIndex], names=['locu', 'foursquare'])

    return locuIndex, foursquareIndex, locu_Address, foursquare_Address, candidate_links


def levenshtein(candidate_links, locuIndex, locu_Address, foursquare_Address, attrs):
    comp = recordlinkage.Compare()
    for att in attrs:
        comp.string(att, att, method='damerau_levenshtein')
    levenDist = comp.compute(candidate_links, locu_Address, foursquare_Address)

    records = locu_Address.shape[0]
    pairs = {}
    for index in locuIndex:
        level = levenDist.loc[levenDist.index.get_level_values("locu") == index].reset_index()
        level['distance'] = level[0] ** 2 + level[1] ** 2 + level[2] ** 2
        co = np.argmax(level['distance'].tolist())
        if level['distance'].tolist()[co] > 2.9:
            match_index = level['foursquare'][co]
            if locu_Address["name"][index] != "Lizzie's Restaurant":
                pairs[locu_Address["id"][index]] = foursquare_Address["id"][match_index]

    return pairs


def compareAddress(knn_foursquare, knn_locu, attrs):
    locuIndex, foursquareIndex, locu_Address, foursquare_Address, candidate_links = compareString(knn_foursquare,
                                                                                                  knn_locu, attrs)
    attributes = ['AddressNumber', 'StreetName', 'StreetNamePreDirectional']
    levenshtainpairs = levenshtein(candidate_links, locuIndex, locu_Address, foursquare_Address, attributes)

    matched, Nmatched, locu, fq = validate(levenshtainpairs, knn_locu, knn_foursquare, "name")
    left_foursquare = knn_foursquare[~knn_foursquare["id"].isin(fq)]
    left_locu = knn_locu[~knn_locu["id"].isin(locu)]
    return matched, Nmatched, left_foursquare, left_locu, levenshtainpairs


# # Compare phone/name

def comparePhone(train_foursquare, train_locu, attribute):
    notnull_foursquare = train_foursquare[train_foursquare[attribute] != '']
    notnull_locu = train_locu[train_locu[attribute] != '']

    columns = ["id", attribute]
    compare_foursquare = notnull_foursquare[columns]
    compare_locu = notnull_locu[columns]

    # merge attribute
    compare_result = compare_foursquare.merge(compare_locu, left_on=compare_foursquare[attribute],
                                              right_on=compare_locu[attribute], how="inner")

    compare_cols = ["id_x", "id_y"]  # x: foursquare, y: locu
    predict = compare_result[compare_cols]
    predict = predict.set_index("id_y")
    predict = predict.to_dict()
    compare_predict = predict["id_x"]

    matched, Nmatched, locu, fq = validate(compare_predict, train_locu, train_foursquare, "name")

    train_foursquare = train_foursquare[~train_foursquare["id"].isin(fq)]
    train_locu = train_locu[~train_locu["id"].isin(locu)]

    return matched, Nmatched, train_foursquare, train_locu


# # URL

def levenshteinUrl(candidate_links, locuIndex, locu_Address, foursquare_Address, attrs):
    comp = recordlinkage.Compare()
    comp.string(attrs, attrs, method='damerau_levenshtein')
    levenDist = comp.compute(candidate_links, locu_Address, foursquare_Address)

    records = locu_Address.shape[0]
    pairs = {}
    for index in locuIndex:
        level = levenDist.loc[levenDist.index.get_level_values("locu") == index].reset_index()
        co = np.argmax(level[0].tolist())
        match_index = level['foursquare'][co]
        if level[0].tolist()[co] > 0.8 and (
            locu_Address["postal_code"][index] == foursquare_Address['postal_code'][match_index]):
            pairs[locu_Address["id"][index]] = foursquare_Address["id"][match_index]

    return pairs


def compareUrl(name_foursquare, name_locu, attrs):
    locuIndex, foursquareIndex, locu_Address, foursquare_Address, candidate_links = compareString(name_foursquare,
                                                                                                  name_locu, attrs)
    urlpairs = levenshteinUrl(candidate_links, locuIndex, locu_Address, foursquare_Address, attrs)
    return urlpairs


# # LEFT records

def removeAccents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return str(only_ascii)


def levenshteinName(candidate_links, locuIndex, locu_Address, foursquare_Address, attrs):
    comp = recordlinkage.Compare()
    comp.string(attrs, attrs, method='damerau_levenshtein')
    levenDist = comp.compute(candidate_links, locu_Address, foursquare_Address)

    pairs = {}
    for index in locuIndex:
        level = levenDist.loc[levenDist.index.get_level_values("locu") == index].reset_index()
        co = np.argmax(level[0].tolist())
        match_index = level['foursquare'][co]
        locu_add = locu_Address["street_address"][index]
        foursquare_add = foursquare_Address["street_address"][match_index]
        locu_zip = locu_Address["postal_code"][index]
        foursquare_zip = foursquare_Address["postal_code"][match_index]

        log1 = len(locu_add) == 0 or len(foursquare_add) == 0  # one of addresses missing
        log2 = locu_zip == foursquare_zip  # same zip code
        log3 = ((len(locu_zip) == 0) or len(foursquare_zip) == 0)  # one of zipcodes missing

        if level[0].tolist()[co] > 0.9:
            pairs[locu_Address["id"][index]] = foursquare_Address["id"][match_index]

        elif level[0].tolist()[co] > 0.7 and log1 and (log2 or log3):

            pairs[locu_Address["id"][index]] = foursquare_Address["id"][match_index]
    return pairs


def NameCompare(name_foursquare, name_locu, attrs):
    name_foursquare[attrs] = name_foursquare[attrs].apply(lambda x: removeAccents(x))
    name_locu[attrs] = name_locu[attrs].apply(lambda x: removeAccents(x))

    locuIndex, foursquareIndex, locu_Address, foursquare_Address, candidate_links = compareString(name_foursquare,
                                                                                                  name_locu, attrs)
    namepairs = levenshteinName(candidate_links, locuIndex, locu_Address, foursquare_Address, attrs)

    return namepairs


# test algorithm against on  match train result
def toDict(matches_train_path):
    match = pd.read_csv(matches_train_path)
    match = match.set_index("locu_id")
    match = match.to_dict()

    compare_match = match["foursquare_id"]
    return compare_match



def matchRecords(matches_train_path,  compare_predict):
    match_dict = toDict(matches_train_path)
    matched = {}
    Nmatched = {}
    match_count = 0
    Nmatch_count = 0
    keylist = match_dict.keys()
    locu = []
    fq = []

    for key, value in compare_predict.items():
        if key in keylist and match_dict[key] == value:
            match_count += 1
            matched[key] = value
            locu.append(key)
            fq.append(value)
        else:
            Nmatch_count += 1
            Nmatched[key] = value

    precision = match_count * 1.0 / (len(compare_predict.keys()))
    recall = match_count * 1.0 / (len(keylist))

    F1 = 2 * precision * recall / (precision + recall)
    print("matched records: {}".format(match_count))
    print("not matched records: {}".format(Nmatch_count))
    print("train datasets precision: {}, recall: {}, F1-score: {}".format(precision, recall, F1))

 #   return locu, fq, matched, Nmatched


# # Pipeline

def pipeLine(test_locu, test_foursquare):
    locu = preprocess(test_locu)
    foursquare = preprocess(test_foursquare)

    # KNN
    knn_validated, knn_Nvalidated, knn_foursquare, knn_locu = trainModels("svc", foursquare, locu)
    # Compare Address after KNN
    address_matched, address_Nmatched, left_foursquare, left_locu, addresspairs = compareAddress(knn_foursquare,
                                                                                                 knn_locu,
                                                                                                 attrs="street_address")
    # Compare Phone & Name after Address Compare
    phone_matched, phone_Nmatched, phone_foursquare, phone_locu = comparePhone(left_foursquare, left_locu, "phone")
    name_matched, name_Nmatched, name_foursquare, name_locu = comparePhone(left_foursquare, left_locu, "name")
    # Compare url after Name Compare
    urlpairs = compareUrl(name_foursquare, name_locu, attrs="url")
    # Compare Name after Name Compare
    namepairs = NameCompare(name_foursquare, name_locu, attrs="name")

    return knn_validated, addresspairs, phone_matched, name_matched, urlpairs, namepairs


def execute(locu, foursquare):
    knn_validated, addresspairs, phone_matched, name_matched, urlpairs, namepairs = pipeLine(locu, foursquare)

    result = addresspairs.copy()
    result.update(knn_validated)
    result.update(name_matched)
    result.update(phone_matched)
    result.update(namepairs)
    result.update(urlpairs)

    return result




def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):
    """
        In this function, You need to design your own algorithm or model to find the matches and generate
        a matches_test.csv in the current folder.

        you are given locu_train, foursquare_train json file path and matches_train.csv path to train
        your model or algorithm.

        Then you should test your model or algorithm with locu_test and foursquare_test json file.
        Make sure that you write the test matches to a file in the same directory called matches_test.csv.

    """

    # train algorithm on given train foursquare locu datasets
    # print precision, recall, F1 score of algorithm on give match_train datasets
    train_foursquare, train_locu = load_data(foursquare_train_path, locu_train_path)
    train_result = execute(train_locu, train_foursquare)
    matchRecords(matches_train_path, train_result)


    test_foursquare, test_locu = load_data(foursquare_test_path, locu_test_path)
    test_result = execute(test_locu, test_foursquare)
    predictions = pd.DataFrame.from_dict(test_result, orient='index')
    predictions.reset_index(inplace=True)
    predictions.columns = ["locu_id", "foursquare_id"]
    predictions.to_csv("matches_test.csv", encoding='utf-8', index = False)

