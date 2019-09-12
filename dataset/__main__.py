import json
import os
import glob
import random

NUM_SAMPLE = 2000


def main():
    data_path = os.path.dirname(os.path.realpath(__file__))

    cat2id = {
        'ABBREVIATION': 0,
        'ENTITY': 1,
        'DESCRIPTION': 2,
        'HUMAN': 3,
        'LOCATION': 4,
        'NUMERIC': 5
    }

    id2cat = {v: k for k, v in cat2id.items()}

    with open(data_path + '/labels.json', 'w') as json_file:
        json.dump(cat2id, json_file, indent=2)

    abbreviation = []
    entity = []
    description = []
    human = []
    location = []
    numeric = []

    for lbl_file in glob.iglob(data_path + '/train_val/*.label'):
        with open(lbl_file) as fw:
            for line in fw:
                words = line.split()
                category = words.pop(0)
                if 'ABBR' in category:
                    abbreviation.append(words)
                elif 'ENTY' in category:
                    entity.append(words)
                elif 'DESC' in category:
                    description.append(words)
                elif 'HUM' in category:
                    human.append(words)
                elif 'LOC' in category:
                    location.append(words)
                elif 'NUM' in category:
                    numeric.append(words)

    # print(len(abbreviation))
    # print(len(entity))
    # print(len(description))
    # print(len(human))
    # print(len(location))
    # print(len(numeric))

    train_val = {}
    random.seed(42)
    train_val[cat2id['ABBREVIATION']] = random.sample(abbreviation, k=200)
    train_val[cat2id['ENTITY']] = random.sample(entity, k=NUM_SAMPLE)
    train_val[cat2id['DESCRIPTION']] = random.sample(description, k=NUM_SAMPLE)
    train_val[cat2id['HUMAN']] = random.sample(human, k=NUM_SAMPLE)
    train_val[cat2id['LOCATION']] = random.sample(location, k=NUM_SAMPLE)
    train_val[cat2id['NUMERIC']] = random.sample(numeric, k=NUM_SAMPLE)

    with open(data_path + '/train_val/train_val.json', 'w') as json_file:
        json.dump(train_val, json_file, indent=2)

    test = {}
    test[cat2id['ABBREVIATION']] = []
    test[cat2id['ENTITY']] = []
    test[cat2id['DESCRIPTION']] = []
    test[cat2id['HUMAN']] = []
    test[cat2id['LOCATION']] = []
    test[cat2id['NUMERIC']] = []
    with open(data_path + '/test/TREC_10.label') as lbl_file:
        for line in lbl_file:
            words = line.split()
            category = words.pop(0)
            if 'ABBR' in category:
                test[cat2id['ABBREVIATION']].append(words)
            elif 'ENTY' in category:
                test[cat2id['ENTITY']].append(words)
            elif 'DESC' in category:
                test[cat2id['DESCRIPTION']].append(words)
            elif 'HUM' in category:
                test[cat2id['HUMAN']].append(words)
            elif 'LOC' in category:
                test[cat2id['LOCATION']].append(words)
            elif 'NUM' in category:
                test[cat2id['NUMERIC']].append(words)

    with open(data_path + '/test/test.json', 'w') as json_file:
        json.dump(test, json_file, indent=2)


if __name__ == '__main__':
    main()
