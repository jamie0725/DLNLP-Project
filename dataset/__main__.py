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

    for lbl_file in glob.iglob(data_path + '/train/*.label'):
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

    train = {}
    val = {}
    train_portion = 0.9

    # Shuffle the data.
    random.seed(42)
    random.shuffle(abbreviation)
    random.shuffle(entity)
    random.shuffle(description)
    random.shuffle(human)
    random.shuffle(location)
    random.shuffle(numeric)

    # Truncate the data.
    abbreviation = abbreviation[:200]
    entity = entity[:NUM_SAMPLE]
    description = description[:NUM_SAMPLE]
    human = human[:NUM_SAMPLE]
    location = location[:NUM_SAMPLE]
    numeric = numeric[:NUM_SAMPLE]

    # Split into training set and validation set.
    train[cat2id['ABBREVIATION']] = abbreviation[:int(200 * train_portion)]
    train[cat2id['ENTITY']] = entity[:int(NUM_SAMPLE * train_portion)]
    train[cat2id['DESCRIPTION']] = description[:int(NUM_SAMPLE * train_portion)]
    train[cat2id['HUMAN']] = human[:int(NUM_SAMPLE * train_portion)]
    train[cat2id['LOCATION']] = location[:int(NUM_SAMPLE * train_portion)]
    train[cat2id['NUMERIC']] = numeric[:int(NUM_SAMPLE * train_portion)]

    val[cat2id['ABBREVIATION']] = abbreviation[int(200 * train_portion):]
    val[cat2id['ENTITY']] = entity[int(NUM_SAMPLE * train_portion):]
    val[cat2id['DESCRIPTION']] = description[int(NUM_SAMPLE * train_portion):]
    val[cat2id['HUMAN']] = human[int(NUM_SAMPLE * train_portion):]
    val[cat2id['LOCATION']] = location[int(NUM_SAMPLE * train_portion):]
    val[cat2id['NUMERIC']] = numeric[int(NUM_SAMPLE * train_portion):]

    with open(data_path + '/train/train.json', 'w') as json_file:
        json.dump(train, json_file, indent=2)

    with open(data_path + '/val/val.json', 'w') as json_file:
        json.dump(val, json_file, indent=2)

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
