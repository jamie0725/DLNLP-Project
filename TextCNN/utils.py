import sys
import os
import json


class Logger(object):
    '''
    Export print to logs.
    '''

    def __init__(self, file_loc):
        self.terminal = sys.stdout
        self.log = open(file_loc, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def load_json(file_loc, mapping=None, reverse=False, name=None):
    '''
    Load json file at a given location.
    '''

    with open(str(file_loc), 'r') as file:
        data = json.load(file)
        file.close()
    if reverse:
        data = dict(map(reversed, data.items()))
    print_logs(data, mapping, name)
    return data


def print_logs(data, mapping, name, keep=3):
    '''
    Print detailed information of given data iterator.
    '''

    print('Statistics of {}:'.format(name))
    try:
        count = 0
        for key in data:
            print('* Number of data in class {}: {}'.format(mapping[int(key)][:keep], len(data[key])))
            count += len(data[key])
        print('+ Total: {}'.format(count))
    except:
        print('* Class: {}'.format(data))
        print('+ Total: {}'.format(len(data)))


def print_statement(statement, isCenter=False, symbol='=', number=15, newline=False):
    '''
    Print required statement in a given format.
    '''

    if newline:
        print()
    if number > 0:
        prefix = symbol * number + ' '
        suffix = ' ' + symbol * number
        statement = prefix + statement + suffix
    if isCenter:
        print(statement.center(os.get_terminal_size().columns))
    else:
        print(statement)


def print_flags(args):
    """
    Print all entries in args variable.
    """

    for key, value in vars(args).items():
        print(key + ' : ' + str(value))


def print_result(result, keep=3):
    """
    Print result matrix.
    """

    if type(result) == dict:
        # Individual result.
        for key in result:
            prec = result[key]['precision']
            rec = result[key]['recall']
            f1 = result[key]['f1score']
            key = key.replace('__label__', '')[:keep]
            print('* {} PREC: {:.2f}, {} REC: {:.2f}, {} F1: {:.2f}'.format(key, prec, key, rec, key, f1))
    elif type(result) == tuple:
        # Overall result.
        print('Testing on {} data:'.format(result[0]))
        print('+ Overall ACC: {:.3f}'.format(result[1]))
        assert result[1] == result[2]
    else:
        raise TypeError


def print_value(name, value):
    print(name + f':{value}')
