from unidecode import unidecode
import numpy as np

chars = '\n !"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя…{|}~U'
charset = set(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def fix_char(c):
    if c.isupper():
        return 'U' + c.lower()
    elif c in charset:
        return c
    elif c == '\t':
        return '    '
    else:
        return ''

def encode(text):
    text = text.replace('\r','').replace('–','-').replace('€','$').replace('£','$').replace('₽','$')
    return ''.join(fix_char(c) for c in text)


def decode(chars):
    upper = False
    for c in chars:
        if c == 'U':
            upper = True
        elif upper:
            upper = False
            yield c.upper()
        else:
            yield c

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)




