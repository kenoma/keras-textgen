# -*- coding: utf-8 -*-
import re
import collections
import codecs
import nltk.data
import string

class TextChecker():
    def __init__(self):
        vocabular = ''
        with codecs.open('english.txt', "r",encoding='windows-1251', errors='strict') as fdata:
            vocabular += fdata.read()
        with codecs.open('russian.txt', "r",encoding='utf-8', errors='strict') as fdata:
            vocabular += fdata.read()
        with codecs.open('sc_vocabular.txt', "r",encoding='windows-1251', errors='strict') as fdata:
            vocabular += fdata.read()

        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.NWORDS = self.train(self.words(vocabular))
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчщшьъэюяABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧЩШЬЪЭЮЯ'

    def words(self, text):
        return re.findall(u'[A-Za-zА-Яа-я]+', text)

    def train(self, features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def edits1(self, word):
        s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in s if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in s for c in self.alphabet if b]
        inserts = [a + c + b     for a, b in s for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words):
        return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        if any(a in string.punctuation for a in word):
            return word
        
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        
        return max(candidates, key=self.NWORDS.get)

    def spellcheck_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        #pos_tags = nltk.pos_tag(tokens)
        #print(nltk.ne_chunk(pos_tags, binary=True))
        tokens = [self.correct(word) for word in tokens]
        
        return ''.join([' ' + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

    def correct_text(self, text):
        sentences = self.sent_tokenizer.tokenize(text)
        sentences = [self.spellcheck_sentence(sent).capitalize() for sent in sentences]
        return ' '.join(sentences)

    
    