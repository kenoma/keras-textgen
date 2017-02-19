# -*- coding: utf-8 -*-
import unittest
from RussianTextPreprocessing import RussianTextPreprocessing
from TextChecker import TextChecker

class TestFuncs(unittest.TestCase):

    def test_ngrams_1(self):
        proc = RussianTextPreprocessing()
        res = proc.ngrams(u'это',2)
        self.assertEqual(len(res), 2)

    def test_ngrams_2(self):
        proc = RussianTextPreprocessing()
        res = proc.ngrams(u'во',2)
        self.assertEqual(len(res), 1)

    def test_ngrams_3(self):
        proc = RussianTextPreprocessing()
        res = proc.ngrams(u'боль',2)
        self.assertEqual(len(res), 2)

    def test_ngrams_4(self):
        proc = RussianTextPreprocessing()
        res = proc.ngrams(u'вовка',2)
        self.assertEqual(len(res), 3)

    def test_ngrams_5(self):
        proc = RussianTextPreprocessing()
        res = proc.sentence_to_tokens(u'вовка внес залог')
        self.assertEqual(len(res), 11)
        self.assertEqual(res[10][1], 0)
        self.assertEqual(res[10][2], 3)

    def test_ngrams_capitalization(self):
        proc = RussianTextPreprocessing()
        res = proc.sentence_to_tokens(u'Вовка внес залог за Windows 10')
        self.assertEqual(len(res), 20)
        self.assertEqual(res[1][3], 1)
        self.assertEqual(res[2][3], 0)
        self.assertEqual(res[13][3], 0)
        self.assertEqual(res[14][3], 1)
        self.assertEqual(res[15][3], 0)

    def test_ngrams_6(self):
        proc = RussianTextPreprocessing()
        res = proc.sentence_to_tokens(u'вовка не смогуща')
        self.assertEqual(len(res), 11)
        self.assertTrue('_' in res[5][0])
        self.assertEqual(res[10][1], 1)
        self.assertEqual(res[10][2], 2)

    def test_ngrams_capitalized(self):
        proc = RussianTextPreprocessing()
        res = proc.sentence_to_tokens(u'вовка Не смогуща')
        self.assertEqual(len(res), 11)
        self.assertTrue('_' in res[5][0])
        self.assertEqual(res[10][1], 1)
        self.assertEqual(res[10][2], 2)

    def test_ngrams_7(self):
        proc = RussianTextPreprocessing()
        res = proc.sentence_to_tokens(u'вовка, остынь!')
        self.assertEqual(len(res), 10)
        self.assertEqual(res[4][0], ',')
        self.assertEqual(res[9][0], '!')
        self.assertEqual(res[9][1], 0)
        self.assertEqual(res[9][2], 2)

    def test_ngrams_8(self):
        proc = RussianTextPreprocessing()
        water,senlen = proc.count_stats(u'вовка _и детерминизм морали.',100,100)
        self.assertEqual(water, 1)
        self.assertEqual(senlen, 3)

    #def test_review_to_sentences_9(self):
    #    proc = RussianTextPreprocessing()
    #    res = proc.review_to_sentences('И она заявила # первое\r\nи второе')
    #    self.assertEqual(len(res), 10)
    #    self.assertEqual(res[4][0], ',')
    #    self.assertEqual(res[9][0], '!')
    #    self.assertEqual(res[9][1], 0)
    #    self.assertEqual(res[9][2], 2)


    def test_ngrams_10(self):
        proc = RussianTextPreprocessing()
        water,senlen = proc.count_stats(u'вовка _и детерминизм морали',100,100)
        self.assertEqual(water, 1)
        self.assertEqual(senlen, 3)
        
    def test_spellchecker_1(self):
        spcheck = TextChecker()
        res = spcheck.correct('abaljienated')
        self.assertEqual(res, 'abalienated');

    def test_spellchecker_2(self):
        spcheck = TextChecker()
        res = spcheck.correct('фортпиано')
        self.assertEqual(res, 'фортепиано');

    def test_spellchecker_3(self):  
        spcheck = TextChecker()
        res = spcheck.correct('алжир')
        self.assertEqual(res, 'Алжир');

    def test_correct_text_1(self):
        spcheck = TextChecker()
        res = spcheck.correct_text('a period might occur inside a sentence e.g., see! and the sentence may ends without the dot!')
        self.assertEqual(res, 'A period might occur inside a sentence e.g., see! And the sentence may ends without the dot!');

    #def test_correct_text_2(self):
    #    spcheck = TextChecker()
    #    res = spcheck.correct_text('новый проект на конференц game informer 2016. e3 2016: прямая трансляция the witness появится на xbox one, а версия консоли xbox one для playstation 4. по словам разработчиков, в конце 2016 года на playstation 4 и xbox one. e3 2016: the witness появилась на пресс конференции microsoft на e3 2016. e3 2016: презентации проекта появилась на конференции microsoft в рамках e3 2016. e3 2016: представители the witness получили контактный возможность взломать и раньше пользователей. в прошлом году представители разработчиков и компаньона, но в продажу поступит на пресс конференции microsoft в рамках е3 2016. в настоящее время покажет проект проекта для xbox one. также на выставке е3 2016 появилась возможности представители проекта и компаньона на протяжении нескольких лет, а также компания подтвердила, что в ролике будет поддерживать компанию с ', None)
    #    self.assertEqual(res, 'A period might occur inside a sentence e.g., see! And the sentence may ends without the dot!');

if __name__ == '__main__':
    unittest.main()