﻿import re
import nltk, string

class RussianTextPreprocessing():
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.stopwords = {u'000anchor':0, u'000newline':0,  u'а':0 , u'в':0 , u'г':0 , u'е':0 , u'ж':0 , u'и':0 , u'к':0 , u'м':0 , u'о':0 , u'с':0 , u'т':0 , u'у':0 , u'я':0 , u'бы':0 , u'во':0 , u'вы':0 , u'да':0 , u'до':0 , u'ее':0 , u'ей':0 , u'ею':0 , u'её':0 , u'же':0 , u'за':0 , u'из':0 , u'им':0 , u'их':0 , u'ли':0 , u'мы':0 , u'на':0 , u'не':0 , u'ни':0 , u'но':0 , u'ну':0 , u'нх':0 , u'об':0 , u'он':0 , u'от':0 , u'по':0 , u'со':0 , u'та':0 , u'те':0 , u'то':0 , u'ту':0 , u'ты':0 , u'уж':0 , u'без':0 , u'был':0 , u'вам':0 , u'вас':0 , u'ваш':0 , u'вон':0 , u'вот':0 , u'все':0 , u'всю':0 , u'вся':0 , u'всё':0 , u'где':0 , u'год':0 , u'два':0 , u'две':0 , u'дел':0 , u'для':0 , u'его':0 , u'ему':0 , u'еще':0 , u'ещё':0 , u'или':0 , u'ими':0 , u'имя':0 , u'как':0 , u'кем':0 , u'ком':0 , u'кто':0 , u'лет':0 , u'мне':0 , u'мог':0 , u'мож':0 , u'мои':0 , u'мой':0 , u'мор':0 , u'моя':0 , u'моё':0 , u'над':0 , u'нам':0 , u'нас':0 , u'наш':0 , u'нее':0 , u'ней':0 , u'нем':0 , u'нет':0 , u'нею':0 , u'неё':0 , u'них':0 , u'оба':0 , u'она':0 , u'они':0 , u'оно':0 , u'под':0 , u'пор':0 , u'при':0 , u'про':0 , u'раз':0 , u'сам':0 , u'сих':0 , u'так':0 , u'там':0 , u'тем':0 , u'тех':0 , u'том':0 , u'тот':0 , u'тою':0 , u'три':0 , u'тут':0 , u'уже':0 , u'чем':0 , u'что':0 , u'эта':0 , u'эти':0 , u'это':0 , u'эту':0 , u'алло':0 , u'буду':0 , u'будь':0 , u'бывь':0 , u'была':0 , u'были':0 , u'было':0 , u'быть':0 , u'вами':0 , u'ваша':0 , u'ваше':0 , u'ваши':0 , u'ведь':0 , u'весь':0 , u'вниз':0 , u'всем':0 , u'всех':0 , u'всею':0 , u'года':0 , u'году':0 , u'даже':0 , u'двух':0 , u'день':0 , u'если':0 , u'есть':0 , u'зато':0 , u'кого':0 , u'кому':0 , u'куда':0 , u'лишь':0 , u'люди':0 , u'мало':0 , u'меля':0 , u'меня':0 , u'мимо':0 , u'мира':0 , u'мной':0 , u'мною':0 , u'мочь':0 , u'надо':0 , u'нами':0 , u'наша':0 , u'наше':0 , u'наши':0 , u'него':0 , u'нему':0 , u'ниже':0 , u'ними':0 , u'один':0 , u'пока':0 , u'пора':0 , u'пять':0 , u'рано':0 , u'сама':0 , u'сами':0 , u'само':0 , u'саму':0 , u'свое':0 , u'свои':0 , u'свою':0 , u'себе':0 , u'себя':0 , u'семь':0 , u'стал':0 , u'суть':0 , u'твой':0 , u'твоя':0 , u'твоё':0 , u'тебе':0 , u'тебя':0 , u'теми':0 , u'того':0 , u'тоже':0 , u'тому':0 , u'туда':0 , u'хоть':0 , u'хотя':0 , u'чаще':0 , u'чего':0 , u'чему':0 , u'чтоб':0 , u'чуть':0 , u'этим':0 , u'этих':0 , u'этой':0 , u'этом':0 , u'этот':0 , u'более':0 , u'будем':0 , u'будет':0 , u'будто':0 , u'будут':0 , u'вверх':0 , u'вдали':0 , u'вдруг':0 , u'везде':0 , u'внизу':0 , u'время':0 , u'всего':0 , u'всеми':0 , u'всему':0 , u'всюду':0 , u'давно':0 , u'даром':0 , u'долго':0 , u'друго':0 , u'жизнь':0 , u'занят':0 , u'затем':0 , u'зачем':0 , u'здесь':0 , u'иметь':0 , u'какая':0 , u'какой':0 , u'когда':0 , u'кроме':0 , u'лучше':0 , u'между':0 , u'менее':0 , u'много':0 , u'могут':0 , u'может':0 , u'можно':0 , u'можхо':0 , u'назад':0 , u'низко':0 , u'нужно':0 , u'одной':0 , u'около':0 , u'опять':0 , u'очень':0 , u'перед':0 , u'позже':0 , u'после':0 , u'потом':0 , u'почти':0 , u'пятый':0 , u'разве':0 , u'рядом':0 , u'самим':0 , u'самих':0 , u'самой':0 , u'самом':0 , u'своей':0 , u'своих':0 , u'сеаой':0 , u'снова':0 , u'собой':0 , u'собою':0 , u'такая':0 , u'также':0 , u'такие':0 , u'такое':0 , u'такой':0 , u'тобой':0 , u'тобою':0 , u'тогда':0 , u'тысяч':0 , u'уметь':0 , u'часто':0 , u'через':0 , u'чтобы':0 , u'шесть':0 , u'этими':0 , u'этого':0 , u'этому':0 , u'близко':0 , u'больше':0 , u'будете':0 , u'будешь':0 , u'бывает':0 , u'важная':0 , u'важное':0 , u'важные':0 , u'важный':0 , u'вокруг':0 , u'восемь':0 , u'всегда':0 , u'второй':0 , u'далеко':0 , u'дальше':0 , u'девять':0 , u'десять':0 , u'должно':0 , u'другая':0 , u'другие':0 , u'других':0 , u'другое':0 , u'другой':0 , u'занята':0 , u'занято':0 , u'заняты':0 , u'значит':0 , u'именно':0 , u'иногда':0 , u'каждая':0 , u'каждое':0 , u'каждые':0 , u'каждый':0 , u'кругом':0 , u'меньше':0 , u'начала':0 , u'нельзя':0 , u'нибудь':0 , u'никуда':0 , u'ничего':0 , u'обычно':0 , u'однако':0 , u'одного':0 , u'отсюда':0 , u'первый':0 , u'потому':0 , u'почему':0 , u'просто':0 , u'против':0 , u'раньше':0 , u'самими':0 , u'самого':0 , u'самому':0 , u'своего':0 , u'сейчас':0 , u'сказал':0 , u'совсем':0 , u'теперь':0 , u'только':0 , u'третий':0 , u'хорошо':0 , u'хотеть':0 , u'хочешь':0 , u'четыре':0 , u'шестой':0 , u'восьмой':0 , u'впрочем':0 , u'времени':0 , u'говорил':0 , u'говорит':0 , u'девятый':0 , u'десятый':0 , u'кажется':0 , u'конечно':0 , u'которая':0 , u'которой':0 , u'которые':0 , u'который':0 , u'которых':0 , u'наверху':0 , u'наконец':0 , u'недавно':0 , u'немного':0 , u'нередко':0 , u'никогда':0 , u'однажды':0 , u'посреди':0 , u'сегодня':0 , u'седьмой':0 , u'сказала':0 , u'сказать':0 , u'сколько':0 , u'слишком':0 , u'сначала':0 , u'спасибо':0 , u'человек':0 , u'двадцать':0 , u'довольно':0 , u'которого':0 , u'наиболее':0 , u'недалеко':0 , u'особенно':0 , u'отовсюду':0 , u'двадцатый':0 , u'миллионов':0 , u'несколько':0 , u'прекрасно':0 , u'процентов':0 , u'четвертый':0 , u'двенадцать':0 , u'непрерывно':0 , u'пожалуйста':0 , u'пятнадцать':0 , u'семнадцать':0 , u'тринадцать':0 , u'двенадцатый':0 , u'одиннадцать':0 , u'пятнадцатый':0 , u'семнадцатый':0 , u'тринадцатый':0 , u'шестнадцать':0 , u'восемнадцать':0 , u'девятнадцать':0 , u'одиннадцатый':0 , u'четырнадцать':0 , u'шестнадцатый':0 , u'восемнадцатый':0 , u'девятнадцатый':0 , u'действительно':0 , u'четырнадцатый':0 , u'многочисленная':0 , u'многочисленное':0 , u'многочисленные':0 , u'многочисленный':0}
        self.r = re.compile(r'[^0-9a-zA-Zа-яА-Я\s\!\"\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^_\`\{\|\}\~]+')#.format(re.escape(string.punctuation)
        #self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчщшьъэюяABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧЩШЬЪЭЮЯ'+string.punctuation
        
    def ngrams(self, word, n):
        output = []
        tmp = ''
        for i in range(len(word)):
            tmp += word[i]
            if (i + 1) % n == 0:
                output.append(tmp)
                tmp = ''

        if tmp is not '':
            output.append(tmp)
        return output

    def sentence_to_tokens(self, sentence, token_lenght=2):
        rev_words = nltk.word_tokenize(sentence)
        water_index = 0
        sent_len = 0
    
        retval = []
        for a in rev_words:
            if a in self.stopwords:
                water_index += 1
                self.stopwords[a] += 1
                retval.append([' ', water_index, sent_len, 0])
                retval.append(['_' + a.lower(), water_index, sent_len, 1 if a[0].isupper() else 0])
            else:
                if not a in string.punctuation:
                    sent_len += 1
                    retval.append([' ', water_index, sent_len, 0])
                    retval.extend([[g.lower(), water_index, sent_len, 1 if g[0].isupper() else 0] for g in self.ngrams(a, token_lenght)])
                else:
                    retval.append([a, water_index, sent_len, 0])

        return retval

    def review_to_sentences(self, review, token_lenght=2):
        review=self.r.sub(' 000anchor ',review)
        review = review.replace('\r\n',' 000newline ')\
                        .replace('\'\'',' ')\
                        .replace('“',' ')\
                        .replace('”',' ')\
                        .replace('`',' ')\
                        .replace('`',' ')\
                        .replace('ё','е')\
                        .replace('"',' ')\
                        .replace('\'','')\
                        .replace('\\',' \\ ')\
                        .replace('--','-')\
                        .replace('--','-')\
                        .replace('--','-')\
                        .replace('/',' \ ')\
                        .replace('^','')\
                        .replace('+',' + ')\
                        .replace('=',' = ')\
                        .replace('-',' - ')\
                        .replace(':',' : ')\
                        .replace('_',' ')

        raw_sentences = self.tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                s = self.sentence_to_tokens(raw_sentence, token_lenght)
                sentences.append(s)
        return sentences

    def count_stats(self, sentence, water_max, senlen_max):
        wc = 0
        sc = 0
        for w in sentence.split(' '):
            if '_' in w:
                wc += 1
            else:
                sc += 1
            if w == '.':
                wc = 0
                sc = 0

        return min(water_max, wc), min(senlen_max, sc)



