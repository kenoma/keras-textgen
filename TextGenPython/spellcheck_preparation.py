import codecs,glob
import nltk.data, string

class spellcheck_preparation(object):
    """description of class"""
    def prepare(self, path_to_sources="D:\projects\TextGenPython\software\*.txt"):
        all_data=''
        for filename in glob.glob(path_to_sources):
            print(filename)
            with codecs.open(filename, "r",encoding='windows-1251', errors='strict') as fdata:
                all_data += fdata.read()
        
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_tokenizer.tokenize(all_data)
        dict={}
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            tokens[0] = tokens[0].lower()
            for token in tokens:
                if any(a in string.punctuation for a in token):
                    continue
                dict[token] = 1

        with open("sc_vocabular.txt", "w", encoding="windows-1251") as file:
            for a in dict.keys():
                file.write('%s\r'%a)
           
        



