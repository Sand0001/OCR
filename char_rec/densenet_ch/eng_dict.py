import re
class eng_dict():
    def __init__(self,corpus_path):
        self.corpus = corpus_path
        self.word_dict = {}
        self.load_corpus()
    def load_corpus(self):
        courpus = open(self.corpus, 'r')
        courpus_lines = courpus.readlines()
        for line in courpus_lines:
            line = line.strip()
            line = re.compile(r'[^a-zA-Z0-9_/-°]').sub(' ', line)
            line = list(filter(None, line.split(' ')))
            for word in line:
                if word in self.word_dict:
                    self.word_dict[word] += 1
                else:
                    self.word_dict[word] = 1



