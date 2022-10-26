import nltk
from nltk.corpus import stopwords
import pandas as pd
from collections import defaultdict

# Download if it is the first run
# nltk.download('punkt')
# nltk.download('stopwords')

class NaiveBayesReviewsClassifier:
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.total_elements = len(dataframe.index)
        self.stop_words = set(stopwords.words('spanish'))
        self.words_freq = defaultdict(dict)

    def fit(self):
        self.freq_class = self.df['type'].value_counts()
        self._calc_freq_words()
        self.total_words = len(self.words_freq.keys())
        
    def predict(self, review):
        # P(C)
        class_prob = self.freq_class/self.total_elements
        # print(class_prob)
        
        post_prob = class_prob

        # P(Fn|C)
        tokenized_review = self._tokenize_text(review)
        for word in tokenized_review:
            if word in self.words_freq:
                #P(word)
                word_prob = sum(self.words_freq[word].values())/self.total_words
                # print(f" P({word}) = {word_prob}")

                # P(C|word)
                cond_prob = pd.Series(self.words_freq[word])/self.total_words / word_prob
                # print(f" P( C | {word} ) = {cond_prob}")

                # P(word|C)
                likelihood = (cond_prob * word_prob) / class_prob
                # print(f" P({word} | C ) = {likelihood}")
                
                post_prob *= likelihood


        return post_prob.fillna(0)     


    def _calc_freq_words(self):
        self.df.apply(lambda row: self._count_words(row['review'], row['type']), axis = 1)

    def _count_words(self, tokenized_text, review_type):
        frequencies = pd.Series(self._tokenize_text(tokenized_text)).value_counts()
        for word in frequencies.keys():
            self._save_word_freq(word, frequencies[word], review_type)

    def _save_word_freq(self, word, value, review_type):
        actual_value = 1 if word not in self.words_freq or review_type not in self.words_freq[word] else self.words_freq[word][review_type]
        self.words_freq[word][review_type] = actual_value + value

    def _tokenize_text(self, text):
        return  (
                list(filter(lambda word: word not in self.stop_words, 
                list(map(lambda token: token.lower(),
                list(filter(lambda token: token.isalpha(),
                nltk.word_tokenize(text)))))))
            )

