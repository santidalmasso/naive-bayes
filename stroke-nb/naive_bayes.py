import pandas as pd

class NaiveBayesClassifier:

    def __init__(self, dataframe, classification_var):
        self.df = dataframe
        self.classification_var = classification_var

    def fit(self):
        self.total_classes = self.df[self.classification_var].value_counts()
        self.keys_classes = self.total_classes.keys()
        self.total_rows = sum(self.total_classes)
        df_without_class = self.df.drop(self.classification_var, axis = 1)
        self.attributes_freq = {}
        self.class_and_attr_freq = {}
        
        for column in df_without_class:
            tags = df_without_class[column].value_counts()
            self.attributes_freq[column] = tags
            for _ in self.attributes_freq[column].keys():
                self.class_and_attr_freq[column] = self.df.groupby(column).apply(lambda row : row.groupby(self.classification_var)[column].value_counts())
         
        return None
        
        
    def predict(self, variables):
        # P(C)
        prior_prob = self.total_classes/self.total_rows

        post_prob = pd.Series([1.0 for i in self.keys_classes], index=self.keys_classes) 

        for variable, value in variables.items(): 
            # P(Fn) 
            const_prob = (self.attributes_freq[variable][value] / self.total_rows)

            for key in self.keys_classes:
                # P(Fn|C)
                likelihood = self.class_and_attr_freq[variable][value][key][value] / self.attributes_freq[variable][value]
                post_prob[key] *= ((likelihood / prior_prob[key]) * const_prob)

        print(post_prob)
