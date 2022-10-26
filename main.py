import pandas as pd
from nb_reviews import NaiveBayesReviewsClassifier

filename = 'reviews.csv'


if __name__ == '__main__':
    reviews = pd.read_csv(filename)

    nb_reviews_classifier = NaiveBayesReviewsClassifier(reviews)
    nb_reviews_classifier.fit()

    review = input('\nIntruduce el comentario a clasificar: ')
    result = nb_reviews_classifier.predict(review)
    
    #print(result)

    if result[1] > result[-1]:
        print('\n\n\033[92mComentario positivo\n')
    else:
	    print('\n\n\033[91mComentario negativo\n')
