import pandas as pd
from naive_bayes import NaiveBayesClassifier

filename = 'healthcare-dataset-stroke-data.csv'
columns_mask = ['gender','age','hypertension','heart_disease','ever_married','Residence_type','smoking_status','stroke']


people2 = pd.DataFrame([
        ["hombre", "casado", "si", "compra"],
        ["hombre", "casado", "no", "compra"],
        ["hombre", "soltero", "no", "compra"],
        ["mujer", "casado", "si", "no compra"],
        ["mujer", "casado", "no", "compra"],
        ["hombre", "casado", "no", "compra"],
        ["hombre", "casado", "no", "no compra"],
        ["mujer", "soltero", "no", "no compra"],
        ["hombre", "soltero", "no", "no compra"],
        ["mujer", "casado", "si", "compra"]
    ], columns = ['sexo', 'estado civil', 'hijos', 'compro']
)


vars_to_predict = {
    'gender': 'Male',
    'age': 'old',
    'hypertension': 1,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'Residence_type': 'Rural',
    'smoking_status': 'smokes'
}

vars_to_predict2 = {
        'sexo': 'hombre',
        'estado civil': 'casado',
        'hijos': 'no',
        }

# young (under 30)
# adult (between 30 and 60)
# old (over 65)

def map_age(df):

    df = df['age'].astype(str)

    return df.apply(lambda age: 'young' if int(float(age)) < 30 else 'old' if int(float(age)) < 65 else 'adult')


if __name__ == '__main__':
    people = pd.read_csv(filename)
    people = people[columns_mask]
    people = people[people['gender'] != 'Other']
    people = people[people['smoking_status'] != 'Unknown']
    people['age'] = map_age(people)

    nv_classifier = NaiveBayesClassifier(people, 'stroke')

    nv_classifier.fit()

    nv_classifier.predict(vars_to_predict)

    
    # nv_classifier = NaiveBayesClassifier(people2, 'compro')
    # nv_classifier.fit()
    # nv_classifier.predict(vars_to_predict2)
    
