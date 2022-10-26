# Inteligencia artificial

Estuvimos haciendo varias pruebas.
La primera fue un dataset de [reviews](reviews.csv), que está compuesto por el mensaje del comentario y un número que indica si la opinión fue positiva o negativa (positiva: 1, negativa: -1)

### Ejecución

Usar el administrador de paquetes [pip](https://pip.pypa.io/en/stable/) para installar las dependencias (o conda).

```bash
python3 main.py
```

#

Después también probamos el algoritmo de la librería [scikit learn](https://scikit-learn.org/stable/modules/naive_bayes.html). Pero para este caso usamos otro dataset que tiene todas variables cualitativas. El dataset muestra la probabilidad de tener un [ataque cardíaco](https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility)


Por último, probamos también usar otro dataset, el cual contiene datos de pacientes que tuvieron algún [ataque cerebrovascular](https://www.kaggle.com/asaumya/healthcare-problem-prediction-stroke-patients). Pero no tuvimos resultados debido a que tuvimos problemas a la hora de implementar el algoritmo manualmente y falta de tiempo. Este caso se encuentra en la carpeta `/stroke-nb`