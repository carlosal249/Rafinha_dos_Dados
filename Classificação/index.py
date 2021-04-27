import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import subprocess
import shlex

#df significa data frame
df = pd.read_csv('MLBE_base_aula_1_classificacao.csv')

#checar se o montante de dados é suficiente para a análise
print(df.anomes.value_counts().sort_index()) 

trimestres = [
     [ 1, 2, 3],
     [ 4, 5, 6],
     [ 7, 8, 9],
     [10,11,12],
]

#checar o tipo de dado em cada coluna do arquivo csv
print(df.dtypes)

#converter a coluna anomes (tipo objeto) para tipo data
df.loc[:, 'anomes'] = pd.to_datetime(df.anomes)

def gera_trimestre(row):
    if row.anomes.month in trimestres[0]:
        return 1
    elif row.anomes.month in trimestres[1]:
        return 2
    elif row.anomes.month in trimestres[2]:
        return 3
    else:
        return 4

#criar uma nova coluna no arquivo csv referente ao trimestre usando apply
df.loc[:,'trimestre'] = df.apply(gera_trimestre, axis=1)
df.loc[:,'ano'] = df.anomes.dt.year
print(df.head())

#arquivo de teste
df_tr = df.loc[(df.anomes >= '2017-01-01') & (df.anomes < '2018-04-01')].copy()

#arquivo de validação
df_vl = df.loc[(df.anomes >= '2018-04-01') & (df.anomes < '2019-09-01')].copy()

#filtrar produtos com mais de 10 vendas/avaliações
aux = df_tr.product_id.value_counts()
produtos = aux.loc[aux>=10].index

df2_tr = df_tr.loc[df_tr.product_id.isin(produtos)].groupby(['product_id', 'trimestre', 'ano']).agg({
    'review_score': 'mean',
    'price': ['min', 'mean', 'max'],
    'freight_value': ['min', 'mean', 'max'],
    'product_category_name': pd.Series.mode, #média == moda
    'product_name_lenght': 'mean',
    'product_description_lenght': 'mean',
    'product_photos_qty': 'mean',
    'product_weight_g': 'mean',
    'product_length_cm': 'mean',
    'product_height_cm': 'mean',
    'product_width_cm': 'mean',
})

df2_vl = df_vl.loc[df_vl.product_id.isin(produtos)].groupby(['product_id', 'trimestre', 'ano']).agg({
    'review_score': 'mean',
    'price': ['min', 'mean', 'max'],
    'freight_value': ['min', 'mean', 'max'],
    'product_category_name': pd.Series.mode, #média == moda
    'product_name_lenght': 'mean',
    'product_description_lenght': 'mean',
    'product_photos_qty': 'mean',
    'product_weight_g': 'mean',
    'product_length_cm': 'mean',
    'product_height_cm': 'mean',
    'product_width_cm': 'mean',
})

#simplificar acesso às colunas
df2_tr.columns = ["_".join(x) for x in df2_tr.columns.values.ravel()]
df2_vl.columns = ["_".join(x) for x in df2_vl.columns.values.ravel()]
print(df2_tr.columns)

#salvar gráfico referente a média dos review_score
fname = './test.pdf'
plt.figure(figsize=(15,5))
plt.hist(df2_tr.review_score_mean, bins=15, rwidth=.8)
plt.savefig(fname)
