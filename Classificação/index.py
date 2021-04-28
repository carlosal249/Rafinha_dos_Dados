import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

#df significa data frame
df = pd.read_csv('MLBE_base_aula_1_classificacao.csv')

#checar se o montante de dados é suficiente para a análise
#print(df.anomes.value_counts().sort_index()) 

trimestres = [
     [ 1, 2, 3],
     [ 4, 5, 6],
     [ 7, 8, 9],
     [10,11,12],
]

#checar o tipo de dado em cada coluna do arquivo csv
#print(df.dtypes)

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

#salvar gráfico referente a média dos review_score
fname = './test.pdf'
plt.figure(figsize=(15,5))
plt.hist(df2_tr.review_score_mean, bins=15, rwidth=.8)
plt.savefig(fname)

#filtrar os produtos excelentes dos medianos em uma nova tabela
df3_tr = df2_tr.copy()
df3_tr.loc[:,'target'] = (df3_tr.review_score_mean >= 4.5).astype(int)
df3_tr = df3_tr.drop('review_score_mean', axis=1) #axis=1 refere-se a coluna

df3_vl = df2_vl.copy()
df3_vl.loc[:,'target'] = (df3_vl.review_score_mean >= 4.5).astype(int)
df3_vl = df3_vl.drop('review_score_mean', axis=1)

#X significa tabela de variáveis preditivas ou independentes
X = df3_tr.drop('target', axis=1)

#Y significa tabela de variáveis resposta ou dependentes
y = df3_tr.target

X_vl = df3_vl.drop('target', axis=1)
y_vl = df3_vl.target

#Verificar se a tabela de variáveis resposta está balanceada
#y.sum()/y.shape[0] ou y.mean(), y_v1.mean()

#Separação em treino e teste out of sample com 30% das amostras escolhidas aleatoriamente
X_tr, X_ts, y_tr, y_ts = train_test_split(X,y,test_size=0.30, random_state=61658)

#Fazer encoding de variáveis categóricas por Label encoding
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
enc.fit(X_tr[['product_category_name_mode']])

X2_tr = X_tr.copy()
X2_ts = X_ts.copy()
X2_vl = X_vl.copy()

X2_tr.loc[:,'product_category_name_mode'] = enc.transform(X_tr[['product_category_name_mode']])
X2_ts.loc[:,'product_category_name_mode'] = enc.transform(X_ts[['product_category_name_mode']])
X2_vl.loc[:,'product_category_name_mode'] = enc.transform(X_vl[['product_category_name_mode']])

#Tirar as categorias que apareceram pouco na nossa tabela de treino
#print(X2_tr.product_category_name_mode.value_counts())

transformar_categorias = [
    0.0,
    1.0,
    53.0,
    37.0,
    8.0,
    18.0,
    2.0,
    51.0,
    36.0,
    12.0,
    46.0,
    41.0,
    28.0,
    21.0,
    50.0,
    13.0,
    34.0,
    29.0,
    15.0,
    4.0,
    14.0,
    32.0,
    20.0,
    30.0,
    16.0,
    19.0,
    33.0,
    40.0,
    3.0,
]

for cat_num in transformar_categorias:
    X2_tr.loc[:,'product_category_name_mode'] = X2_tr.product_category_name_mode.replace(cat_num, -1)
    X2_ts.loc[:,'product_category_name_mode'] = X2_ts.product_category_name_mode.replace(cat_num, -1)
    X2_vl.loc[:,'product_category_name_mode'] = X2_vl.product_category_name_mode.replace(cat_num, -1)

#print(X2_tr.product_category_name_mode.value_counts())
