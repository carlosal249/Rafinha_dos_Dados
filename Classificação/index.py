import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

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

#Criar modelo RandomForest
params = {
    'max_depth': [4, 6, 8, 10],
    'class_weight': [None, 'balanced'],
    'criterion': ['gini', 'entropy'],
}

grid = GridSearchCV(
    RandomForestClassifier(n_estimators=500, random_state=61658, n_jobs=3),
    params,
    cv=10,
    scoring='roc_auc',
    verbose=10,
    n_jobs=1,
)

grid.fit(X2_tr, y_tr)

grid.best_params_ #verifica o melhor parâmetro 
grid.best_score_ #verifica o melhor score

#Verificar se o melhor modelo deu uma resposta melhor que os outros 
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/- %0.03f) for %r" %(mean, std * 2, params))

#Verificar o modelo com o teste
preds = grid.predict_proba(X2_ts)[:,1]

#AUC
roc_auc_score(y_ts, preds) #AUC SCORE no teste Out-of-Sample
roc_auc_score(y_vl,  grid.predict_proba(X2_vl)[:,1]) #AUC SCORE na validação Out-of-Time

#Checar a quantidade de árvores da randomForest
aucs = []
n_trees = [10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
for n_estimators in n_trees:
    print(n_estimators)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=61658, n_jobs=3, **grid.best_params_)

    rf.fit(X2_tr, y_tr)
    preds = rf.predict_proba(X2_vl)[:,1]
    auc = roc_auc_score(y_vl, preds)

    aucs.append(auc)

fname2 = './arvores.pdf'
plt.figure(figsize=(15,5))
plt.plot(n_trees, aucs, '.-')
plt.grid()

plt.savefig(fname2)

#Salvar a predição do modelo na planilha de teste
X2_ts.loc[:,'pred'] = rf.predict_proba(X2_ts)[:,1]

#Verificar como o modelo está funcionando nas diferentes safras de público
scores = []
ano_tri = []
for ano in np.sort(X2_ts.reset_index().ano.unique()):
    for tri in np.sort(X2_ts.reset_index().trimestre.unique()):
        print(ano,tri)
        y_ts_loc = y_ts.reset_index()
        y_ts_loc = y_ts_loc.loc[(y_ts_loc.ano==ano) & (y_ts_loc.trimestre==tri)]

        X2_ts_loc = X2_ts.reset_index()
        X2_ts_loc = X2_ts_loc.loc[(X2_ts_loc.ano==ano) & (X2_ts_loc.trimestre==tri)]

        if X2_ts_loc.shape[0]==0:
            continue

        auc = roc_auc_score(y_ts_loc.target, X2_ts_loc.pred)

        scores.append(auc)
        ano_tri.append(f'{ano}-{tri}')

fname3 = './safras.pdf'
plt.figure(figsize=(15,5))
plt.plot(scores)
plt.xticks(ticks=range(len(scores)), labels=ano_tri, fontsize=14)
plt.grid()

plt.savefig(fname3)

#Importacia de variáveis
rf.feature_importances_ #retorna o peso de cada coluna da tabela na classificação do produto

imps = rf.feature_importances_
cols = X2_tr.columns
order = np.argsort(imps)[::-1]

for col, imp in zip(cols[order], imps[order]):
    print(f'{col:50s}{imp:.3f}{"*"*int(100*imp)}')

X_interpretacao = X2_ts[['freight_value_min','pred']].copy()
X_interpretacao.loc[:,'freight_bin'] = pd.qcut(X_interpretacao.freight_value_min, 10, duplicates='drop')
print(X_interpretacao)

print(X_interpretacao.freight_bin.value_counts(normalize=True))

plt.figure(figsize=(15,5))
ax = plt.subplot(1,1,1)
X_interpretacao.groupby('freight_bin').pred.mean().plot(rot=45, ax=ax, lw=5.)
X_interpretacao.groupby('freight_bin').pred.quantile(q=0.10).plot(rot=45, ax=ax)
X_interpretacao.groupby('freight_bin').pred.quantile(q=0.90).plot(rot=45, ax=ax)

plt.grid()
plt.xticks(fontsize=14)

