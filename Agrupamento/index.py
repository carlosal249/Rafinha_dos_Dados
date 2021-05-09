import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

import matplotlib 

font = {'size':14}
matplotlib.rc('font', **font)

df = pd.read_csv('base_aula_2_agrupamento.csv')

#Verificar a quantidade de pessoas que compraram mais de uma vez no Ecommerce
print((df.customer_id.value_counts()>1).mean()) #Cerca de 90% dos clientes só fizeram uma única compra

#Normaliza vai retornar a proporção das categorias, iloc vai printar as 12 primeiras linhas

for n_cats in range(1,30):
    print(n_cats, df.product_category_name.value_counts(normalize=True).iloc[:n_cats].sum()
)

#Salvar as categorias mais frequentes
mais_frequentes = df.product_category_name.value_counts(normalize=True).iloc[:15].index

#Agrupar as categorias menos frequentes como outros
def only_freq(row):
    if row.product_category_name not in mais_frequentes:
        return 'outros'
    else:
        return row.product_category_name

df2 = df.copy()
df2.loc[:, 'product_category_name'] = df.apply(only_freq, axis=1)

df_piv = df2.pivot_table(
    index = 'city',
    columns = 'product_category_name',
    values = 'price',
    aggfunc = 'sum',
)

#Preencher valor faltante para regiões que não consumiram determinado produto
df_piv = df_piv.fillna(0)

#Tirar a coluna de outros
df_piv_normed = df_piv.drop('outros',axis=1)

#Normalizar os dados -> Usando distancia cosseno
df_piv_normed = df_piv_normed.div( (df_piv_normed**2).sum(axis=1)**0.5, axis=0 )

def plot_dendrogram(model, **kwargs):
# Create linkage matrix and then plot the dendrogram
# create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for i, merge in enumerate(model.children_):
        current_count = 0

        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 # leaf node

            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

fname = './Aglomeracao.pdf'

X = df_piv_normed.copy()
model = model.fit(X)
plt.figure(figsize=(15,5))
plot_dendrogram(model, truncate_mode='level', p=20)

plt.savefig(fname)