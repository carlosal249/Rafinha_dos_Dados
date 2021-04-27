Satisfação de clientes
- Uma empresa de vendas online quer melhorar a satisfação dos clientes
- Para isso, eles usam a nota com que os produtos vendidos são avaliados
- Interpretabilidade: Eles querem insights mais do que o modelo preditivo em si 

Características da base:
- Cada linha é um produto vendido em um dado momento (dia)
- Um mesmo produto pode receber diversas avaliações
- As características do produto não mudam com o tempo
- O preço do produto e o frete podem mudar com o tempo

Histórico de dados suficientes para a análise:
- Entre 2017-01 e 2018-08: 19 meses

Definição da estratégia:
- Dado que a maior parte dos produtos possuem um volume baixo de vendas por mês e dado que o preço e o frete mudam com o tempo
- Vamos definir a avaliação do produto como a média das avaliações recebidas a cada trimestre
- Ao mesmo tempo, conseguimos considerar diferentes patamares de preço, que podem influenciar na satisfação (custo-benefício)
- Como buscamos obter insights, precisamos de um modelo minimamente interpretável, isto é, não estamos interessados na predição do modelo em si, e sim em quais padrões ele encontrou nos dados que explicam a avaliação de um produto.
- Assim, faremos um modelo que responda a seguinte pergunta: Considerando um produto com determinadas características e preço, qual a probabilidade dele ser bem avaliado? 
- E então, pela estrutura do modelo, podemos descobrir quais características mais impactantes na avaliação 

Considerando as nossas datas, temos 6 trimestres
- Mesmo em um modelo que tem somente a intenção de gerar insights, precisamos garantir que a estrutura dele faz sentido. Um modelo que decorou os dados não gera resultados interessantes

Assim, para garantir que o modelo não decorou, é interessante que deixemos os meses mais recentes separados, somente para a validação. Esse modelo de validação é chamado de "validação Out-of-Time"

- Ao deixar os meses de fora, temos a oportunidade de ver como o modelo se sai no "mundo real"
- É importante que esses meses sejam os últimos, pois é sempre mais difícil prever o futuro do que o passado

Para esse exercício, vamos considerar somente os produtos que tem ao menos 10 vendas/avaliaçãos nos meses de treino