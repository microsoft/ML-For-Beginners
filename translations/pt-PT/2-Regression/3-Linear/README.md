# Construa um modelo de regressão usando Scikit-learn: regressão de quatro formas

## Nota para iniciantes

A regressão linear é usada quando queremos prever um **valor numérico** (por exemplo, preço da casa, temperatura ou vendas).
Funciona encontrando uma linha reta que melhor representa a relação entre as características de entrada e a saída.

Nesta lição, focamos em entender o conceito antes de explorar técnicas de regressão mais avançadas.
![Infográfico de regressão linear vs polinomial](../../../../translated_images/pt-PT/linear-polynomial.5523c7cb6576ccab.webp)
> Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introdução

Até agora explorou o que é regressão com dados de amostra recolhidos do conjunto de dados de preços de abóboras que iremos usar ao longo desta lição. Também o visualizou usando Matplotlib.

Agora está pronto para aprofundar a regressão para ML. Enquanto a visualização lhe permite compreender os dados, o verdadeiro poder do Machine Learning vem do _treino de modelos_. Os modelos são treinados com dados históricos para capturar automaticamente dependências dos dados, e permitem prever resultados para novos dados, que o modelo ainda não viu.

Nesta lição, aprenderá mais sobre dois tipos de regressão: _regressão linear básica_ e _regressão polinomial_, juntamente com alguma da matemática subjacente a estas técnicas. Esses modelos permitir-nos-ão prever os preços das abóboras dependendo de diferentes dados de entrada.

[![ML para iniciantes - Compreendendo a Regressão Linear](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML para iniciantes - Compreendendo a Regressão Linear")

> 🎥 Clique na imagem acima para um breve vídeo de apresentação da regressão linear.

> Ao longo deste currículo, assumimos conhecimentos mínimos de matemática, e procuramos torná-la acessível para estudantes provenientes de outras áreas, por isso repare nas notas, 🧮 chamadas, diagramas e outras ferramentas de aprendizagem para ajudar a compreensão.

### Pré-requisitos

Deverá estar familiarizado agora com a estrutura dos dados da abóbora que estamos a examinar. Pode encontrá-los pré-carregados e pré-limpados no ficheiro _notebook.ipynb_ desta lição. No ficheiro, o preço da abóbora é mostrado por alqueire num novo dataframe. Assegure-se de que consegue executar estes cadernos no kernel do Visual Studio Code.

### Preparação

Como lembrete, está a carregar estes dados para lhes fazer perguntas.

- Qual é o melhor momento para comprar abóboras?
- Que preço posso esperar por um pacote de mini abóboras?
- Devo comprá-las em cestos de meio alqueire ou em caixas de 1 1/9 alqueire?
Vamos continuar a explorar estes dados.

Na lição anterior, criou um dataframe Pandas e povoou-o com parte do conjunto de dados original, padronizando o preço por alqueire. Ao fazer isso, no entanto, conseguiu apenas cerca de 400 pontos de dados e apenas para os meses de outono.

Veja os dados que pré-carregámos no caderno que acompanha esta lição. Os dados estão pré-carregados e foi traçado um gráfico de dispersão inicial para mostrar os dados do mês. Talvez possamos obter um pouco mais de detalhe sobre a natureza dos dados limpando-os mais.

## Uma linha de regressão linear

Como aprendeu na Lição 1, o objetivo de um exercício de regressão linear é poder traçar uma linha para:

- **Mostrar relações entre variáveis**. Mostrar a relação entre variáveis
- **Fazer previsões**. Fazer previsões precisas sobre onde um novo ponto de dados cairia em relação a essa linha.

É típico da **Regressão de Mínimos Quadrados** desenhar este tipo de linha. O termo "Mínimos Quadrados" refere-se ao processo de minimizar o erro total no nosso modelo. Para cada ponto de dados, medimos a distância vertical (chamada resíduo) entre o ponto real e a nossa linha de regressão.

Elevamos essas distâncias ao quadrado por duas razões principais:

1. **Magnitude em vez de Direção:** Queremos tratar um erro de -5 da mesma forma que um erro de +5. Elevar ao quadrado torna todos os valores positivos.

2. **Penalizar Outliers:** Elevar ao quadrado dá mais peso aos erros maiores, forçando a linha a manter-se mais perto dos pontos que estão longe.

Depois somamos todos esses valores ao quadrado. O nosso objetivo é encontrar a linha específica onde esta soma final é a menor (o menor valor possível)—daí o nome "Mínimos Quadrados".

> **🧮 Mostre-me a matemática** 
> 
> Esta linha, chamada de _linha de melhor ajuste_, pode ser expressa por [uma equação](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` é a 'variável explicativa'. `Y` é a 'variável dependente'. A inclinação da linha é `b` e `a` é o intercepto-y, que refere o valor de `Y` quando `X = 0`.
>
>![calcular a inclinação](../../../../translated_images/pt-PT/slope.f3c9d5910ddbfcf9.webp)
>
> Primeiro, calcule a inclinação `b`. Infográfico por [Jen Looper](https://twitter.com/jenlooper)
>
> Por outras palavras, e referindo à questão original dos nossos dados da abóbora: "prever o preço de uma abóbora por alqueire por mês", `X` referir-se-ia ao preço e `Y` ao mês da venda.
>
>![completar a equação](../../../../translated_images/pt-PT/calculation.a209813050a1ddb1.webp)
>
> Calcule o valor de Y. Se está a pagar cerca de 4$, deve ser abril! Infográfico por [Jen Looper](https://twitter.com/jenlooper)
>
> A matemática que calcula a linha deve demonstrar a inclinação da linha, que também depende do intercepto, ou onde `Y` está situado quando `X = 0`.
>
> Pode observar o método de cálculo destes valores no site [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Visite também [este Calculador de mínimos quadrados](https://www.mathsisfun.com/data/least-squares-calculator.html) para ver como os valores dos números impactam a linha.

## Correlação

Mais um termo a compreender é o **Coeficiente de Correlação** entre as variáveis X e Y dadas. Usando um gráfico de dispersão, pode rapidamente visualizar este coeficiente. Um gráfico com pontos de dados dispersos numa linha arrumada tem alta correlação, mas um gráfico com pontos dispersos por toda a parte entre X e Y tem baixa correlação.

Um bom modelo de regressão linear será aquele que tem um alto (mais próximo de 1 do que de 0) Coeficiente de Correlação usando o método de Regressão de Mínimos Quadrados com uma linha de regressão.

✅ Execute o caderno que acompanha esta lição e observe o gráfico de dispersão Mês vs Preço. Os dados que associam Mês a Preço para vendas de abóboras parecem ter alta ou baixa correlação, de acordo com a sua interpretação visual do gráfico de dispersão? Isso muda se usar uma medida mais detalhada em vez de `Mês`, por exemplo, *dia do ano* (isto é, número de dias desde o início do ano)?

No código abaixo, assumiremos que limpámos os dados e obtivemos um dataframe chamado `new_pumpkins`, semelhante ao seguinte:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|--------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> O código para limpar os dados está disponível em [`notebook.ipynb`](notebook.ipynb). Realizámos os mesmos passos de limpeza que na lição anterior, e calculámos a coluna `DayOfYear` usando a seguinte expressão:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Agora que entende a matemática por trás da regressão linear, vamos criar um modelo de regressão para ver se conseguimos prever qual o pacote de abóboras que terá os melhores preços. Alguém a comprar abóboras para um campo de abóboras de férias poderá querer esta informação para otimizar as suas compras de pacotes para o campo.

## Procurando Correlação

[![ML para iniciantes - Procurando Correlação: A Chave para Regressão Linear](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML para iniciantes - Procurando Correlação: A Chave para Regressão Linear")

> 🎥 Clique na imagem acima para um breve vídeo de apresentação sobre correlação.

Na lição anterior provavelmente viu que o preço médio para diferentes meses é assim:

<img alt="Preço médio por mês" src="../../../../translated_images/pt-PT/barchart.a833ea9194346d76.webp" width="50%"/>

Isto sugere que deve haver alguma correlação, e podemos tentar treinar um modelo de regressão linear para prever a relação entre `Month` e `Price`, ou entre `DayOfYear` e `Price`. Aqui está o gráfico de dispersão que mostra a última relação:

<img alt="Gráfico de dispersão do Preço vs. Dia do Ano" src="../../../../translated_images/pt-PT/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Vamos ver se há correlação usando a função `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Parece que a correlação é bastante pequena, -0.15 por `Month` e -0.17 por `DayOfYear`, mas pode haver outra relação importante. Parece que existem diferentes grupos de preços correspondentes a diferentes variedades de abóboras. Para confirmar esta hipótese, vamos traçar cada categoria de abóbora usando uma cor diferente. Passando um parâmetro `ax` para a função de plotagem `scatter` podemos traçar todos os pontos no mesmo gráfico:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Gráfico de dispersão do Preço vs. Dia do Ano" src="../../../../translated_images/pt-PT/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

A nossa investigação sugere que a variedade tem mais efeito no preço global do que a data de venda. Podemos ver isto com um gráfico de barras:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Gráfico de barras do preço vs variedade" src="../../../../translated_images/pt-PT/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Vamos focar por ora numa só variedade de abóbora, o tipo 'pie type', e ver que efeito a data tem no preço:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Gráfico de dispersão do Preço vs. Dia do Ano" src="../../../../translated_images/pt-PT/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Se agora calcularmos a correlação entre `Price` e `DayOfYear` usando a função `corr`, obteremos algo como `-0.27` - o que significa que faz sentido treinar um modelo preditivo.

> Antes de treinar um modelo de regressão linear, é importante garantir que os dados estão limpos. A regressão linear não funciona bem com valores em falta, por isso faz sentido eliminar todas as células vazias:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Outra abordagem seria preencher esses valores vazios com valores médios da coluna correspondente.

## Regressão Linear Simples

[![ML para iniciantes - Regressão Linear e Polinomial usando Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML para iniciantes - Regressão Linear e Polinomial usando Scikit-learn")

> 🎥 Clique na imagem acima para um breve vídeo de apresentação sobre regressão linear e polinomial.

Para treinar o nosso modelo de Regressão Linear, iremos usar a biblioteca **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Começamos por separar os valores de entrada (características) e a saída esperada (rótulo) em arrays numpy separados:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Note que tivemos de realizar um `reshape` nos dados de entrada para que o pacote de Regressão Linear os entenda corretamente. Regressão Linear espera um array 2D como entrada, em que cada linha do array corresponde a um vetor de características de entrada. No nosso caso, como temos apenas uma entrada - precisamos de um array com formato N&times;1, onde N é o tamanho do conjunto de dados.

Depois, precisamos dividir os dados em conjuntos de treino e teste, para podermos validar o nosso modelo após o treino:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Finalmente, treinar o modelo de Regressão Linear propriamente dito leva apenas duas linhas de código. Definimos o objeto `LinearRegression` e ajustamo-lo aos nossos dados usando o método `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

O objeto `LinearRegression` depois de ajustado (`fit`) contém todos os coeficientes da regressão, que podem ser acedidos usando a propriedade `.coef_`. No nosso caso, há apenas um coeficiente, que deverá estar por volta de `-0.017`. Isso significa que os preços parecem baixar um pouco com o tempo, mas não muito, cerca de 2 cêntimos por dia. Podemos também aceder ao ponto de interseção da regressão com o eixo Y usando `lin_reg.intercept_` - será cerca de `21` no nosso caso, indicando o preço no início do ano.

Para ver quão preciso é o nosso modelo, podemos prever preços num conjunto de dados de teste, e depois medir quão próximas as nossas previsões estão dos valores esperados. Isto pode ser feito usando a métrica de erro quadrático médio da raiz (RMSE), que é a raiz da média de todas as diferenças ao quadrado entre o valor esperado e o previsto.

```python
pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')
```

O nosso erro parece ser cerca de 2 pontos, o que é ~17%. Não muito bom. Outro indicador da qualidade do modelo é o **coeficiente de determinação**, que pode ser obtido assim:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Se o valor for 0, significa que o modelo não tem em conta os dados de entrada, e funciona como o *pior preditor linear*, que é simplesmente um valor médio do resultado. O valor 1 significa que podemos prever perfeitamente todas as saídas esperadas. No nosso caso, o coeficiente está por volta de 0.06, o que é bastante baixo.

Podemos também traçar os dados de teste juntamente com a linha de regressão para vermos melhor como a regressão funciona no nosso caso:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/pt-PT/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regressão Polinomial

Outro tipo de Regressão Linear é a Regressão Polinomial. Embora por vezes haja uma relação linear entre variáveis - quanto maior a abóbora em volume, maior o preço - por vezes essas relações não podem ser representadas como um plano ou linha reta. 

✅ Aqui estão [mais alguns exemplos](https://online.stat.psu.edu/stat501/Lesson09#polynomial-regression-examples) de dados que poderiam usar Regressão Polinomial

Dê outra vista de olhos à relação entre Data e Preço. Este gráfico de dispersão parece necessariamente deva ser analisado por uma linha reta? Não podem os preços oscilar? Neste caso, pode tentar regressão polinomial.

✅ Polinómios são expressões matemáticas que podem consistir de uma ou mais variáveis e coeficientes

A regressão polinomial cria uma linha curva para melhor ajustar os dados não lineares. No nosso caso, se incluirmos uma variável ao quadrado `DayOfYear` nos dados de entrada, deveremos conseguir ajustar os nossos dados com uma curva parabólica, que terá um mínimo em certo ponto durante o ano.

O Scikit-learn inclui uma útil [API para pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) para combinar diferentes passos do processamento de dados. Um **pipeline** é uma cadeia de **estimadores**. No nosso caso, vamos criar um pipeline que primeiro adiciona características polinomiais ao nosso modelo, e depois treina a regressão:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Usar `PolynomialFeatures(2)` significa que iremos incluir todos os polinómios de segundo grau dos dados de entrada. No nosso caso, isso significa apenas `DayOfYear`<sup>2</sup>, mas dado duas variáveis de entrada X e Y, isto adicionaria X<sup>2</sup>, XY e Y<sup>2</sup>. Podemos também usar polinómios de grau superior se quisermos.

Pipelines podem ser usados da mesma forma que o objeto original `LinearRegression`, ou seja, podemos `fit` o pipeline, e depois usar `predict` para obter os resultados da previsão:

```python
pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Para traçar a curva de aproximação suave, usamos `np.linspace` para criar um intervalo uniforme de valores de entrada, em vez de traçar diretamente sobre os dados de teste desordenados (o que produziria uma linha em zig-zag):

```python
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
y_range = pipeline.predict(X_range)

plt.scatter(X_test, y_test)
plt.plot(X_range, y_range)
```

Aqui está o gráfico que mostra os dados de teste e a curva de aproximação:

<img alt="Polynomial regression" src="../../../../translated_images/pt-PT/poly-results.ee587348f0f1f60b.webp" width="50%" />

Usando Regressão Polinomial, podemos obter um RMSE ligeiramente mais baixo e um coeficiente de determinação mais alto, mas não significativamente. Precisamos de ter em conta outras características!

> Pode ver que os preços mínimos das abóboras são observados por volta do Halloween. Como pode explicar isto? 

🎃 Parabéns, acaba de criar um modelo que pode ajudar a prever o preço das abóboras para torta. Provavelmente pode repetir o mesmo procedimento para todos os tipos de abóbora, mas isso seria trabalhoso. Vamos agora aprender a ter em conta a variedade da abóbora no nosso modelo!

## Características Categóricas

No mundo ideal, queremos ser capazes de prever preços para diferentes variedades de abóboras usando o mesmo modelo. Contudo, a coluna `Variety` é algo diferente de colunas como `Month`, porque contém valores não numéricos. Essas colunas são chamadas **categóricas**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Clique na imagem acima para ver um breve vídeo sobre o uso de características categóricas.

Aqui pode ver como o preço médio depende da variedade:

<img alt="Average price by variety" src="../../../../translated_images/pt-PT/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Para ter em conta a variedade, primeiro precisamos de a converter em forma numérica, ou **codificar**. Existem várias formas de o fazer:

* Uma simples **codificação numérica** construirá uma tabela das diferentes variedades, e depois substituirá o nome da variedade por um índice nessa tabela. Esta não é a melhor ideia para regressão linear, porque a regressão linear utiliza o valor numérico real do índice, adicionando-o ao resultado, multiplicando por algum coeficiente. No nosso caso, a relação entre o número do índice e o preço é claramente não linear, mesmo que asseguremos que os índices estão ordenados de alguma forma específica.
* A **codificação one-hot** substituirá a coluna `Variety` por 4 colunas diferentes, uma para cada variedade. Cada coluna terá o valor `1` se a linha correspondente for daquela variedade, e `0` caso contrário. Isto significa que haverá quatro coeficientes na regressão linear, um para cada variedade de abóbora, responsável pelo "preço inicial" (ou antes "preço adicional") para aquela variedade em particular.

O código abaixo mostra como podemos fazer a codificação one-hot de uma variedade:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

Para treinar regressão linear usando variedade codificada em one-hot como entrada, só precisamos inicializar corretamente os dados `X` e `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

O resto do código é igual ao que usamos acima para treinar a Regressão Linear. Se tentar, verá que o erro quadrático médio é cerca do mesmo, mas obtemos um coeficiente de determinação muito mais elevado (~77%). Para obter previsões ainda mais precisas, podemos ter em conta mais características categóricas, bem como características numéricas, como `Month` ou `DayOfYear`. Para obter um único grande array de características, podemos usar `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Aqui também temos em conta `City` e `Package` (embalagem), o que nos dá RMSE 2.84 (10.5%), e determinação 0.94!

## Juntando tudo

Para fazer o melhor modelo, podemos usar dados combinados (características categóricas codificadas one-hot + numéricas) do exemplo acima juntamente com regressão polinomial. Aqui está o código completo para sua conveniência:

```python
# preparar dados de treino
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# fazer divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# configurar e treinar o pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# prever resultados para dados de teste
pred = pipeline.predict(X_test)

# calcular RMSE e determinação
rmse = mean_squared_error(y_test, pred, squared=False)
print(f'RMSE: {rmse:3.3} ({rmse/pred.mean()*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Isto deverá dar-nos o melhor coeficiente de determinação de quase 97%, e RMSE=2.23 (~8% de erro de previsão).

| Modelo | RMSE | Determinação |
|-------|-----|---------------|
| Linear `DayOfYear` | 2.77 (17.2%) | 0.07 |
| Polinomial `DayOfYear` | 2.73 (17.0%) | 0.08 |
| Linear `Variety` | 5.24 (19.7%) | 0.77 |
| Todas as características Linear | 2.84 (10.5%) | 0.94 |
| Todas as características Polinomial | 2.23 (8.25%) | 0.97 |

🏆 Muito bem! Criou quatro modelos de Regressão numa lição, e melhorou a qualidade do modelo para 97%. Na secção final sobre Regressão, aprenderá sobre Regressão Logística para determinar categorias. 

---
## 🚀Desafio

Teste várias variáveis diferentes neste notebook para ver como a correlação corresponde à precisão do modelo.

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Nesta lição aprendemos sobre Regressão Linear. Existem outros tipos importantes de Regressão. Leia sobre as técnicas Stepwise, Ridge, Lasso e Elasticnet. Um bom curso para estudar e aprender mais é o [Curso de Aprendizagem Estatística de Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Trabalho 

[Construa um Modelo](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, tenha em atenção que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->