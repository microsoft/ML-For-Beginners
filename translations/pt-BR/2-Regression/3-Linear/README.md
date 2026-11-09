# Construindo um modelo de regressão usando Scikit-learn: regressão de quatro formas

## Nota para iniciantes

A regressão linear é usada quando queremos prever um **valor numérico** (por exemplo, preço de casa, temperatura ou vendas).  
Funciona encontrando uma linha reta que melhor representa a relação entre as características de entrada e o resultado.

Nesta lição, focamos em entender o conceito antes de explorar técnicas de regressão mais avançadas.  
![Infográfico de regressão linear vs polinomial](../../../../translated_images/pt-BR/linear-polynomial.5523c7cb6576ccab.webp)  
> Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)  
## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)  
### Introdução

Até agora, você explorou o que é regressão com dados de amostra coletados do conjunto de dados de preços de abóbora que usaremos durante esta lição. Você também o visualizou usando Matplotlib.

Agora você está pronto para mergulhar mais fundo na regressão para ML. Enquanto a visualização permite que você compreenda os dados, o verdadeiro poder do Machine Learning vem do _treinamento de modelos_. Modelos são treinados com dados históricos para capturar automaticamente dependências dos dados e permitem que você preveja resultados para novos dados, que o modelo não viu antes.

Nesta lição, você aprenderá mais sobre dois tipos de regressão: _regressão linear básica_ e _regressão polinomial_, juntamente com parte da matemática por trás dessas técnicas. Esses modelos nos permitirão prever preços de abóboras dependendo de diferentes dados de entrada.

[![ML para iniciantes - Entendendo Regressão Linear](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML para iniciantes - Entendendo Regressão Linear")

> 🎥 Clique na imagem acima para um vídeo curto sobre regressão linear.

> Ao longo deste currículo, assumimos conhecimento mínimo em matemática, buscando torná-lo acessível para estudantes de outras áreas, então fique atento a notas, 🧮 chamadas, diagramas e outras ferramentas de aprendizado para ajudar na compreensão.

### Pré-requisito

Você deve estar familiarizado agora com a estrutura dos dados da abóbora que estamos examinando. Você pode encontrá-los pré-carregados e pré-limpos no arquivo _notebook.ipynb_ desta lição. No arquivo, o preço da abóbora é exibido por alqueire em um novo data frame. Certifique-se de que pode executar esses notebooks em kernels no Visual Studio Code.

### Preparação

Como lembrete, você está carregando esses dados para poder fazer perguntas sobre eles.

- Quando é o melhor momento para comprar abóboras?  
- Qual preço posso esperar de uma caixa de abóboras miniatura?  
- Devo comprá-las em cestos de meio alqueire ou em caixas de 1 1/9 alqueire?  
Vamos continuar investigando esses dados.

Na lição anterior, você criou um data frame do Pandas e o populou com parte do conjunto de dados original, padronizando o preço por alqueire. Ao fazer isso, porém, você conseguiu apenas cerca de 400 pontos de dados e apenas para os meses de outono.

Dê uma olhada nos dados que pré-carregamos no notebook que acompanha esta lição. Os dados estão pré-carregados e foi plotado um gráfico de dispersão inicial para mostrar os dados de mês. Talvez possamos obter um pouco mais de detalhe sobre a natureza dos dados limpando-os mais.

## Uma linha de regressão linear

Como você aprendeu na Lição 1, o objetivo de um exercício de regressão linear é poder traçar uma linha para:

- **Mostrar relações entre variáveis**. Mostrar a relação entre as variáveis  
- **Fazer previsões**. Fazer previsões precisas de onde um novo ponto de dados cairia em relação a essa linha.

É típico em **Regressão de Mínimos Quadrados** desenhar esse tipo de linha. O termo "Mínimos Quadrados" se refere ao processo de minimizar o erro total em nosso modelo. Para cada ponto de dados, medimos a distância vertical (chamada resíduo) entre o ponto real e nossa linha de regressão.

Elevamos essas distâncias ao quadrado por dois motivos principais:

1. **Magnitude sobre Direção:** Queremos tratar um erro de -5 igual a um erro de +5. Ao elevar ao quadrado, todos os valores ficam positivos.

2. **Penalizando Outliers:** Elevar ao quadrado dá mais peso aos erros maiores, forçando a linha a ficar mais próxima dos pontos que estão distantes.

Então somamos todos esses valores ao quadrado. Nosso objetivo é encontrar a linha específica onde essa soma final seja a menor possível — daí o nome "Mínimos Quadrados".

> **🧮 Mostre-me a matemática**  
>  
> Essa linha, chamada _linha de melhor ajuste_, pode ser expressa por [uma equação](https://en.wikipedia.org/wiki/Simple_linear_regression):  
>  
> ```
> Y = a + bX
> ```
>  
> `X` é a 'variável explicativa'. `Y` é a 'variável dependente'. A inclinação da linha é `b` e `a` é o intercepto y, que se refere ao valor de `Y` quando `X = 0`.  
>  
>![calcular a inclinação](../../../../translated_images/pt-BR/slope.f3c9d5910ddbfcf9.webp)  
>  
> Primeiro, calcule a inclinação `b`. Infográfico por [Jen Looper](https://twitter.com/jenlooper)  
>  
> Em outras palavras, e referindo à pergunta original dos nossos dados de abóbora: "prever o preço de uma abóbora por alqueire segundo o mês", `X` se referiria ao mês e `Y` se referiria ao preço de venda.  
>  
>![completar a equação](../../../../translated_images/pt-BR/calculation.a209813050a1ddb1.webp)  
>  
> Calcule o valor de Y. Se você está pagando cerca de $4, deve ser abril! Infográfico por [Jen Looper](https://twitter.com/jenlooper)  
>  
> A matemática que calcula a linha deve demonstrar a inclinação da linha, que também depende do intercepto, ou onde `Y` se situa quando `X = 0`.  
>  
> Você pode observar o método de cálculo desses valores no site [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Também visite [calculadora de mínimos quadrados](https://www.mathsisfun.com/data/least-squares-calculator.html) para ver como os valores dos números impactam a linha.

## Correlação

Mais um termo para entender é o **Coeficiente de Correlação** entre as variáveis X e Y dadas. Usando um gráfico de dispersão, você pode visualizar rapidamente esse coeficiente. Um gráfico com pontos de dados alinhados em uma linha limpa tem alta correlação, mas um gráfico com pontos dispersos por toda parte tem baixa correlação.

Um bom modelo de regressão linear será aquele que tem um alto Coeficiente de Correlação (mais próximo de 1 do que de 0) usando o método de Regressão de Mínimos Quadrados com uma linha de regressão.

✅ Execute o notebook que acompanha esta lição e observe o gráfico de dispersão Mês versus Preço. Os dados associando Mês a Preço para vendas de abóboras parecem ter alta ou baixa correlação, segundo sua interpretação visual do gráfico de dispersão? Isso muda se você usar uma medida mais detalhada em vez de `Mês`, por exemplo, *dia do ano* (ou seja, número de dias desde o início do ano)?

No código abaixo, assumiremos que limpamos os dados e obtivemos um data frame chamado `new_pumpkins`, semelhante ao seguinte:

ID | Mês | DiaDoAno | Variedade | Cidade | Embalagem | Preço Baixo | Preço Alto | Preço  
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | TIPO TORTA | BALTIMORE | Caixas de 1 1/9 alqueire | 15.0 | 15.0 | 13.636364  
71 | 9 | 267 | TIPO TORTA | BALTIMORE | Caixas de 1 1/9 alqueire | 18.0 | 18.0 | 16.363636  
72 | 10 | 274 | TIPO TORTA | BALTIMORE | Caixas de 1 1/9 alqueire | 18.0 | 18.0 | 16.363636  
73 | 10 | 274 | TIPO TORTA | BALTIMORE | Caixas de 1 1/9 alqueire | 17.0 | 17.0 | 15.454545  
74 | 10 | 281 | TIPO TORTA | BALTIMORE | Caixas de 1 1/9 alqueire | 15.0 | 15.0 | 13.636364  

> O código para limpar os dados está disponível em [`notebook.ipynb`](notebook.ipynb). Realizamos os mesmos passos de limpeza da lição anterior e calculamos a coluna `DayOfYear` usando a seguinte expressão:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```
  
Agora que você entende a matemática por trás da regressão linear, vamos criar um modelo de Regressão para ver se podemos prever qual embalagem de abóboras terá os melhores preços. Alguém comprando abóboras para um campo de abóboras de feriado pode querer essa informação para otimizar suas compras.

## Buscando correlação

[![ML para iniciantes - Buscando Correlação: A Chave para Regressão Linear](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML para iniciantes - Buscando Correlação: A Chave para Regressão Linear")

> 🎥 Clique na imagem acima para um vídeo curto sobre correlação.

Na lição anterior, você provavelmente viu que o preço médio para diferentes meses parece assim:

<img alt="Preço médio por mês" src="../../../../translated_images/pt-BR/barchart.a833ea9194346d76.webp" width="50%"/>

Isso sugere que deve haver alguma correlação, e podemos tentar treinar um modelo de regressão linear para prever a relação entre `Mês` e `Preço`, ou entre `DiaDoAno` e `Preço`. Aqui está o gráfico de dispersão que mostra a segunda relação:

<img alt="Gráfico de dispersão de Preço vs. Dia do Ano" src="../../../../translated_images/pt-BR/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Vamos ver se há correlação usando a função `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```
  
Parece que a correlação é bem pequena, -0.15 pelo `Mês` e -0.17 pelo `DiaDoAno`, mas pode haver outra relação importante. Parece que há diferentes grupos de preços correspondendo a diferentes variedades de abóbora. Para confirmar essa hipótese, vamos plotar cada categoria de abóbora com uma cor diferente. Passando um parâmetro `ax` para a função de plotagem `scatter`, podemos plotar todos os pontos no mesmo gráfico:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```
  
<img alt="Gráfico de dispersão de Preço vs. Dia do Ano" src="../../../../translated_images/pt-BR/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Nossa investigação sugere que a variedade tem mais efeito no preço geral do que a própria data de venda. Podemos ver isso em um gráfico de barras:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```
  
<img alt="Gráfico de barras de preço por variedade" src="../../../../translated_images/pt-BR/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />  

Vamos focar por enquanto em apenas uma variedade de abóbora, o 'tipo torta', e ver qual efeito a data tem no preço:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Gráfico de dispersão de Preço vs. Dia do Ano" src="../../../../translated_images/pt-BR/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" />  

Se agora calcularmos a correlação entre `Preço` e `DiaDoAno` usando a função `corr`, obteremos algo como `-0.27` - o que significa que faz sentido treinar um modelo preditivo.

> Antes de treinar um modelo de regressão linear, é importante garantir que nossos dados estejam limpos. A regressão linear não funciona bem com valores ausentes, por isso faz sentido eliminar todas as células vazias:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```
  
Outra abordagem seria preencher esses valores vazios com a média da coluna correspondente.

## Regressão Linear Simples

[![ML para iniciantes - Regressão Linear e Polinomial usando Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML para iniciantes - Regressão Linear e Polinomial usando Scikit-learn")

> 🎥 Clique na imagem acima para um vídeo curto sobre regressão linear e polinomial.

Para treinar nosso modelo de Regressão Linear, usaremos a biblioteca **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```
  
Começamos separando os valores de entrada (features) e a saída esperada (rótulo) em arrays numpy separados:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```
  
> Note que tivemos que realizar `reshape` nos dados de entrada para que o pacote Linear Regression os entendesse corretamente. A Regressão Linear espera uma matriz 2D como entrada, onde cada linha da matriz corresponde a um vetor de características de entrada. No nosso caso, já que temos apenas uma entrada, precisamos de uma matriz com formato N×1, onde N é o tamanho do conjunto de dados.

Depois, precisamos dividir os dados em datasets de treino e teste, para que possamos validar nosso modelo após o treinamento:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
  
Finalmente, o treinamento do modelo de Regressão Linear em si leva apenas duas linhas de código. Definimos o objeto `LinearRegression` e o ajustamos aos nossos dados usando o método `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```
  
O objeto `LinearRegression` após o `fit` contém todos os coeficientes da regressão, que podem ser acessados usando a propriedade `.coef_`. No nosso caso, há apenas um coeficiente, que deve estar em torno de `-0.017`. Isso significa que os preços parecem cair um pouco com o tempo, mas não muito, cerca de 2 centavos por dia. Também podemos acessar o ponto de interseção da regressão com o eixo Y usando `lin_reg.intercept_` — será cerca de `21` no nosso caso, indicando o preço no início do ano.

Para ver quão preciso nosso modelo é, podemos prever preços em um conjunto de dados de teste e depois medir quão próximas estão nossas previsões dos valores esperados. Isso pode ser feito usando a métrica raiz do erro quadrático médio (RMSE), que é a raiz da média de todas as diferenças quadráticas entre o valor esperado e o previsto.

```python
pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')
```

Nosso erro parece ser em torno de 2 pontos, o que é ~17%. Não muito bom. Outro indicador da qualidade do modelo é o **coeficiente de determinação**, que pode ser obtido assim:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
 Se o valor for 0, significa que o modelo não leva os dados de entrada em conta e atua como o *pior preditor linear*, que é simplesmente um valor médio do resultado. O valor 1 significa que podemos prever perfeitamente todas as saídas esperadas. No nosso caso, o coeficiente está em torno de 0,06, que é bastante baixo.

Também podemos plotar os dados de teste junto com a linha de regressão para ver melhor como a regressão funciona no nosso caso:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/pt-BR/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regressão Polinomial

Outro tipo de Regressão Linear é a Regressão Polinomial. Embora às vezes haja uma relação linear entre variáveis — quanto maior a abóbora em volume, maior o preço — às vezes essas relações não podem ser plotadas como um plano ou linha reta.

✅ Aqui estão [mais alguns exemplos](https://online.stat.psu.edu/stat501/Lesson09#polynomial-regression-examples) de dados que poderiam usar Regressão Polinomial

Dê outra olhada na relação entre Data e Preço. Este diagrama de dispersão parece que deve necessariamente ser analisado por uma linha reta? Os preços não podem oscilar? Neste caso, você pode tentar regressão polinomial.

✅ Polinômios são expressões matemáticas que podem consistir em uma ou mais variáveis e coeficientes.

A regressão polinomial cria uma linha curva para se ajustar melhor a dados não lineares. No nosso caso, se incluirmos uma variável `DayOfYear` ao quadrado nos dados de entrada, devemos ser capazes de ajustar nossos dados com uma curva parabólica, que terá um mínimo em um certo ponto dentro do ano.

O Scikit-learn inclui uma API de [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) útil para combinar diferentes etapas de processamento de dados. Um **pipeline** é uma cadeia de **estimadores**. No nosso caso, criaremos um pipeline que primeiro adiciona características polinomiais ao nosso modelo e depois treina a regressão:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Usar `PolynomialFeatures(2)` significa que incluiremos todos os polinômios de segundo grau dos dados de entrada. No nosso caso, isso significará apenas `DayOfYear`<sup>2</sup>, mas dado duas variáveis de entrada X e Y, isso adicionará X<sup>2</sup>, XY e Y<sup>2</sup>. Também podemos usar polinômios de grau mais alto, se quisermos.

Pipelines podem ser usados da mesma forma que o objeto original `LinearRegression`, ou seja, podemos `fit` o pipeline e depois usar `predict` para obter os resultados da predição:

```python
pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Para plotar a curva aproximada suave, usamos `np.linspace` para criar um intervalo uniforme de valores de entrada, em vez de plotar diretamente nos dados de teste não ordenados (o que produziria uma linha em zigue-zague):

```python
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
y_range = pipeline.predict(X_range)

plt.scatter(X_test, y_test)
plt.plot(X_range, y_range)
```

Aqui está o gráfico mostrando os dados de teste e a curva aproximada:

<img alt="Polynomial regression" src="../../../../translated_images/pt-BR/poly-results.ee587348f0f1f60b.webp" width="50%" />

Usando Regressão Polinomial, podemos obter um RMSE ligeiramente menor e uma determinação maior, mas não significativamente. Precisamos levar em conta outras características!

> Você pode ver que os preços mínimos de abóboras são observados em algum momento próximo ao Halloween. Como você pode explicar isso?

🎃 Parabéns, você acabou de criar um modelo que pode ajudar a prever o preço de abóboras para torta. Você provavelmente pode repetir o mesmo procedimento para todos os tipos de abóboras, mas isso seria tedioso. Vamos aprender agora como levar a variedade da abóbora em conta no nosso modelo!

## Características Categóricas

No mundo ideal, queremos ser capazes de prever preços para diferentes variedades de abóboras usando o mesmo modelo. No entanto, a coluna `Variety` é um pouco diferente de colunas como `Month`, pois contém valores não numéricos. Essas colunas são chamadas de **categóricas**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Clique na imagem acima para um breve vídeo sobre o uso de características categóricas.

Aqui você pode ver como o preço médio depende da variedade:

<img alt="Average price by variety" src="../../../../translated_images/pt-BR/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Para levar a variedade em conta, primeiro precisamos convertê-la para uma forma numérica, ou **codificá-la**. Existem várias maneiras de fazer isso:

* Uma simples **codificação numérica** irá construir uma tabela das diferentes variedades e depois substituir o nome da variedade por um índice nessa tabela. Isso não é uma boa ideia para regressão linear, porque a regressão linear usa o valor numérico real do índice e o adiciona ao resultado, multiplicando por algum coeficiente. No nosso caso, a relação entre o número do índice e o preço é claramente não linear, mesmo que ordenemos os índices de alguma forma específica.
* **One-hot encoding** substituirá a coluna `Variety` por 4 colunas diferentes, uma para cada variedade. Cada coluna conterá `1` se a linha correspondente for daquela variedade, e `0` caso contrário. Isso significa que haverá quatro coeficientes na regressão linear, um para cada variedade de abóbora, responsável pelo "preço inicial" (ou melhor, "preço adicional") para aquela variedade em particular.

O código abaixo mostra como podemos fazer one-hot encoding para uma variedade:

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

Para treinar regressão linear usando a variedade codificada por one-hot como entrada, só precisamos inicializar os dados `X` e `y` corretamente:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

O resto do código é o mesmo que usamos acima para treinar a Regressão Linear. Se você tentar, verá que o erro médio quadrático é aproximadamente o mesmo, mas obtemos um coeficiente de determinação muito maior (~77%). Para obter previsões ainda mais precisas, podemos levar em conta mais características categóricas, bem como características numéricas, como `Month` ou `DayOfYear`. Para obter um grande array de características, podemos usar `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Aqui também levamos em conta `City` e `Package`, o que nos dá RMSE 2,84 (10,5%) e determinação 0,94!

## Colocando tudo junto

Para fazer o melhor modelo, podemos usar dados combinados (categóricos codificados por one-hot + numéricos) do exemplo acima junto com Regressão Polinomial. Aqui está o código completo para sua conveniência:

```python
# configurar dados de treinamento
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# fazer divisão de treino-teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# configurar e treinar o pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# prever resultados para dados de teste
pred = pipeline.predict(X_test)

# calcular RMSE e coeficiente de determinação
rmse = mean_squared_error(y_test, pred, squared=False)
print(f'RMSE: {rmse:3.3} ({rmse/pred.mean()*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Isso deve nos dar o melhor coeficiente de determinação de quase 97%, e RMSE=2,23 (~8% de erro de previsão).

| Modelo | RMSE | Determinação |
|-------|-----|---------------|
| Linear `DayOfYear` | 2,77 (17,2%) | 0,07 |
| Polinomial `DayOfYear` | 2,73 (17,0%) | 0,08 |
| Linear `Variety` | 5,24 (19,7%) | 0,77 |
| Linear com todas características | 2,84 (10,5%) | 0,94 |
| Polinomial com todas características | 2,23 (8,25%) | 0,97 |

🏆 Muito bem! Você criou quatro modelos de regressão em uma lição e melhorou a qualidade do modelo para 97%. Na seção final sobre Regressão, você aprenderá sobre Regressão Logística para determinar categorias.

---
## 🚀Desafio

Teste várias variáveis diferentes neste notebook para ver como a correlação corresponde à precisão do modelo.

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Nesta lição aprendemos sobre Regressão Linear. Existem outros tipos importantes de Regressão. Leia sobre as técnicas Stepwise, Ridge, Lasso e Elasticnet. Um bom curso para estudar e aprender mais é o [curso de Aprendizado Estatístico de Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Tarefa

[Construa um Modelo](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:  
Este documento foi traduzido usando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, por favor, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->