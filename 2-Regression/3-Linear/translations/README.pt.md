# Crie um modelo de regressão utilizando o Scikit-learning: regressão de dois modos

![Regressão linear vs polinomial infográfica](./images/linear-polynomial.png)
> Infográfico de [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Questionário pré-seleção](https://white-water-09ec41f0f.azurestaticapps.net/quiz/13/)

> ### [Esta lição está disponível em R!](./solution/R/lesson_3-R.ipynb)
### Introdução

Até agora, vocês exploraram o que é a regressão com os dados de exemplo recolhidos a partir do conjunto de dados de preços da abóbora que vamos usar ao longo desta lição. Também o visualizaram utilizando Matplotlib.

Agora está preparado para mergulhar mais profundamente na regressão para o ML. Nesta lição, você vai aprender mais sobre dois tipos de regressão: _regressão linear básica_ e _regressão polinomial_, juntamente com alguma da matemática subjacente a estas técnicas.

> Ao longo deste currículo, assumimos um conhecimento mínimo de matemática, e procuramos torná-lo acessível a estudantes provenientes de outras áreas, por isso, procuremos notas, notas de 🧮, diagramas e outras ferramentas de aprendizagem para ajudar na compreensão.

### Pré-requisitos

Já devem conhecer a estrutura dos dados relativos à abóbora que estamos a analisar. Pode encontrá-lo pré-carregado e previamente limpo no ficheiro _notebook.ipynb_ desta lição. No ficheiro, o preço da abóbora é apresentado por defeito num novo dataframe.  Certifique-se de que pode executar estes blocos de notas em kernels no Código do Visual Studio.

### Preparação

Como lembrete, está a carregar estes dados para fazer perguntas sobre os mesmos.

- Quando é o melhor momento para comprar abóboras?
- Que preço posso esperar de um caso de abóbora miniatura?
- Devo comprá-los em cestos de meia-bushel ou pela caixa de bushel 1 1/9?
Vamos continuar a investigar estes dados.

Na lição anterior, você criou um dataframe Pandas e o preencheu com parte do conjunto de dados original, padronizando os preços pelo bushel. Ao fazer isso, no entanto, você só conseguiu reunir cerca de 400 datapops e apenas nos meses de outono.

Dê uma vista de olhos aos dados que pré-carregámos no bloco de notas que acompanha esta lição. Os dados são pré-carregados e um gráfico de dispersão inicial é desenhado para mostrar os dados do mês. Talvez possamos obter um pouco mais de detalhe sobre a natureza dos dados limpando-os mais.

## Uma linha de regressão linear

Como aprenderam na lição 1, o objetivo de um exercício de regressão linear é conseguir desenhar uma linha para:

- **Mostrar relações de variáveis***. Mostrar a relação entre variáveis
- **Faça previsões**. Faça previsões precisas sobre onde um novo ponto de dados cairia em relação a essa linha.

É típico de **Regressão dos Quadrados Menos** desenhar este tipo de linha. O termo 'menos quadrados' significa que todos os pontos de dados em torno da linha de regressão são são quadrados e depois adicionados. Idealmente, essa soma final é o mais pequena possível, porque queremos um número reduzido de erros, ou `menos quadrados` .

Fazemo-lo porque queremos modelar uma linha que tenha a menor distância cumulativa de todos os nossos pontos de dados. Nós também fazemos o quadrado dos termos antes de os adicionarmos, uma vez que estamos preocupados com a sua magnitude e não com a sua direção.

> ** 🧮 Mostrar a matemática**
>
> Esta linha, denominada a _linha de best fit_, pode ser expressa por [uma equação](https://en.wikipedia.org/wiki/Simple_linear_regression):
>
> ```
> Y = a + bX
> ```
>
> `X` é a "variável explicativa". `Y` é a 'variável dependente'. O declive da linha é `b` e `a` é a interceção y, que se refere ao valor de `Y` quando `X = 0`.
>
>![calcule o declive](images/slope.png)
>
> Primeiro, calcular o declive `b`. Infográfico por [Jen Looper](https://twitter.com/jenlooper)
>
> Por outras palavras, e referindo-se à pergunta original dos nossos dados de abóbora: "prever o preço de uma abóbora por bordel por mês", `X` referiria-se ao preço e `Y` referiria-se ao mês de venda.
>
>![complete a equação](images/calculation.png)
>
> Calcular o valor de Y. Se você está pagando por volta de $4, deve ser abril! Infográfico por [Jen Looper](https://twitter.com/jenlooper)
>
> A matemática que calcula a linha deve demonstrar o declive da linha, que também depende da interceção, ou onde `Y` está situado quando `X = 0`.
>
> Pode observar o método de cálculo destes valores no Web site [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Visite também [esta calculadora de Menos quadrados](https://www.mathsisfun.com/data/least-squares-calculator.html) para ver como os valores dos números têm impacto na linha.

## Correlação

Mais um termo a compreender é o **Coeficiente de Correlação** entre as variáveis X e Y fornecidas. Usando um gráfico de dispersão, você pode visualizar rapidamente este coeficiente. Um desenho com pontos de dados dispersos numa linha reta tem uma correlação elevada, mas um desenho com pontos de dados dispersos por todo o lado entre X e Y tem uma correlação baixa.

Um bom modelo de regressão linear será aquele que tem um Coeficiente de Correlação elevado (mais perto de 1 que 0) utilizando o método de Regressão dos Menos Quadrados com uma linha de regressão.

✅ Executar o bloco de notas que acompanha esta lição e olhar para o gráfico de distribuição City to Price. Os dados que associam a cidade ao preço das vendas de abóbora parecem ter uma correlação alta ou baixa, de acordo com a sua interpretação visual da distribuição?


## Preparar os dados para regressão

Agora que têm uma compreensão da matemática por detrás deste exercício, criem um modelo de Regressão para ver se conseguem prever que pacote de abóbora terá os melhores preços de abóbora. Alguém que adquira abóbora para uma correção de abóbora de férias poderá querer que esta informação seja capaz de otimizar as suas compras de pacotes de abóbora para a correção.

Já que você vai usar o Scikit-learning, não há razão para fazer isso à mão (embora você pudesse!). No bloco principal de processamento de dados do bloco de notas de lição, adicione uma biblioteca do Scikit-learning para converter automaticamente todos os dados de cadeia em números:

```python
from sklearn.preprocessing import LabelEncoder

new_pumpkins.iloc[:, 0:-1] = new_pumpkins.iloc[:, 0:-1].apply(LabelEncoder().fit_transform)
```

Se olharem para o dataframe new_bompkins, veem que todas as cadeias são agora numéricas. Isto torna mais difícil para você ler, mas muito mais inteligível para o Scikit - aprender!
Agora, pode tomar decisões mais educadas (não apenas com base no aparecimento de um gráfico de dispersão) sobre os dados que melhor se adequam à regressão.

Tente encontrar uma boa correlação entre dois pontos dos seus dados para criar, potencialmente, um bom modelo preditivo. Acontece que há apenas uma correlação fraca entre a Cidade e o Preço:

```python
print(new_pumpkins['City'].corr(new_pumpkins['Price']))
0.32363971816089226
```

No entanto, há uma correlação um pouco melhor entre o Pacote e o seu Preço. Isso faz sentido, certo? Normalmente, quanto maior for a caixa de produção, maior será o preço.

```python
print(new_pumpkins['Package'].corr(new_pumpkins['Price']))
0.6061712937226021
```

Uma boa pergunta a fazer sobre estes dados será: 'Que preço posso esperar de um determinado pacote de abóbora?'

Vamos construir este modelo de regressão

## A criar um modelo linear

Antes de criar o seu modelo, faça mais uma arrumação dos seus dados. Remova todos os dados nulos e verifique novamente como são os dados.

```python
new_pumpkins.dropna(inplace=True)
new_pumpkins.info()
```

Em seguida, crie um novo dataframe a partir deste conjunto mínimo e imprima-o:

```python
new_columns = ['Package', 'Price']
lin_pumpkins = new_pumpkins.drop([c for c in new_pumpkins.columns if c not in new_columns], axis='columns')

lin_pumpkins
```

```output
	Package	Price
70	0	13.636364
71	0	16.363636
72	0	16.363636
73	0	15.454545
74	0	13.636364
...	...	...
1738	2	30.000000
1739	2	28.750000
1740	2	25.750000
1741	2	24.000000
1742	2	24.000000
415 rows × 2 columns
```

1. Agora, pode atribuir os seus dados de coordenadas X e y:

   ```python
   X = lin_pumpkins.values[:, :1]
   y = lin_pumpkins.values[:, 1:2]
   ```
✅ O que está acontecendo aqui? Está a utilizar [Python slice notation](https://stackoverflow.com/questions/509211/understanding-slice-notation/509295#509295) para criar matrizes para povoar ‘X’ e ‘y’.

2. Em seguida, inicie as rotinas de construção de modelos de regressão:

   ```python
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   lin_reg = LinearRegression()
   lin_reg.fit(X_train,y_train)

   pred = lin_reg.predict(X_test)

   accuracy_score = lin_reg.score(X_train,y_train)
   print('Model Accuracy: ', accuracy_score)
   ```

   Porque a correlação não é particularmente boa, o modelo produzido não é terrivelmente preciso.

   ```output
   Model Accuracy:  0.3315342327998987
   ```

3. Pode visualizar a linha desenhada no processo:

   ```python
   plt.scatter(X_test, y_test,  color='black')
   plt.plot(X_test, pred, color='blue', linewidth=3)

   plt.xlabel('Package')
   plt.ylabel('Price')

   plt.show()
   ```
   ![Um gráfico de dispersão que mostra a relação preço/pacote](./images/linear.png)

4. Teste o modelo contra uma variedade hipotética:

   ```python
   lin_reg.predict( np.array([ [2.75] ]) )
   ```
   
   O preço devolvido por esta Variedades mitológicas é:

   ```output
   array([[33.15655975]])
   ```

Esse número faz sentido, se a lógica da linha de regressão se mantiver verdadeira.

🎃 Parabéns, criaram um modelo que pode ajudar a prever o preço de algumas variedades de abóbora. A sua mancha de abóbora de férias será bonita. Mas é provável que se possa criar um modelo melhor!
## Regressão polinomial

Outro tipo de regressão linear é a regressão polinomial. Embora por vezes haja uma relação linear entre variáveis - quanto maior é o volume da abóbora, maior é o preço - por vezes estas relações não podem ser desenhadas como um plano ou uma linha reta.

✅ Aqui estão [mais alguns exemplos](https://online.stat.psu.edu/stat501/lesson/9/9.8) de dados que podem utilizar regressão polinomial

Vejam outra vez a relação entre Varity e Price no desenho anterior. Parece que este gráfico de dispersão deve ser necessariamente analisado por uma linha reta? Talvez não. Neste caso, pode-se tentar uma regressão polinomial.

✅ Polinomiais são expressões matemáticas que podem ser compostas por uma ou mais variáveis e coeficientes

A regressão polinomial cria uma linha curvada para ajustar melhor os dados não lineares.

1. Vamos recriar um dataframe povoado com um segmento dos dados originais da abóbora:

   ```python
   new_columns = ['Variety', 'Package', 'City', 'Month', 'Price']
   poly_pumpkins = new_pumpkins.drop([c for c in new_pumpkins.columns if c not in new_columns], axis='columns')

   poly_pumpkins
   ```

Uma boa maneira de visualizar as correlações entre os dados nos dataframes é exibi-los em um gráfico 'colorido':

2. Utilize o método `Background_gradient()` com o valor de argumento `colarm`:

   ```python
   corr = poly_pumpkins.corr()
   corr.style.background_gradient(cmap='coolwarm')
   ```
   Este código cria um mapa de calor:
![Um mapa de calor que mostra a correlação de dados](./images/heatmap.png)

Olhando para este gráfico, pode visualizar a boa correlação entre Pacote e Preço. Portanto, deveriam ser capazes de criar um modelo um pouco melhor do que o último.
### Criar um pipeline

Scikit-learning inclui uma API útil para a construção de modelos de regressão polinomial - o `make_pipeline` [API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline). É criado um "pipeline" que é uma cadeia de estimadores. Neste caso, o pipeline inclui funcionalidades polinomiais ou previsões que formam um caminho não linear.

1. Criar as colunas X e y:

   ```python
   X=poly_pumpkins.iloc[:,3:4].values
   y=poly_pumpkins.iloc[:,4:5].values
   ```

2. Crie o pipeline chamando o método "make_pipeline()":

   ```python
   from sklearn.preprocessing import PolynomialFeatures
   from sklearn.pipeline import make_pipeline

   pipeline = make_pipeline(PolynomialFeatures(4), LinearRegression())

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

   pipeline.fit(np.array(X_train), y_train)

   y_pred=pipeline.predict(X_test)
   ```

### Criar uma sequência

Neste ponto, é necessário criar um novo dataframe com dados _sorted_ para que o pipeline possa criar uma sequência.

Adicionar o seguinte código:

   ```python
   df = pd.DataFrame({'x': X_test[:,0], 'y': y_pred[:,0]})
   df.sort_values(by='x',inplace = True)
   points = pd.DataFrame(df).to_numpy()

   plt.plot(points[:, 0], points[:, 1],color="blue", linewidth=3)
   plt.xlabel('Package')
   plt.ylabel('Price')
   plt.scatter(X,y, color="black")
   plt.show()
   ```

Criou um novo dataframe chamando `pd.DataFrame`. Em seguida, ordenou os valores chamando `sort_values()`. Finalmente criou um desenho polinomial:

![Um desenho polinomial que mostra a relação pacote/preço](./images/polynomial.png)

Pode ver uma linha curvada que se adapta melhor aos seus dados.

Vamos verificar a precisão do modelo:

   ```python
   accuracy_score = pipeline.score(X_train,y_train)
   print('Model Accuracy: ', accuracy_score)
   ```

   E voilá!

   ```output
   Model Accuracy:  0.8537946517073784
   ```

Isso é melhor! Tente prever um preço:

### Efetuar uma previsão

Podemos introduzir um novo valor e obter uma previsão?

Chame `predict()` para fazer uma previsão:
 
   ```python
   pipeline.predict( np.array([ [2.75] ]) )
   ```
   É-lhe dada esta previsão:

   ```output
   array([[46.34509342]])
   ```

Faz sentido, dado o enredo! E, se este é um modelo melhor do que o anterior, olhando para os mesmos dados, é preciso orçar para estas abrigas mais caras!

🏆 Parabéns! Criaram dois modelos de regressão numa lição. Na última secção sobre regressão, irá obter informações sobre regressão logística para determinar categorias.

—
## 🚀 desafio

Teste várias variáveis diferentes neste bloco de notas para ver como a correlação corresponde à precisão do modelo.
##[Questionário pós-palestra](https://white-water-09ec41f0f.azurestaticapps.net/quiz/14/)

## Revisão e Estudo Automático

Nesta lição, aprendemos sobre a Regressão Linear. Há outros tipos importantes de Regressão. Leia sobre as técnicas Stepwise, Ridge, Lasso e Elasticnet. Um bom curso para estudar para aprender mais é o [curso de Aprendizagem Estatística de Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Atribuição

[Criar um Modelo](assignment.md)
