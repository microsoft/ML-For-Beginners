# Começar com Python e Scikit-learn para modelos de regressão

![Resumo das regressões numa nota de esboço
](../../../sketchnotes/ml-regression.png)

> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Questionário pré-palestra](https://white-water-09ec41f0f.azurestaticapps.net/quiz/9/)

> ### [Esta lição está disponível em R!](./solution/R/lesson_1-R.ipynb)

## Introdução

Nestas quatro lições, você vai descobrir como construir modelos de regressão. Discutiremos para que são em breve. Mas antes de fazer qualquer coisa, certifique-se de ter as ferramentas certas para iniciar o processo!

Nesta lição, aprenderá a:

- Configurar o seu computador para tarefas locais de aprendizagem automática.
- Trabalhe com cadernos Jupyter.
- Utilize scikit-learn, incluindo a instalação.
- Explore a regressão linear com um exercício prático.

## Instalações e configurações

[![Configurar Python com código de estúdio visual
](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configurar Python com código de estúdio visual
")

> 🎥 Clique na imagem acima para um vídeo: utilizando Python dentro do Código VS.

1. **Instalar Python**. Certifique-se de que [Python](https://www.python.org/downloads/) está instalado no seu computador. Você usará Python para muitas tarefas de ciência de dados e machine learning. A maioria dos sistemas informáticos já inclui uma instalação Python. Há úteis [Python Pacotes de codificação](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-15963-cxa) disponível também, para facilitar a configuração para alguns utilizadores.

  Alguns usos de Python, no entanto, requerem uma versão do software, enquanto outros requerem uma versão diferente. Por esta razão, é útil trabalhar dentro de um [ambiente virtual](https://docs.python.org/3/library/venv.html).

2. **Instalar código de estúdio visual**. Certifique-se de que tem o Código do Estúdio Visual instalado no seu computador. Siga estas instruções para
[instalar Código do Estúdio Visual](https://code.visualstudio.com/) para a instalação básica. Você vai usar Python em Código estúdio visual neste curso, então você pode querer relembrá-lo [configurar código de estúdio visual](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-15963-cxa) para o desenvolvimento de Python.

> Fique confortável com python trabalhando através desta coleção de [Aprender módulos](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-15963-cxa)

3. **Instale Scikit-learn**, seguindo [estas instruções]
(https://scikit-learn.org/stable/install.html). Uma vez que precisa de garantir que utiliza o Python 3, recomenda-se que utilize um ambiente virtual. Note que se estiver a instalar esta biblioteca num Mac M1, existem instruções especiais na página acima ligada.

1. **Instale o Caderno Jupyter**. Você precisará [instalar o pacote Jupyter](https://pypi.org/project/jupyter/).

## O seu ambiente de autoria ML
Você vai usar **cadernos** para desenvolver o seu código Python e criar modelos de aprendizagem automática. Este tipo de ficheiro é uma ferramenta comum para cientistas de dados, e podem ser identificados pelo seu sufixo ou extensão `.ipynb`.

Os cadernos são um ambiente interativo que permite ao desenvolvedor codificar e adicionar notas e escrever documentação em torno do código que é bastante útil para projetos experimentais ou orientados para a investigação.

## Exercício - trabalhe com um caderno

Nesta pasta, encontrará o ficheiro _notebook.ipynb_.

1. Abra _notebook.ipynb_ em Código de Estúdio Visual.
   
   Um servidor Jupyter começará com o Python 3+ iniciado. Encontrará áreas do caderno que podem ser `executadas`, peças de código. Pode executar um bloco de código, selecionando o ícone que parece um botão de reprodução.

2. Selecione o ícone `md` e adicione um pouco de marcação, e o seguinte texto **# Bem-vindo ao seu caderno**.

   Em seguida, adicione um pouco de código Python.

5. Escreva **print ('olá caderno')** no bloco de código.
   
6. Selecione a seta para executar o código.

 Deve ver a declaração impressa:

 ```saída
Olá caderno
```
![Código VS com um caderno aberto](../images/notebook.jpg)

Pode interligar o seu código com comentários para auto-documentar o caderno.

✅ Pense por um minuto como o ambiente de trabalho de um web developer é diferente do de um cientista de dados.

## Em funcionamento com Scikit-learn

Agora que python está montado no seu ambiente local, e você está confortável com os cadernos jupyter, vamos ficar igualmente confortáveis com Scikit-learn (pronunciá-lo 'sci' como em 'ciência'). Scikit-learn fornece uma [API extensiva](https://scikit-learn.org/stable/modules/classes.html#api-ref) para ajudá-lo a executar tarefas ML.

De acordo com o seu [site](https://scikit-learn.org/stable/getting_started.html), "O Scikit-learn é uma biblioteca de aprendizagem automática de código aberto que suporta a aprendizagem supervisionada e sem supervisão. Também fornece várias ferramentas para a montagem de modelos, pré-processamento de dados, seleção e avaliação de modelos, e muitas outras utilidades."

Neste curso, você usará scikit-learn e outras ferramentas para construir modelos de machine learning para executar o que chamamos de tarefas tradicionais de aprendizagem automática. Evitámos deliberadamente redes neurais e aprendizagem profunda, uma vez que estão melhor cobertas no nosso próximo currículo de IA para principiantes.



O scikit-learn torna simples construir modelos e avaliá-los para uso. Está focado principalmente na utilização de dados numéricos e contém vários conjuntos de dados prontos para uso como ferramentas de aprendizagem. Também inclui modelos pré-construídos para os alunos experimentarem. Vamos explorar o processo de carregamento de dados pré-embalados e usar um modelo ml incorporado no estimador com o Scikit-learn com alguns dados básicos.

## Exercício - o seu primeiro caderno Scikit-learn

> Este tutorial foi inspirado no exemplo [de regressão linear](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) no site da Scikit-learn.

No ficheiro _notebook.ipynb_ associado a esta lição, limpe todas as células premindo o ícone 'caixote do lixo'.

Nesta secção, você vai trabalhar com um pequeno conjunto de dados sobre diabetes que é incorporado em Scikit-learn para fins de aprendizagem. Imagine que queria testar um tratamento para pacientes diabéticos. Os modelos de Machine Learning podem ajudá-lo a determinar quais os pacientes que responderiam melhor ao tratamento, com base em combinações de variáveis. Mesmo um modelo de regressão muito básico, quando visualizado, pode mostrar informações sobre variáveis que o ajudariam a organizar os seus ensaios clínicos teóricos.

✅ There are many types of regression methods, and which one you pick depends on the answer you're looking for. If you want to predict the probable height for a person of a given age, you'd use linear regression, as you're seeking a **numeric value**. If you're interested in discovering whether a type of cuisine should be considered vegan or not, you're looking for a **category assignment** so you would use logistic regression. You'll learn more about logistic regression later. Think a bit about some questions you can ask of data, and which of these methods would be more appropriate.

Vamos começar com esta tarefa.

### Bibliotecas de importação

Para esta tarefa importaremos algumas bibliotecas:

- **matplotlib**. É uma ferramenta útil [de grafimento](https://matplotlib.org/) e vamos usá-lo para criar um enredo de linha.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) é uma biblioteca útil para o tratamento de dados numéricos em Python.
- **sklearn**. Este é o [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) biblioteca.

Importe algumas bibliotecas para ajudar nas suas tarefas.

1. Adicione as importações digitando o seguinte código:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Acima está a importar `matplottlib`, `numpy` e está a importar `datasets`, `linear_model` e `model_selection` de `sklearn`. É utilizado `model_selection` para dividir dados em conjuntos de treino e teste.

   ## O conjunto de dados da diabetes
   O conjunto de dados incorporado [diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) Inclui 442 amostras de dados em torno da diabetes, com 10 variáveis de características, algumas das quais incluem:

   - idade: idade em anos
   - bmi: índice de massa corporal
   - bp: pressão arterial média
   - s1 tc: T-Cells (um tipo de glóbulos brancos)
  
  ✅ Este conjunto de dados inclui o conceito de 'sexo' como uma variável de característica importante para a investigação em torno da diabetes. Muitos conjuntos de dados médicos incluem este tipo de classificação binária. Pense um pouco sobre como categorizações como esta podem excluir certas partes de uma população de tratamentos.

Agora, carregue os dados X e Y.

> 🎓 Lembre-se, isto é aprendizagem supervisionada, e precisamos de um alvo chamado "y".

Numa nova célula de código, carregue o conjunto de dados da diabetes chamando `load_diabetes()` A entrada `return_X_y=True` indica que `X` será uma matriz de dados, e `y` será o alvo de regressão.

1. Adicione alguns comandos de impressão para mostrar a forma da matriz de dados e o seu primeiro elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    O que estás a receber como resposta, é um tuple. O que está a fazer é atribuir os dois primeiros valores da tuple para `X` and `y` respectivamente. Saiba mais [sobre tuples](https://wikipedia.org/wiki/Tuple).

   Pode ver que estes dados têm 442 itens moldados em matrizes de 10 elementos:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pense um pouco sobre a relação entre os dados e o alvo de regressão. A regressão linear prevê relações entre a característica X e a variável alvo. Pode encontrar o [alvo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) para o conjunto de dados da diabetes na documentação? O que é que este conjunto de dados está a demonstrar, tendo em conta esse objetivo?

2. Em seguida, selecione uma parte deste conjunto de dados para traçar, organizando-o numa nova matriz usando a função `newaxis` da Numpy. Vamos usar a regressão linear para gerar uma linha entre valores nestes dados, de acordo com um padrão que determina.

   ```python
   X = X[:, np.newaxis, 2]
   ```

   ✅ A qualquer momento, imprima os dados para verificar a sua forma.
   
   3. Agora que tem dados prontos a serem traçados, pode ver se uma máquina pode ajudar a determinar uma divisão lógica entre os números deste conjunto de dados. Para isso, é necessário dividir os dados (X) e o alvo (y) em conjuntos de teste e treino. O Scikit-learn tem uma forma simples de o fazer; pode dividir os seus dados de teste num dado momento.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Agora está pronto para treinar o seu modelo! Carregue o modelo linear de regressão e treine-o com os seus conjuntos de treinamento X e y usando `modelo.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `modelo.fit()` é uma função que você verá em muitas bibliotecas ML, como TensorFlow

5. Em seguida, crie uma previsão utilizando dados de teste, utilizando a função `predict()`. Isto será usado para traçar a linha entre grupos de dados
    ```python
    y_pred = model.predict(X_test)
    ```

6. Agora é hora de mostrar os dados num enredo. Matplotlib é uma ferramenta muito útil para esta tarefa. Crie uma dispersão de todos os dados de teste X e y, e use a previsão para traçar uma linha no local mais apropriado, entre os agrupamentos de dados do modelo.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![a scatterplot showing datapoints around diabetes](../images/scatterplot.png)

   ✅ Pense um pouco sobre o que está acontecendo aqui. Uma linha reta está a passar por muitos pequenos pontos de dados, mas o que está a fazer exatamente? Consegue ver como deve ser capaz de usar esta linha para prever onde um novo ponto de dados invisível se deve encaixar em relação ao eixo y do enredo? Tente colocar em palavras o uso prático deste modelo.

Parabéns, construíste o teu primeiro modelo linear de regressão, criaste uma previsão com ele, e exibiste-o num enredo!
---
## 🚀Challenge

Defina uma variável diferente deste conjunto de dados. Dica: edite esta linha:`X = X[:, np.newaxis, 2]`. Tendo em conta o objetivo deste conjunto de dados, o que é que consegue descobrir sobre a progressão da diabetes como uma doença?
## [Questionário pós-palestra](https://white-water-09ec41f0f.azurestaticapps.net/quiz/10/)

## Review & Self Study

Neste tutorial, trabalhou com uma simples regressão linear, em vez de univariado ou regressão linear múltipla. Leia um pouco sobre as diferenças entre estes métodos, ou dê uma olhada[este vídeo](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Leia mais sobre o conceito de regressão e pense sobre que tipo de perguntas podem ser respondidas por esta técnica. Tome este [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-15963-cxa) para aprofundar a sua compreensão.
## Missão 

[Um conjunto de dados diferente](assignment.md)
