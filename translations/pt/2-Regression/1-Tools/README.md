<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T08:37:58+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "pt"
}
-->
# Introdução ao Python e Scikit-learn para modelos de regressão

![Resumo de regressões em um sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introdução

Nestes quatro módulos, irá aprender a construir modelos de regressão. Vamos discutir brevemente para que servem. Mas antes de começar, certifique-se de que tem as ferramentas certas para iniciar o processo!

Nesta lição, irá aprender a:

- Configurar o seu computador para tarefas locais de machine learning.
- Trabalhar com Jupyter notebooks.
- Utilizar Scikit-learn, incluindo a instalação.
- Explorar regressão linear com um exercício prático.

## Instalações e configurações

[![ML para iniciantes - Configure as suas ferramentas para criar modelos de Machine Learning](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML para iniciantes - Configure as suas ferramentas para criar modelos de Machine Learning")

> 🎥 Clique na imagem acima para um vídeo curto sobre como configurar o seu computador para ML.

1. **Instale o Python**. Certifique-se de que o [Python](https://www.python.org/downloads/) está instalado no seu computador. Irá utilizar Python para muitas tarefas de ciência de dados e machine learning. A maioria dos sistemas já inclui uma instalação do Python. Existem também [Pacotes de Codificação Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) úteis para facilitar a configuração para alguns utilizadores.

   No entanto, algumas utilizações do Python requerem uma versão específica do software. Por isso, é útil trabalhar num [ambiente virtual](https://docs.python.org/3/library/venv.html).

2. **Instale o Visual Studio Code**. Certifique-se de que tem o Visual Studio Code instalado no seu computador. Siga estas instruções para [instalar o Visual Studio Code](https://code.visualstudio.com/) para uma instalação básica. Vai utilizar Python no Visual Studio Code neste curso, por isso pode ser útil rever como [configurar o Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) para desenvolvimento em Python.

   > Familiarize-se com Python ao explorar esta coleção de [módulos de aprendizagem](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Configurar Python com Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configurar Python com Visual Studio Code")
   >
   > 🎥 Clique na imagem acima para um vídeo: usar Python no VS Code.

3. **Instale o Scikit-learn**, seguindo [estas instruções](https://scikit-learn.org/stable/install.html). Como precisa de garantir que utiliza Python 3, é recomendado que use um ambiente virtual. Note que, se estiver a instalar esta biblioteca num Mac com M1, existem instruções específicas na página acima.

4. **Instale o Jupyter Notebook**. Será necessário [instalar o pacote Jupyter](https://pypi.org/project/jupyter/).

## O seu ambiente de desenvolvimento para ML

Irá utilizar **notebooks** para desenvolver o seu código Python e criar modelos de machine learning. Este tipo de ficheiro é uma ferramenta comum para cientistas de dados e pode ser identificado pela extensão `.ipynb`.

Os notebooks são um ambiente interativo que permite ao programador tanto codificar como adicionar notas e escrever documentação em torno do código, o que é bastante útil para projetos experimentais ou de pesquisa.

[![ML para iniciantes - Configurar Jupyter Notebooks para começar a criar modelos de regressão](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML para iniciantes - Configurar Jupyter Notebooks para começar a criar modelos de regressão")

> 🎥 Clique na imagem acima para um vídeo curto sobre este exercício.

### Exercício - trabalhar com um notebook

Nesta pasta, encontrará o ficheiro _notebook.ipynb_.

1. Abra o _notebook.ipynb_ no Visual Studio Code.

   Um servidor Jupyter será iniciado com Python 3+. Encontrará áreas do notebook que podem ser `executadas`, ou seja, blocos de código. Pode executar um bloco de código selecionando o ícone que parece um botão de reprodução.

2. Selecione o ícone `md` e adicione um pouco de markdown com o seguinte texto: **# Bem-vindo ao seu notebook**.

   Em seguida, adicione algum código Python.

3. Escreva **print('hello notebook')** no bloco de código.
4. Selecione a seta para executar o código.

   Deverá ver a seguinte saída:

    ```output
    hello notebook
    ```

![VS Code com um notebook aberto](../../../../2-Regression/1-Tools/images/notebook.jpg)

Pode intercalar o seu código com comentários para auto-documentar o notebook.

✅ Pense por um momento como o ambiente de trabalho de um programador web é diferente do de um cientista de dados.

## Começar com Scikit-learn

Agora que o Python está configurado no seu ambiente local e está confortável com Jupyter notebooks, vamos ficar igualmente confortáveis com o Scikit-learn (pronuncia-se `sci` como em `science`). O Scikit-learn fornece uma [API extensa](https://scikit-learn.org/stable/modules/classes.html#api-ref) para ajudá-lo a realizar tarefas de ML.

De acordo com o [site oficial](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn é uma biblioteca de machine learning de código aberto que suporta aprendizagem supervisionada e não supervisionada. Também fornece várias ferramentas para ajuste de modelos, pré-processamento de dados, seleção e avaliação de modelos, entre outras utilidades."

Neste curso, irá utilizar Scikit-learn e outras ferramentas para construir modelos de machine learning para realizar o que chamamos de tarefas de 'machine learning tradicional'. Evitamos deliberadamente redes neurais e deep learning, pois estes tópicos serão abordados no nosso futuro currículo 'AI for Beginners'.

O Scikit-learn torna simples a construção e avaliação de modelos para uso. Ele é focado principalmente no uso de dados numéricos e contém vários conjuntos de dados prontos para uso como ferramentas de aprendizagem. Também inclui modelos pré-construídos para os alunos experimentarem. Vamos explorar o processo de carregar dados pré-embalados e usar um estimador para o primeiro modelo de ML com Scikit-learn com alguns dados básicos.

## Exercício - o seu primeiro notebook com Scikit-learn

> Este tutorial foi inspirado pelo [exemplo de regressão linear](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) no site do Scikit-learn.

[![ML para iniciantes - O seu primeiro projeto de regressão linear em Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML para iniciantes - O seu primeiro projeto de regressão linear em Python")

> 🎥 Clique na imagem acima para um vídeo curto sobre este exercício.

No ficheiro _notebook.ipynb_ associado a esta lição, limpe todas as células pressionando o ícone da 'lixeira'.

Nesta secção, irá trabalhar com um pequeno conjunto de dados sobre diabetes que está incluído no Scikit-learn para fins de aprendizagem. Imagine que queria testar um tratamento para pacientes diabéticos. Modelos de Machine Learning podem ajudá-lo a determinar quais pacientes responderiam melhor ao tratamento, com base em combinações de variáveis. Mesmo um modelo de regressão muito básico, quando visualizado, pode mostrar informações sobre variáveis que o ajudariam a organizar os seus ensaios clínicos teóricos.

✅ Existem muitos tipos de métodos de regressão, e a escolha depende da resposta que procura. Se quiser prever a altura provável de uma pessoa com uma determinada idade, utilizaria regressão linear, pois está à procura de um **valor numérico**. Se estiver interessado em descobrir se um tipo de cozinha deve ser considerado vegan ou não, está à procura de uma **atribuição de categoria**, então utilizaria regressão logística. Aprenderá mais sobre regressão logística mais tarde. Pense um pouco sobre algumas perguntas que pode fazer aos dados e qual destes métodos seria mais apropriado.

Vamos começar esta tarefa.

### Importar bibliotecas

Para esta tarefa, iremos importar algumas bibliotecas:

- **matplotlib**. É uma ferramenta útil para [criação de gráficos](https://matplotlib.org/) e será usada para criar um gráfico de linha.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) é uma biblioteca útil para manipulação de dados numéricos em Python.
- **sklearn**. Esta é a biblioteca [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importe algumas bibliotecas para ajudar nas suas tarefas.

1. Adicione as importações escrevendo o seguinte código:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Acima, está a importar `matplotlib`, `numpy` e está a importar `datasets`, `linear_model` e `model_selection` do `sklearn`. `model_selection` é usado para dividir os dados em conjuntos de treino e teste.

### O conjunto de dados sobre diabetes

O [conjunto de dados sobre diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) incluído possui 442 amostras de dados sobre diabetes, com 10 variáveis de características, algumas das quais incluem:

- age: idade em anos
- bmi: índice de massa corporal
- bp: pressão arterial média
- s1 tc: T-Cells (um tipo de glóbulos brancos)

✅ Este conjunto de dados inclui o conceito de 'sexo' como uma variável de característica importante para a pesquisa sobre diabetes. Muitos conjuntos de dados médicos incluem este tipo de classificação binária. Pense um pouco sobre como categorizações como esta podem excluir certas partes da população de tratamentos.

Agora, carregue os dados X e y.

> 🎓 Lembre-se, isto é aprendizagem supervisionada, e precisamos de um alvo 'y' nomeado.

Numa nova célula de código, carregue o conjunto de dados sobre diabetes chamando `load_diabetes()`. O parâmetro `return_X_y=True` indica que `X` será uma matriz de dados e `y` será o alvo da regressão.

1. Adicione alguns comandos `print` para mostrar a forma da matriz de dados e o seu primeiro elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    O que está a receber como resposta é uma tupla. O que está a fazer é atribuir os dois primeiros valores da tupla a `X` e `y`, respetivamente. Saiba mais [sobre tuplas](https://wikipedia.org/wiki/Tuple).

    Pode ver que estes dados têm 442 itens organizados em arrays de 10 elementos:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pense um pouco sobre a relação entre os dados e o alvo da regressão. A regressão linear prevê relações entre a característica X e a variável alvo y. Consegue encontrar o [alvo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) para o conjunto de dados sobre diabetes na documentação? O que este conjunto de dados está a demonstrar, dado o alvo?

2. Em seguida, selecione uma parte deste conjunto de dados para plotar, escolhendo a 3ª coluna do conjunto de dados. Pode fazer isso utilizando o operador `:` para selecionar todas as linhas e, em seguida, selecionando a 3ª coluna usando o índice (2). Também pode remodelar os dados para serem um array 2D - conforme necessário para plotagem - utilizando `reshape(n_rows, n_columns)`. Se um dos parâmetros for -1, a dimensão correspondente será calculada automaticamente.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ A qualquer momento, imprima os dados para verificar a sua forma.

3. Agora que tem os dados prontos para serem plotados, pode verificar se uma máquina pode ajudar a determinar uma divisão lógica entre os números neste conjunto de dados. Para isso, precisa de dividir tanto os dados (X) quanto o alvo (y) em conjuntos de teste e treino. O Scikit-learn tem uma forma simples de fazer isso; pode dividir os seus dados de teste num ponto específico.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Agora está pronto para treinar o seu modelo! Carregue o modelo de regressão linear e treine-o com os seus conjuntos de treino X e y utilizando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` é uma função que verá em muitas bibliotecas de ML, como TensorFlow.

5. Em seguida, crie uma previsão utilizando os dados de teste, com a função `predict()`. Isto será usado para desenhar a linha entre os grupos de dados.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Agora é hora de mostrar os dados num gráfico. O Matplotlib é uma ferramenta muito útil para esta tarefa. Crie um scatterplot de todos os dados de teste X e y, e utilize a previsão para desenhar uma linha no local mais apropriado, entre os agrupamentos de dados do modelo.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![um scatterplot mostrando pontos de dados sobre diabetes](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Pensa um pouco sobre o que está a acontecer aqui. Uma linha reta está a passar por muitos pequenos pontos de dados, mas o que está realmente a fazer? Consegues perceber como deverias ser capaz de usar esta linha para prever onde um novo ponto de dados, ainda não visto, deveria encaixar em relação ao eixo y do gráfico? Tenta expressar em palavras a utilidade prática deste modelo.

Parabéns, construíste o teu primeiro modelo de regressão linear, criaste uma previsão com ele e exibiste-a num gráfico!

---
## 🚀Desafio

Representa graficamente uma variável diferente deste conjunto de dados. Dica: edita esta linha: `X = X[:,2]`. Dado o objetivo deste conjunto de dados, o que consegues descobrir sobre a progressão da diabetes como doença?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo Individual

Neste tutorial, trabalhaste com regressão linear simples, em vez de regressão univariada ou múltipla. Lê um pouco sobre as diferenças entre estes métodos ou dá uma olhada [neste vídeo](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Lê mais sobre o conceito de regressão e reflete sobre que tipo de perguntas podem ser respondidas com esta técnica. Faz este [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) para aprofundar o teu entendimento.

## Tarefa

[Um conjunto de dados diferente](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.