# Comece com Python e Scikit-learn para modelos de regressão

![Resumo das regressões numa sketchnote](../../../../translated_images/pt-PT/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introdução

Nestes quatro módulos, irá descobrir como construir modelos de regressão. Falaremos para que servem dentro de pouco tempo. Mas antes de fazer qualquer coisa, certifique-se de que tem as ferramentas certas para iniciar o processo!

Nesta lição, vai aprender a:

- Configurar o seu computador para tarefas de machine learning local.
- Trabalhar com Jupyter Notebooks.
- Usar Scikit-learn, incluindo a instalação.
- Explorar regressão linear com um exercício prático.

## Instalações e configurações

[![ML para iniciantes - Prepare as suas ferramentas para construir modelos de Machine Learning](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML para iniciantes -Prepare as suas ferramentas para construir modelos de Machine Learning")

> 🎥 Clique na imagem acima para assistir a um vídeo breve explicando como configurar o seu computador para ML.

1. **Instale Python**. Certifique-se de que o [Python](https://www.python.org/downloads/) está instalado no seu computador. Vai usar Python para muitas tarefas de ciência de dados e machine learning. A maioria dos sistemas informáticos já inclui uma instalação Python. Existem também [pacotes de programação Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) úteis para facilitar a configuração para alguns utilizadores.

   Alguns usos do Python, no entanto, requerem uma versão do software, enquanto outros precisam de uma versão diferente. Por isso, é útil trabalhar num [ambiente virtual](https://docs.python.org/3/library/venv.html).

2. **Instale o Visual Studio Code**. Certifique-se de que tem o Visual Studio Code instalado no seu computador. Siga estas instruções para [instalar o Visual Studio Code](https://code.visualstudio.com/) para a instalação básica. Vai usar Python no Visual Studio Code neste curso, por isso pode querer familiarizar-se com a forma de [configurar o Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) para desenvolvimento em Python.

   > Familiarize-se com Python trabalhando nestes [módulos de aprendizagem](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Configure Python com Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configure Python com Visual Studio Code")
   >
   > 🎥 Clique na imagem acima para um vídeo: usar Python dentro do VS Code.

3. **Instale o Scikit-learn**, seguindo [estas instruções](https://scikit-learn.org/stable/install.html). Como precisa de assegurar que usa Python 3, recomenda-se que trabalhe num ambiente virtual. Note que, se estiver a instalar esta biblioteca num Mac M1, existem instruções especiais na página acima ligada.

1. **Instale o Jupyter Notebook**. Vai precisar de [instalar o pacote Jupyter](https://pypi.org/project/jupyter/).

## O seu ambiente de autoria ML

Vai usar **notebooks** para desenvolver o seu código Python e criar modelos de machine learning. Este tipo de ficheiro é uma ferramenta comum para cientistas de dados e pode ser identificado pelo seu sufixo ou extensão `.ipynb`.

Os notebooks são um ambiente interativo que permite ao programador tanto codificar como adicionar notas e escrever documentação em redor do código, o que é bastante útil para projetos experimentais ou orientados para investigação.

[![ML para iniciantes - Configure Jupyter Notebooks para começar a construir modelos de regressão](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML para iniciantes - Configure Jupyter Notebooks para começar a construir modelos de regressão")

> 🎥 Clique na imagem acima para um vídeo curto trabalhando este exercício.

### Exercício - trabalhe com um notebook

Nesta pasta, vai encontrar o ficheiro _notebook.ipynb_.

1. Abra _notebook.ipynb_ no Visual Studio Code.

   Um servidor Jupyter vai arrancar com Python 3+ iniciado. Vai encontrar áreas do notebook que podem ser `executadas`, pedaços de código. Pode executar um bloco de código, selecionando o ícone que parece um botão de 'play'.

1. Selecione o ícone `md` e adicione um pouco de markdown, e o seguinte texto **# Bem-vindo ao seu notebook**.

   A seguir, adicione algum código Python.

1. Escreva **print('hello notebook')** no bloco de código.
1. Selecione a seta para executar o código.

   Deve ver a frase impressa:

    ```output
    hello notebook
    ```

![VS Code com um notebook aberto](../../../../translated_images/pt-PT/notebook.4a3ee31f396b8832.webp)

Pode intercalar o seu código com comentários para auto-documentar o notebook.

✅ Pense por um minuto na diferença entre o ambiente de trabalho de um programador web e o de um cientista de dados.

## A funcionar com o Scikit-learn

Agora que o Python está instalado no seu ambiente local e se sente confortável com os Jupyter Notebooks, vamos sentir o mesmo em relação ao Scikit-learn (pronuncia-se `sci` como em `science`). O Scikit-learn fornece uma [API extensa](https://scikit-learn.org/stable/modules/classes.html#api-ref) para o ajudar a realizar tarefas de ML.

De acordo com o seu [site](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn é uma biblioteca open source de machine learning que suporta aprendizagem supervisionada e não supervisionada. Também fornece várias ferramentas para ajustar modelos, pré-processar dados, seleção e avaliação de modelos, e muitas outras utilidades."

Neste curso, vai usar o Scikit-learn e outras ferramentas para construir modelos de machine learning para realizar o que chamamos de tarefas de 'machine learning tradicional'. Evitámos propositadamente redes neuronais e deep learning, pois são mais bem abordados no nosso futuro currículo 'IA para Iniciantes'.

O Scikit-learn torna simples construir modelos e avaliá-los para utilização. Está focado principalmente em usar dados numéricos e contém vários conjuntos de dados prontos a usar como ferramentas de aprendizagem. Também inclui modelos pré-construídos para os estudantes experimentarem. Vamos explorar o processo de carregar dados pré-embalados e usar um estimador incorporado para criar o seu primeiro modelo ML com Scikit-learn usando alguns dados básicos.

## Exercício - o seu primeiro notebook Scikit-learn

> Este tutorial foi inspirado pelo [exemplo de regressão linear](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) no site do Scikit-learn.


[![ML para iniciantes - O seu Primeiro Projeto de Regressão Linear em Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML para iniciantes - O seu Primeiro Projeto de Regressão Linear em Python")

> 🎥 Clique na imagem acima para um vídeo breve a trabalhar este exercício.

No ficheiro _notebook.ipynb_ associado a esta lição, limpe todas as células pressionando o ícone do 'balde de lixo'.

Nesta secção, vai trabalhar com um pequeno conjunto de dados sobre diabetes integrado no Scikit-learn para fins de aprendizagem. Imagine que queria testar um tratamento para pacientes diabéticos. Modelos de Machine Learning podem ajudá-lo a determinar quais os pacientes que responderiam melhor ao tratamento, com base em combinações de variáveis. Mesmo um modelo de regressão muito básico, visualizado, pode mostrar informações sobre variáveis que o ajudariam a organizar os seus ensaios clínicos teóricos.

✅ Existem vários tipos de métodos de regressão, e qual escolher depende da resposta que procura. Se quiser prever a altura provável de uma pessoa para uma dada idade, use regressão linear, pois procura um **valor numérico**. Se estiver interessado em descobrir se um tipo de cozinha deve ser considerado vegan ou não, procura uma **atribuição de categoria**, por isso usaria regressão logística. Vai aprender mais sobre regressão logística mais tarde. Pense um pouco em algumas perguntas que pode fazer aos dados, e qual destes métodos seria mais apropriado.

Vamos começar esta tarefa.

### Importar bibliotecas

Para esta tarefa, vamos importar algumas bibliotecas:

- **matplotlib**. É uma [ferramenta de gráficos](https://matplotlib.org/) útil e vamos usá-la para criar um gráfico de linhas.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) é uma biblioteca útil para manipular dados numéricos em Python.
- **sklearn**. Esta é a biblioteca [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importe algumas bibliotecas para ajudar nas suas tarefas.

1. Adicione as importações escrevendo o seguinte código:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Acima está a importar `matplotlib`, `numpy` e está a importar `datasets`, `linear_model` e `model_selection` de `sklearn`. `model_selection` é usado para dividir dados em conjuntos de treino e teste.

### O conjunto de dados diabetes

O [conjunto de dados diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) incorporado inclui 442 amostras de dados sobre diabetes, com 10 variáveis características, algumas das quais incluem:

- idade: idade em anos
- IMC: índice de massa corporal
- pressão arterial média
- s1 tc: Células T (um tipo de glóbulos brancos)

✅ Este conjunto de dados inclui o conceito de 'sexo' como uma variável característica importante para investigação em diabetes. Muitos conjuntos de dados médicos incluem este tipo de classificação binária. Pense um pouco em como categorizações como esta podem excluir certas partes da população de tratamentos.

Agora, carregue os dados X e y.

> 🎓 Lembre-se, este é um aprendizado supervisionado, e precisamos de um alvo 'y' nomeado.

Numa nova célula de código, carregue o conjunto de dados diabetes chamando `load_diabetes()`. A entrada `return_X_y=True` indica que `X` será uma matriz de dados e `y` será o alvo de regressão.

1. Adicione alguns comandos print para mostrar a forma da matriz de dados e o seu primeiro elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    O que está a receber de volta como resposta é uma tupla. O que está a fazer é atribuir os dois primeiros valores da tupla a `X` e `y`, respetivamente. Saiba mais [sobre tuplas](https://wikipedia.org/wiki/Tuple).

    Pode ver que estes dados têm 442 itens estruturados em arrays de 10 elementos:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pense um pouco na relação entre os dados e o alvo da regressão. A regressão linear prevê relações entre a característica X e a variável alvo y. Consegue encontrar o [alvo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) para o conjunto de dados diabetes na documentação? O que este conjunto de dados demonstra, dado esse alvo?

2. A seguir, selecione uma parte deste conjunto de dados para plotar selecionando a 3ª coluna do conjunto. Pode fazer isto usando o operador `:` para selecionar todas as linhas, e depois selecionar a 3ª coluna usando o índice (2). Também pode remodelar os dados para uma matriz 2D - como exigido para plotar - usando `reshape(n_linhas, n_colunas)`. Se um dos parâmetros for -1, a dimensão correspondente é calculada automaticamente.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Em qualquer momento, imprima os dados para verificar a sua forma.

3. Agora que tem dados prontos para plotar, pode verificar se uma máquina pode ajudar a determinar uma divisão lógica entre os números deste conjunto de dados. Para isso, precisa de dividir tanto os dados (X) como o alvo (y) em conjuntos de teste e treino. O Scikit-learn tem uma forma simples de fazer isso; pode dividir os seus dados teste num ponto dado.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Agora está pronto para treinar o seu modelo! Carregue o modelo de regressão linear e treine-o com os seus conjuntos de treino X e y usando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` é uma função que verá em muitas bibliotecas ML, como o TensorFlow

5. Depois, crie uma previsão usando dados de teste, usando a função `predict()`. Esta será usada para desenhar a linha entre os grupos de dados

    ```python
    y_pred = model.predict(X_test)
    ```

6. Agora é altura de mostrar os dados num gráfico. O Matplotlib é uma ferramenta muito útil para esta tarefa. Crie um gráfico de dispersão de todos os dados de teste X e y, e use a previsão para desenhar uma linha no local mais apropriado, entre as agrupações dos dados do modelo.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![um gráfico de dispersão mostrando pontos de dados sobre diabetes](../../../../translated_images/pt-PT/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Pense um pouco sobre o que está a acontecer aqui. Uma linha reta atravessa muitos pequenos pontos de dados, mas o que está a fazer exatamente? Consegue ver como deve usar esta linha para prever onde um novo ponto de dados, não visto antes, deve encaixar em relação ao eixo y do gráfico? Tente pôr em palavras a utilização prática deste modelo.

Parabéns, construiu o seu primeiro modelo de regressão linear, criou uma previsão com ele e exibiu-o num gráfico!

---
## 🚀Desafio

Plote uma variável diferente deste conjunto de dados. Dica: edite esta linha: `X = X[:,2]`. Dado o alvo deste conjunto de dados, o que consegue descobrir sobre a progressão da diabetes enquanto doença?
## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Neste tutorial, trabalhou com regressão linear simples, em vez de regressão linear univariada ou múltipla. Leia um pouco sobre as diferenças entre estes métodos, ou veja [este vídeo](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Leia mais sobre o conceito de regressão e pense em que tipos de questões podem ser respondidas por esta técnica. Faça este [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) para aprofundar a sua compreensão.

## Tarefa

[Um conjunto de dados diferente](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes da utilização desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->