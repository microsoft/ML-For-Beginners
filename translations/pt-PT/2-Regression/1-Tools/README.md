# Comece com Python e Scikit-learn para modelos de regressão

![Sumário das regressões num sketchnote](../../../../translated_images/pt-PT/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introdução

Nestes quatro lições, você vai descobrir como construir modelos de regressão. Falaremos em breve para que servem estes modelos. Mas antes de fazer qualquer coisa, certifique-se de que tem as ferramentas certas instaladas para começar o processo!

Nesta lição, vai aprender a:

- Configurar o seu computador para tarefas locais de machine learning.
- Trabalhar com Jupyter Notebooks.
- Usar Scikit-learn, incluindo a instalação.
- Explorar regressão linear com um exercício prático.

## Instalações e configurações

[![ML para iniciantes - Configure as suas ferramentas para construir modelos de Machine Learning](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML para iniciantes - Configure as suas ferramentas para construir modelos de Machine Learning")

> 🎥 Clique na imagem acima para um vídeo curto a mostrar como configurar o seu computador para ML.

1. **Instale o Python**. Certifique-se que o [Python](https://www.python.org/downloads/) está instalado no seu computador. Você vai usar Python para muitas tarefas de ciência de dados e machine learning. A maioria dos sistemas informáticos já inclui uma instalação de Python. Existem [Pacotes de Código Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) úteis disponíveis, para facilitar a configuração a alguns utilizadores.

   Alguns usos do Python, no entanto, requerem uma versão do software, enquanto outros requerem uma versão diferente. Por isso, é útil trabalhar num [ambiente virtual](https://docs.python.org/3/library/venv.html).

2. **Instale o Visual Studio Code**. Certifique-se que tem o Visual Studio Code instalado no seu computador. Siga estas instruções para [instalar o Visual Studio Code](https://code.visualstudio.com/) para uma instalação básica. Vai usar Python no Visual Studio Code neste curso, por isso pode querer rever como [configurar o Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) para desenvolvimento Python.

   > Fique confortável com Python trabalhando nestas coleções de [módulos Learn](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Configurar Python com Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configurar Python com Visual Studio Code")
   >
   > 🎥 Clique na imagem acima para um vídeo: usar Python dentro do VS Code.

3. **Instale o Scikit-learn**, seguindo [estas instruções](https://scikit-learn.org/stable/install.html). Como precisa garantir que usa o Python 3, é recomendado usar um ambiente virtual. Nota, se estiver a instalar esta biblioteca num Mac M1, há instruções especiais na página indicada acima.

1. **Instale o Jupyter Notebook**. Vai precisar de [instalar o pacote Jupyter](https://pypi.org/project/jupyter/).

## O seu ambiente de criação ML

Vai usar **notebooks** para desenvolver o seu código Python e criar modelos de machine learning. Este tipo de ficheiro é uma ferramenta comum para cientistas de dados, e podem ser identificados pela extensão `.ipynb`.

Os notebooks são um ambiente interativo que permite ao programador tanto codificar como adicionar notas e escrever documentação em torno do código, o que é bastante útil para projetos experimentais ou orientados para investigação.

[![ML para iniciantes - Configure Jupyter Notebooks para começar a construir modelos de regressão](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML para iniciantes - Configure Jupyter Notebooks para começar a construir modelos de regressão")

> 🎥 Clique na imagem acima para um vídeo curto a mostrar este exercício.

### Exercício - trabalhar com um notebook

Nesta pasta, vai encontrar o ficheiro _notebook.ipynb_.

1. Abra _notebook.ipynb_ no Visual Studio Code.

   Um servidor Jupyter vai arrancar com Python 3+ iniciado. Vai encontrar áreas do notebook que podem ser `executadas`, blocos de código. Pode executar um bloco de código selecionando o ícone que parece um botão de play.

1. Selecione o ícone `md` e adicione um pouco de markdown, e o seguinte texto **# Bem-vindo ao seu notebook**.

   A seguir, adicione algum código Python.

1. Escreva **print('hello notebook')** no bloco de código.
1. Selecione a seta para executar o código.

   Deve ver a declaração impressa:

    ```output
    hello notebook
    ```

![VS Code com um notebook aberto](../../../../translated_images/pt-PT/notebook.4a3ee31f396b8832.webp)

Pode intercalar o seu código com comentários para auto-documentar o notebook.

✅ Pense por um minuto na diferença entre o ambiente de trabalho de um programador web e de um cientista de dados.

## A funcionar com Scikit-learn

Agora que o Python está configurado no seu ambiente local, e que está confortável com Jupyter Notebooks, vamos ficar igualmente confortáveis com o Scikit-learn (pronuncie `sci` como em `science`). O Scikit-learn fornece uma [API extensa](https://scikit-learn.org/stable/modules/classes.html#api-ref) para ajudar a realizar tarefas de ML.

Segundo o seu [website](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn é uma biblioteca open source de machine learning que suporta aprendizado supervisionado e não supervisionado. Fornece também várias ferramentas para ajuste de modelos, pré-processamento de dados, seleção e avaliação de modelos, e muitas outras utilidades."

Neste curso, vai usar Scikit-learn e outras ferramentas para construir modelos de machine learning para realizar o que chamamos tarefas de 'machine learning tradicional'. Evitámos de propósito redes neurais e deep learning, pois são melhor abordados no nosso próximo currículo 'AI para Iniciantes'.

O Scikit-learn torna simples construir modelos e avaliá-los para uso. Está focado principalmente no uso de dados numéricos e contém vários datasets prontos para uso como ferramentas de aprendizagem. Inclui ainda modelos pré-construídos para os estudantes experimentarem. Vamos explorar o processo de carregar dados pré-embalados e usar um estimador incorporado, primeiro modelo ML com Scikit-learn com alguns dados básicos.

## Exercício - o seu primeiro notebook Scikit-learn

> Este tutorial foi inspirado no [exemplo de regressão linear](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) no site do Scikit-learn.


[![ML para iniciantes - O seu Primeiro Projeto de Regressão Linear em Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML para iniciantes - O seu Primeiro Projeto de Regressão Linear em Python")

> 🎥 Clique na imagem acima para um vídeo curto a mostrar este exercício.

No ficheiro _notebook.ipynb_ associado a esta lição, limpe todas as células clicando no ícone da 'lixeira'.

Nesta secção, vai trabalhar com um pequeno conjunto de dados sobre diabetes que está incorporado no Scikit-learn para fins didáticos. Imagine que quer testar um tratamento para pacientes diabéticos. Modelos de Machine Learning podem ajudar a determinar quais os pacientes que responderiam melhor ao tratamento, com base em combinações de variáveis. Mesmo um modelo muito básico de regressão, quando visualizado, pode mostrar informações sobre variáveis que o ajudariam a organizar os seus ensaios clínicos teóricos.

✅ Existem muitos tipos de métodos de regressão, e qual escolher depende da resposta que procura. Se quiser prever a altura provável de uma pessoa numa dada idade, usaria regressão linear, pois busca um **valor numérico**. Se estiver interessado em descobrir se um tipo de culinária deve ser considerado vegan ou não, está a procurar uma **atribuição de categoria**, pelo que usaria regressão logística. Vai aprender mais sobre regressão logística mais tarde. Pense um pouco em algumas perguntas que pode fazer sobre dados, e qual destes métodos seria mais apropriado.

Vamos começar esta tarefa.

### Importar bibliotecas

Para esta tarefa vamos importar algumas bibliotecas:

- **matplotlib**. É uma útil [ferramenta de gráficos](https://matplotlib.org/) e vamos usá-la para criar um gráfico de linhas.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) é uma biblioteca útil para manipular dados numéricos em Python.
- **sklearn**. Esta é a biblioteca [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importe algumas bibliotecas para ajudar nas suas tarefas.

1. Adicione as importações digitando o seguinte código:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Aqui está a importar `matplotlib`, `numpy` e está a importar `datasets`, `linear_model` e `model_selection` do `sklearn`. `model_selection` é usado para dividir os dados em conjuntos de treino e teste.

### O conjunto de dados de diabetes

O [conjunto de dados diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) incluído tem 442 amostras de dados sobre diabetes, com 10 variáveis características, algumas das quais incluem:

- idade: idade em anos
- bmi: índice de massa corporal
- bp: pressão arterial média
- s1 tc: células T (um tipo de glóbulos brancos)

✅ Este conjunto de dados inclui o conceito de 'sexo' como uma variável característica importante para pesquisas sobre diabetes. Muitos conjuntos de dados médicos incluem este tipo de classificação binária. Pense um pouco em como categorias como esta podem excluir certas partes da população de tratamentos.

Agora, carregue os dados X e y.

> 🎓 Lembre-se, este é aprendizado supervisionado, e precisamos de um alvo nomeado 'y'.

Numa nova célula de código, carregue o conjunto de dados diabetes chamando `load_diabetes()`. A entrada `return_X_y=True` indica que `X` será uma matriz de dados, e `y` será o alvo de regressão.

1. Acrescente alguns comandos print para mostrar a forma da matriz de dados e o seu primeiro elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    O que está a obter como resposta é uma tupla. O que está a fazer é atribuir os dois primeiros valores da tupla a `X` e `y` respetivamente. Saiba mais [sobre tuplas](https://wikipedia.org/wiki/Tuple).

    Pode ver que este conjunto tem 442 itens organizados em arrays de 10 elementos:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pense um pouco na relação entre os dados e o alvo da regressão. A regressão linear prevê relações entre a característica X e a variável alvo y. Consegue encontrar o [alvo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) para o conjunto de dados diabetes na documentação? O que este conjunto demonstra, dado esse alvo?

2. A seguir, selecione uma parte deste conjunto para representar num gráfico, escolhendo a 3ª coluna do dataset. Pode fazer isto usando o operador `:` para selecionar todas as linhas, e depois a 3ª coluna usando o índice (2). Pode também mudar a forma dos dados para um array 2D - como exigido para graficar - usando `reshape(n_linhas, n_colunas)`. Se um dos parâmetros for -1, a dimensão correspondente é calculada automaticamente.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Em qualquer altura, imprima os dados para verificar a sua forma.

3. Agora que tem dados prontos para graficar, pode ver se uma máquina pode ajudar a determinar uma divisão lógica entre os números neste conjunto. Para isso, precisa dividir os dados (X) e o alvo (y) em conjuntos de treino e teste. Scikit-learn tem uma forma simples de fazer isto; pode dividir os seus dados de teste num dado ponto.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Agora está pronto para treinar o seu modelo! Carregue o modelo de regressão linear e treine-o com os seus conjuntos X e y de treino usando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` é uma função que verá em muitas bibliotecas ML como TensorFlow

5. Depois, crie uma previsão usando os dados de teste, usando a função `predict()`. Isto será usado para traçar a linha entre grupos de dados

    ```python
    y_pred = model.predict(X_test)
    ```

6. Agora é hora de mostrar os dados num gráfico. Matplotlib é uma ferramenta muito útil para esta tarefa. Crie um scatterplot com todos os dados X e y do teste, e use a previsão para desenhar uma linha na posição mais adequada, entre os grupos de dados do modelo.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![um scatterplot a mostrar pontos de dados sobre diabetes](../../../../translated_images/pt-PT/scatterplot.ad8b356bcbb33be6.webp)


   ✅ Pense um pouco sobre o que está a acontecer aqui. Uma linha reta está a passar por muitos pequenos pontos de dados, mas o que está ela a fazer exatamente? Consegue ver como deve ser capaz de usar esta linha para prever onde um novo ponto de dados, nunca antes visto, deve encaixar em relação ao eixo y do gráfico? Tente pôr em palavras a utilidade prática deste modelo.

Parabéns, construiu o seu primeiro modelo de regressão linear, criou uma previsão com ele e apresentou-a num gráfico!

---
## 🚀Desafio

Plote uma variável diferente deste conjunto de dados. Dica: edite esta linha: `X = X[:,2]`. Dado o alvo deste conjunto de dados, o que consegue descobrir sobre a progressão da diabetes como doença?
## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Neste tutorial, trabalhou com regressão linear simples, em vez de regressão linear univariada ou múltipla. Leia um pouco sobre as diferenças entre estes métodos, ou veja [este vídeo](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Leia mais sobre o conceito de regressão e pense que tipos de questões podem ser respondidas por esta técnica. Faça este [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) para aprofundar o seu entendimento.

## Exercício

[Um conjunto de dados diferente](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes da utilização desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->