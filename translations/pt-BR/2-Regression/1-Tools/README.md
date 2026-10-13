# Comece com Python e Scikit-learn para modelos de regressão

![Resumo de regressões em um sketchnote](../../../../translated_images/pt-BR/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introdução

Nessas quatro aulas, você irá descobrir como construir modelos de regressão. Em breve discutiremos para que eles servem. Mas antes de fazer qualquer coisa, certifique-se de ter as ferramentas certas para iniciar o processo!

Nesta aula, você vai aprender a:

- Configurar seu computador para tarefas locais de machine learning.
- Trabalhar com Jupyter Notebooks.
- Usar Scikit-learn, incluindo a instalação.
- Explorar regressão linear com um exercício prático.

## Instalações e configurações

[![ML para iniciantes - Prepare suas ferramentas para construir modelos de Machine Learning](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML para iniciantes - Prepare suas ferramentas para construir modelos de Machine Learning")

> 🎥 Clique na imagem acima para um vídeo curto explicando como configurar seu computador para ML.

1. **Instale o Python**. Garanta que o [Python](https://www.python.org/downloads/) esteja instalado no seu computador. Você usará Python para muitas tarefas de ciência de dados e machine learning. A maioria dos sistemas já inclui uma instalação do Python. Também existem úteis [Pacotes de Codificação em Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) disponíveis para facilitar a configuração para alguns usuários.

   Alguns usos do Python, no entanto, requerem uma versão do software, enquanto outros requerem uma versão diferente. Por isso, é útil trabalhar dentro de um [ambiente virtual](https://docs.python.org/3/library/venv.html).

2. **Instale o Visual Studio Code**. Certifique-se de que o Visual Studio Code está instalado no seu computador. Siga estas instruções para [instalar o Visual Studio Code](https://code.visualstudio.com/) na instalação básica. Você vai usar Python no Visual Studio Code neste curso, então pode querer se familiarizar com como [configurar o Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) para desenvolvimento em Python.

> Fique confortável com Python passando por esta coleção de [módulos Learn](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
>
> [![Configuração do Python com Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configuração do Python com Visual Studio Code")
>
> 🎥 Clique na imagem acima para um vídeo: usando Python dentro do VS Code.

3. **Instale o Scikit-learn**, seguindo [estas instruções](https://scikit-learn.org/stable/install.html). Como é necessário garantir o uso do Python 3, é recomendado usar um ambiente virtual. Note que, se você está instalando essa biblioteca em um Mac M1, há instruções especiais na página mencionada acima.

1. **Instale o Jupyter Notebook**. Você precisará [instalar o pacote Jupyter](https://pypi.org/project/jupyter/).

## Seu ambiente para desenvolvimento de ML

Você vai usar **notebooks** para desenvolver seu código Python e criar modelos de machine learning. Esse tipo de arquivo é uma ferramenta comum para cientistas de dados, e eles podem ser identificados pela sua extensão `.ipynb`.

Notebooks são um ambiente interativo que permite ao desenvolvedor tanto codificar quanto adicionar notas e escrever documentações ao redor do código, o que é bastante útil para projetos experimentais ou orientados à pesquisa.

[![ML para iniciantes - Configure Jupyter Notebooks para iniciar a construção de modelos de regressão](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML para iniciantes - Configure Jupyter Notebooks para iniciar a construção de modelos de regressão")

> 🎥 Clique na imagem acima para um vídeo curto explicando este exercício.

### Exercício - trabalhe com um notebook

Nesta pasta, você encontrará o arquivo _notebook.ipynb_.

1. Abra _notebook.ipynb_ no Visual Studio Code.

   Um servidor Jupyter será iniciado com Python 3+. Você encontrará áreas do notebook que podem ser `executadas` — partes de código. Você pode executar um bloco de código selecionando o ícone que parece um botão de play.

1. Selecione o ícone `md` e adicione um pouco de markdown, e o texto seguinte **# Bem-vindo ao seu notebook**.

   Em seguida, adicione algum código Python.

1. Digite **print('hello notebook')** no bloco de código.
1. Selecione a seta para executar o código.

   Você deverá ver a declaração impressa:

    ```output
    hello notebook
    ```

![VS Code com notebook aberto](../../../../translated_images/pt-BR/notebook.4a3ee31f396b8832.webp)

Você pode intercalar seu código com comentários para auto-documentar o notebook.

✅ Pense por um minuto em como é diferente o ambiente de trabalho de um desenvolvedor web versus o de um cientista de dados.

## Configurando e usando Scikit-learn

Agora que o Python está configurado no seu ambiente local, e você está confortável com Jupyter Notebooks, vamos ficar igualmente à vontade com o Scikit-learn (pronuncia-se `sci` como em `science`). O Scikit-learn oferece uma [API extensa](https://scikit-learn.org/stable/modules/classes.html#api-ref) para ajudar você a realizar tarefas de ML.

Segundo o [site oficial](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn é uma biblioteca de machine learning de código aberto que suporta aprendizado supervisionado e não supervisionado. Também fornece várias ferramentas para ajuste de modelos, pré-processamento de dados, seleção e avaliação de modelos, além de muitas outras utilidades."

Neste curso, você usará o Scikit-learn e outras ferramentas para construir modelos de machine learning para realizar o que chamamos de tarefas de 'machine learning tradicional'. Deliberadamente evitamos redes neurais e deep learning, pois esses temas são melhor abordados em nosso futuro currículo 'IA para iniciantes'.

O Scikit-learn facilita a construção e avaliação de modelos para uso. Ele é focado principalmente em dados numéricos e contém diversos conjuntos de dados prontos para usar como ferramentas de aprendizado. Inclui também modelos pré-construídos para os estudantes experimentarem. Vamos explorar o processo de carregar dados predefinidos e usar um estimador embutido — o primeiro modelo de ML com Scikit-learn — com dados básicos.

## Exercício - seu primeiro notebook Scikit-learn

> Este tutorial foi inspirado pelo [exemplo de regressão linear](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) no site do Scikit-learn.


[![ML para iniciantes - Seu Primeiro Projeto de Regressão Linear em Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML para iniciantes - Seu Primeiro Projeto de Regressão Linear em Python")

> 🎥 Clique na imagem acima para um vídeo curto explicando este exercício.

No arquivo _notebook.ipynb_ associado a esta aula, limpe todas as células pressionando o ícone da 'lixeira'.

Nesta seção, você trabalhará com um pequeno conjunto de dados sobre diabetes, embutido no Scikit-learn para fins de aprendizado. Imagine que você queira testar um tratamento para pacientes diabéticos. Modelos de Machine Learning podem ajudar a determinar quais pacientes responderiam melhor ao tratamento, com base em combinações de variáveis. Mesmo um modelo de regressão bastante básico, quando visualizado, pode mostrar informações sobre variáveis que ajudariam a organizar seus supostos ensaios clínicos.

✅ Existem muitos tipos de métodos de regressão, e qual você escolhe depende da resposta que procura. Se quiser prever a altura provável de uma pessoa de determinada idade, usará regressão linear, já que está buscando um **valor numérico**. Se estiver interessado em descobrir se um tipo de comida deve ser considerada vegana ou não, estará procurando uma **atribuição de categoria**, então usaria regressão logística. Você aprenderá mais sobre regressão logística mais adiante. Pense um pouco sobre algumas perguntas que você pode fazer aos dados e qual desses métodos seria mais apropriado.

Vamos começar essa tarefa.

### Importar bibliotecas

Para esta tarefa, vamos importar algumas bibliotecas:

- **matplotlib**. É uma útil [ferramenta de gráficos](https://matplotlib.org/) e vamos usá-la para criar um gráfico de linha.
- **numpy**. O [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) é uma biblioteca útil para manipular dados numéricos em Python.
- **sklearn**. Esta é a biblioteca [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importe algumas bibliotecas para ajudar nas suas tarefas.

1. Adicione as importações digitando o seguinte código:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Acima você está importando `matplotlib`, `numpy` e importando `datasets`, `linear_model` e `model_selection` do `sklearn`. `model_selection` é usado para dividir dados em conjuntos de treino e teste.

### O conjunto de dados de diabetes

O [conjunto de dados diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) incluído possui 442 amostras relacionadas ao diabetes, com 10 variáveis características, algumas das quais incluem:

- age: idade em anos
- bmi: índice de massa corporal
- bp: pressão arterial média
- s1 tc: Células T (tipo de glóbulos brancos)

✅ Este conjunto de dados inclui o conceito de 'sexo' como uma variável característica importante para pesquisas sobre diabetes. Muitos conjuntos médicos incluem esse tipo de classificação binária. Pense um pouco sobre como categorias como essa podem excluir certas partes da população de tratamentos.

Agora, carregue os dados X e y.

> 🎓 Lembre-se, este é aprendizado supervisionado, e precisamos de um alvo chamado 'y'.

Em uma nova célula de código, carregue o conjunto de dados diabetes chamando `load_diabetes()`. O parâmetro `return_X_y=True` indica que `X` será uma matriz de dados, e `y` será o alvo da regressão.

1. Adicione alguns comandos print para mostrar a forma da matriz de dados e seu primeiro elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    O que você está recebendo como resposta é uma tupla. O que você está fazendo é atribuir os dois primeiros valores da tupla para `X` e `y` respectivamente. Saiba mais [sobre tuplas](https://wikipedia.org/wiki/Tuple).

    Você pode ver que esses dados têm 442 itens organizados em arrays de 10 elementos:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pense um pouco sobre a relação entre os dados e o alvo da regressão. A regressão linear prevê relações entre a característica X e a variável alvo y. Você consegue encontrar o [alvo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) no conjunto diabetes na documentação? O que esse conjunto demonstra, dado esse alvo?

2. Em seguida, selecione uma parte desse conjunto para plotar, escolhendo a 3ª coluna do conjunto. Você pode usar o operador `:` para selecionar todas as linhas, e então selecione a terceira coluna pelo índice (2). Você também pode remodelar os dados para um array 2D — conforme exigido para plotagem — usando `reshape(n_linhas, n_colunas)`. Se um dos parâmetros for -1, a dimensão correspondente será calculada automaticamente.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ A qualquer momento, imprima os dados para checar sua forma.

3. Agora que você tem os dados prontos para plotar, veja se uma máquina pode ajudar a determinar uma divisão lógica entre os números deste conjunto. Para isso, você precisa dividir tanto os dados (X) quanto o alvo (y) em conjuntos de teste e treino. O Scikit-learn tem uma forma simples de fazer isso; você pode dividir seus dados de teste em um ponto dado.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Agora você está pronto para treinar seu modelo! Carregue o modelo de regressão linear e treine-o com seus conjuntos X e y de treino usando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` é uma função que você verá em muitas bibliotecas de ML como TensorFlow

5. Depois, crie uma predição usando os dados de teste, usando a função `predict()`. Isso será usado para desenhar uma linha entre os grupos de dados.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Agora é hora de mostrar os dados em um gráfico. Matplotlib é uma ferramenta muito útil para essa tarefa. Crie um gráfico de dispersão de todos os dados X e y do teste, e use a predição para desenhar uma linha no lugar mais apropriado, entre os agrupamentos de dados do modelo.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![um gráfico de dispersão mostrando pontos de dados sobre diabetes](../../../../translated_images/pt-BR/scatterplot.ad8b356bcbb33be6.webp)


   ✅ Pense um pouco sobre o que está acontecendo aqui. Uma linha reta está passando por muitos pequenos pontos de dados, mas o que exatamente ela está fazendo? Você consegue ver como deveria poder usar essa linha para prever onde um novo ponto de dados, ainda não visto, deve encaixar em relação ao eixo y do gráfico? Tente colocar em palavras a utilidade prática desse modelo.

Parabéns, você construiu seu primeiro modelo de regressão linear, criou uma previsão com ele e a exibiu em um gráfico!

---
## 🚀Desafio

Plote uma variável diferente desse conjunto de dados. Dica: edite esta linha: `X = X[:,2]`. Considerando o alvo deste conjunto de dados, o que você consegue descobrir sobre a progressão do diabetes como doença?
## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Neste tutorial, você trabalhou com regressão linear simples, ao invés de regressão linear univariada ou múltipla. Leia um pouco sobre as diferenças entre esses métodos, ou dê uma olhada neste [vídeo](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Leia mais sobre o conceito de regressão e pense sobre que tipos de perguntas podem ser respondidas por essa técnica. Faça este [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) para aprofundar sua compreensão.

## Tarefa

[Um conjunto de dados diferente](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido usando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, por favor, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->