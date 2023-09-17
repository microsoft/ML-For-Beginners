# Crie um aplicativo Web para usar um modelo de ML

Nesta lição, você treinará um modelo de ML em um conjunto de dados que está fora deste mundo: _avistamentos de OVNIs no século passado_, obtidos do banco de dados do NUFORC.

Você vai aprender:

- Como 'pickle' um modelo treinado
- Como usar esse modelo em uma aplicação Flask

Continuaremos nosso uso de notebooks para limpar dados e treinar nosso modelo, mas você pode levar o processo um passo adiante, explorando o uso de um modelo 'em estado selvagem', por assim dizer: em um aplicativo web.

Para fazer isso, você precisa construir um aplicativo da web usando o Flask.

## [Teste pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17?loc=ptbr)

## Construindo um aplicativo

Existem inúmeras maneiras de criar aplicativos web para consumir modelos de machine learning (aprendizado de máquina). Sua arquitetura web pode influenciar a maneira como seu modelo é treinado. Imagine que você está trabalhando em uma empresa em que o grupo de ciência de dados treinou um modelo que eles desejam que você use em um aplicativo.

### Considerações

Existem muitas perguntas que você precisa fazer:

- **É um aplicativo web ou um aplicativo mobile?** Se você estiver criando um aplicativo mobile ou precisar usar o modelo em um contexto de IoT, poderá usar o [TensorFlow Lite](https://www.tensorflow.org/lite/) e usar o modelo em um aplicativo Android ou iOS.
- **Onde o modelo residirá?** Na nuvem ou localmente?
- **Suporte offline.** O aplicativo precisa funcionar offline??
- **Qual tecnologia foi usada para treinar o modelo?** A tecnologia escolhida pode influenciar o ferramental que você precisa usar.
    - **Usando o fluxo do Tensor.** Se você estiver treinando um modelo usando o TensorFlow, por exemplo, esse ecossistema oferece a capacidade de converter um modelo do TensorFlow para uso em um aplicativo da web usando [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Usando o PyTorch.** Se você estiver construindo um modelo usando uma biblioteca como [PyTorch](https://pytorch.org/), você tem a opção de exportá-lo em formato [ONNX](https://onnx.ai/) (Troca de rede neural aberta (Open Neural Network Exchange)) para uso em aplicativos web JavaScript que podem usar o [Onnx Runtime](https://www.onnxruntime.ai/). Esta opção será explorada em uma lição futura para um modelo treinado para aprender com Scikit.
    - **Usando Lobe.ai ou Azure Custom Vision.** Se você estiver usando um sistema ML SaaS (Software as a Service), como [Lobe.ai](https://lobe.ai/) ou [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) para treinar um modelo, este tipo de software fornece maneiras de exportar o modelo para muitas plataformas, incluindo a construção de uma API sob medida para ser consultada na nuvem por seu aplicativo online.

Você também tem a oportunidade de construir um aplicativo web Flask inteiro que seria capaz de treinar o próprio modelo em um navegador da web. Isso também pode ser feito usando TensorFlow.js em um contexto JavaScript.

Para nossos propósitos, já que estamos trabalhando com notebooks baseados em Python, vamos explorar as etapas que você precisa seguir para exportar um modelo treinado de tal notebook para um formato legível por um aplicativo web construído em Python.

## Ferramenta

Para esta tarefa, você precisa de duas ferramentas: Flask e Pickle, ambos executados em Python.

✅ O que é [Flask](https://palletsprojects.com/p/flask/)? Definido como um 'micro-framework' por seus criadores, o Flask fornece os recursos básicos de estruturas web usando Python e um mecanismo de modelagem para construir páginas web. Dê uma olhada [neste módulo de aprendizagem](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para praticar a construção com Flask.

✅ O que é [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 é um módulo Python que serializa e desserializa a estrutura de um objeto Python. Quando você 'pickle' um modelo, serializa ou aplaina sua estrutura para uso na web. Tenha cuidado: pickle não é intrinsecamente seguro, então tome cuidado se for solicitado para ser feito um 'un-pickle' em um arquivo. Um arquivo tem o sufixo `.pkl`.

## Exercício - limpe seus dados

Nesta lição, você usará dados de 80.000 avistamentos de OVNIs, coletados pelo [NUFORC](https://nuforc.org) (Centro Nacional de Relatos de OVNIs). Esses dados têm algumas descrições interessantes de avistamentos de OVNIs, por exemplo:

- **Exemplo de descrição longa.** "Um homem emerge de um feixe de luz que brilha em um campo gramado à noite e corre em direção ao estacionamento da Texas Instruments".
- **Exemplo de descrição curta.** "as luzes nos perseguiram".

A planilha [ufos.csv](../data/ufos.csv) inclui colunas sobre a `city`, `state` e `country` onde o avistamento ocorreu, a `shape` do objeto e sua `latitude` e `longitude`.
_nota da tradução: city é a coluna referente a cidade, state é a coluna referente ao estado e country é a coluna referente ao país._

Em um [notebook](../notebook.ipynb) branco incluído nesta lição:

1. importe as bibliotecas `pandas`, `matplotlib`, e `numpy` como você fez nas lições anteriores e importe a planilha ufos. Você pode dar uma olhada em um conjunto de dados de amostra:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

2. Converta os dados ufos em um pequeno dataframe com títulos novos. Verifique os valores únicos no campo  `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

3. Agora, você pode reduzir a quantidade de dados com os quais precisamos lidar, descartando quaisquer valores nulos e importando apenas avistamentos entre 1 a 60 segundos:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

4. Importe a biblioteca `LabelEncoder` do Scikit-learn para converter os valores de texto de países em um número:

    ✅ LabelEncoder encodes data alphabetically

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Seus dados devem ser assim:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Exercício - construa seu modelo

Agora você pode se preparar para treinar um modelo, dividindo os dados no grupo de treinamento e teste.

1. Selecione os três recursos que deseja treinar como seu vetor X, e o vetor y será o `Country`. Você quer ser capaz de inserir `Seconds`, `Latitude` e `Longitude` e obter um id de país para retornar.

_nota da tradução: seconds são os segundos e country são os países._

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

2. Treine seu modelo usando regressão logística:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

A precisão não é ruim **(cerca de 95%)**, o que não é surpresa, já que `Country` e `Latitude/Longitude` se correlacionam.

O modelo que você criou não é muito revolucionário, pois você deve ser capaz de inferir um `País` de sua `Latitude` e `Longitude`, mas é um bom exercício tentar treinar a partir de dados brutos que você limpou, exportou e, em seguida, use este modelo em um aplicativo da web.

## Exercício - 'pickle' seu modelo

Agora, é hora de _pickle_ seu modelo! Você pode fazer isso em algumas linhas de código. Depois de _pickled_, carregue seu modelo pickled e teste-o em uma matriz de dados de amostra contendo valores para segundos, latitude e longitude,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

O modelo retorna **'3'**, que é o código do país para o Reino Unido. Maneiro! 👽

## Exercício - construir um aplicativo Flask

Agora você pode construir uma aplicação Flask para chamar seu modelo e retornar resultados semelhantes, mas de uma forma visualmente mais agradável.

1. Comece criando uma pasta chamada **web-app** ao lado do arquivo _notebook.ipynb_ onde o arquivo _ufo-model.pkl_ reside.

2. Nessa pasta, crie mais três pastas: **static**, com uma pasta **css** dentro dela, e **templates**. Agora você deve ter os seguintes arquivos e diretórios::

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulte a pasta da solução para uma visão da aplicação concluído

3. O primeiro arquivo a ser criado na pasta _web-app_ é o arquivo **requirements.txt**. Como _package.json_ em uma aplicação JavaScript, este arquivo lista as dependências exigidas pela aplicação. Em **requirements.txt**, adicione as linhas:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

4. Agora, execute este arquivo navegando até o _web-app_:

    ```bash
    cd web-app
    ```

5. Em seu terminal, digite `pip install`, para instalar as bibliotecas listadas em _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

6. Agora, você está pronto para criar mais três arquivos para finalizar o aplicativo:

    1. Crie **app.py** na raiz do projeto.
    2. Crie **index.html** no diretório _templates_.
    3. Crie **styles.css** no diretório _static/css_.

7. Construa o arquivo _styles.css_ com alguns estilos:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

8. Em seguida, crie o arquivo _index.html_:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 Predição de aparência de OVNIs! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>De acordo com o número de segundos, latitude e longitude, que país provavelmente relatou ter visto um OVNI?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Preveja o país onde o OVNI vai ser visto</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    Dê uma olhada no modelo neste arquivo. Observe a sintaxe do 'mustache' em torno das variáveis que serão fornecidas pelo aplicativo, como o texto de previsão: `{{}}`. Há também um formulário que posta uma previsão para a rota `/predict`.

    Finalmente, você está pronto para construir o arquivo python que direciona o consumo do modelo e a exibição de previsões:

9. Em `app.py` adicione:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 Dica: quando você adiciona [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) enquanto executa o aplicativo da web usando o Flask, todas as alterações feitas em seu aplicativo será refletido imediatamente, sem a necessidade de reiniciar o servidor. Cuidado! Não ative este modo em um aplicativo de produção.

Se você executar `python app.py` ou `python3 app.py` - seu servidor web inicializa, localmente, e você pode preencher um pequeno formulário para obter uma resposta à sua pergunta candente sobre onde OVNIs foram avistados!

Antes de fazer isso, dê uma olhada nas partes do `app.py`:

1. Primeiro, as dependências são carregadas e o aplicativo é iniciado.
2. Em seguida, o modelo é importado.
3. Em seguida, index.html é renderizado na rota inicial.

Na rota `/predict`, várias coisas acontecem quando o formulário é postado:

1. As variáveis do formulário são reunidas e convertidas em um array numpy. Eles são então enviados para o modelo e uma previsão é retornada.
2. Os países que desejamos exibir são renderizados novamente como texto legível de seu código de país previsto e esse valor é enviado de volta para index.html para ser renderizado no modelo.

Usar um modelo dessa maneira, com o Flask e um modelo em conserva, é relativamente simples. O mais difícil é entender qual é o formato dos dados que devem ser enviados ao modelo para se obter uma previsão. Tudo depende de como o modelo foi treinado. Este possui três pontos de dados a serem inseridos a fim de obter uma previsão.

Em um ambiente profissional, você pode ver como uma boa comunicação é necessária entre as pessoas que treinam o modelo e aqueles que o consomem em um aplicativo da web ou móvel. No nosso caso, é apenas uma pessoa, você!

---

## 🚀 Desafio

Em vez de trabalhar em um notebook e importar o modelo para o aplicativo Flask, você pode treinar o modelo diretamente no aplicativo Flask! Tente converter seu código Python no notebook, talvez depois que seus dados forem limpos, para treinar o modelo de dentro do aplicativo em uma rota chamada `train`. Quais são os prós e os contras de seguir esse método?

## [Teste pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18?loc=ptbr)

## Revisão e autoestudo

Existem muitas maneiras de construir um aplicativo da web para consumir modelos de ML. Faça uma lista das maneiras pelas quais você pode usar JavaScript ou Python para construir um aplicativo da web para alavancar o aprendizado de máquina. Considere a arquitetura: o modelo deve permanecer no aplicativo ou na nuvem? Nesse último caso, como você acessaria? Desenhe um modelo arquitetônico para uma solução da Web de ML aplicada.

## Tarefa

[Experimente um modelo diferente](assignment.pt-br.md)
