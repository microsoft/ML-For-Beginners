<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2680c691fbdb6367f350761a275e2508",
  "translation_date": "2025-08-29T21:36:56+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "br"
}
-->
# Construa um Aplicativo Web para Usar um Modelo de ML

Nesta lição, você irá treinar um modelo de ML em um conjunto de dados fora deste mundo: _avistamentos de OVNIs ao longo do último século_, provenientes do banco de dados do NUFORC.

Você aprenderá:

- Como 'pickle' um modelo treinado
- Como usar esse modelo em um aplicativo Flask

Continuaremos utilizando notebooks para limpar os dados e treinar nosso modelo, mas você pode levar o processo um passo adiante explorando o uso de um modelo "na prática", por assim dizer: em um aplicativo web.

Para fazer isso, você precisa construir um aplicativo web usando Flask.

## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Construindo um aplicativo

Existem várias maneiras de construir aplicativos web para consumir modelos de aprendizado de máquina. Sua arquitetura web pode influenciar a forma como seu modelo é treinado. Imagine que você está trabalhando em uma empresa onde o grupo de ciência de dados treinou um modelo que eles querem que você use em um aplicativo.

### Considerações

Há muitas perguntas que você precisa fazer:

- **É um aplicativo web ou um aplicativo móvel?** Se você está construindo um aplicativo móvel ou precisa usar o modelo em um contexto de IoT, você pode usar [TensorFlow Lite](https://www.tensorflow.org/lite/) e utilizar o modelo em um aplicativo Android ou iOS.
- **Onde o modelo ficará armazenado?** Na nuvem ou localmente?
- **Suporte offline.** O aplicativo precisa funcionar offline?
- **Qual tecnologia foi usada para treinar o modelo?** A tecnologia escolhida pode influenciar as ferramentas que você precisa usar.
    - **Usando TensorFlow.** Se você está treinando um modelo usando TensorFlow, por exemplo, esse ecossistema oferece a capacidade de converter um modelo TensorFlow para uso em um aplicativo web utilizando [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Usando PyTorch.** Se você está construindo um modelo usando uma biblioteca como [PyTorch](https://pytorch.org/), você tem a opção de exportá-lo no formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) para uso em aplicativos web JavaScript que podem utilizar o [Onnx Runtime](https://www.onnxruntime.ai/). Essa opção será explorada em uma lição futura para um modelo treinado com Scikit-learn.
    - **Usando Lobe.ai ou Azure Custom Vision.** Se você está utilizando um sistema de ML SaaS (Software como Serviço) como [Lobe.ai](https://lobe.ai/) ou [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) para treinar um modelo, esse tipo de software oferece maneiras de exportar o modelo para várias plataformas, incluindo a construção de uma API personalizada para ser consultada na nuvem pelo seu aplicativo online.

Você também tem a oportunidade de construir um aplicativo web Flask completo que seria capaz de treinar o modelo diretamente em um navegador web. Isso também pode ser feito usando TensorFlow.js em um contexto JavaScript.

Para nossos propósitos, já que estamos trabalhando com notebooks baseados em Python, vamos explorar os passos necessários para exportar um modelo treinado de um notebook para um formato legível por um aplicativo web construído em Python.

## Ferramenta

Para esta tarefa, você precisa de duas ferramentas: Flask e Pickle, ambas executadas em Python.

✅ O que é [Flask](https://palletsprojects.com/p/flask/)? Definido como um 'micro-framework' por seus criadores, Flask fornece os recursos básicos de frameworks web usando Python e um mecanismo de templates para construir páginas web. Confira [este módulo de aprendizado](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para praticar a construção com Flask.

✅ O que é [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 é um módulo Python que serializa e desserializa uma estrutura de objeto Python. Quando você 'pickle' um modelo, você serializa ou achata sua estrutura para uso na web. Atenção: pickle não é intrinsecamente seguro, então tenha cuidado ao ser solicitado a 'despickle' um arquivo. Um arquivo pickled tem o sufixo `.pkl`.

## Exercício - limpe seus dados

Nesta lição, você usará dados de 80.000 avistamentos de OVNIs, coletados pelo [NUFORC](https://nuforc.org) (Centro Nacional de Relatórios de OVNIs). Esses dados têm descrições interessantes de avistamentos de OVNIs, por exemplo:

- **Descrição longa de exemplo.** "Um homem emerge de um feixe de luz que brilha em um campo gramado à noite e corre em direção ao estacionamento da Texas Instruments".
- **Descrição curta de exemplo.** "as luzes nos perseguiram".

A planilha [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) inclui colunas sobre a `cidade`, `estado` e `país` onde o avistamento ocorreu, o `formato` do objeto e sua `latitude` e `longitude`.

No [notebook](notebook.ipynb) em branco incluído nesta lição:

1. Importe `pandas`, `matplotlib` e `numpy` como você fez em lições anteriores e importe a planilha de ufos. Você pode dar uma olhada em um conjunto de dados de amostra:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Converta os dados de ufos para um pequeno dataframe com títulos novos. Verifique os valores únicos no campo `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Agora, você pode reduzir a quantidade de dados com que precisamos lidar, descartando quaisquer valores nulos e importando apenas avistamentos entre 1-60 segundos:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importe a biblioteca `LabelEncoder` do Scikit-learn para converter os valores de texto dos países em números:

    ✅ LabelEncoder codifica dados alfabeticamente

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Seus dados devem se parecer com isto:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Exercício - construa seu modelo

Agora você pode se preparar para treinar um modelo dividindo os dados em grupos de treinamento e teste.

1. Selecione as três características que você deseja treinar como seu vetor X, e o vetor y será o `Country`. Você quer ser capaz de inserir `Seconds`, `Latitude` e `Longitude` e obter um id de país como retorno.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Treine seu modelo usando regressão logística:

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

A precisão não é ruim **(cerca de 95%)**, o que não é surpreendente, já que `Country` e `Latitude/Longitude` estão correlacionados.

O modelo que você criou não é muito revolucionário, já que você deveria ser capaz de inferir um `Country` a partir de sua `Latitude` e `Longitude`, mas é um bom exercício para tentar treinar a partir de dados brutos que você limpou, exportou e depois usou esse modelo em um aplicativo web.

## Exercício - 'pickle' seu modelo

Agora é hora de _pickle_ seu modelo! Você pode fazer isso em algumas linhas de código. Uma vez que ele esteja _pickled_, carregue seu modelo pickled e teste-o contra um array de dados de amostra contendo valores para segundos, latitude e longitude.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

O modelo retorna **'3'**, que é o código do país para o Reino Unido. Incrível! 👽

## Exercício - construa um aplicativo Flask

Agora você pode construir um aplicativo Flask para chamar seu modelo e retornar resultados semelhantes, mas de uma maneira mais visualmente agradável.

1. Comece criando uma pasta chamada **web-app** ao lado do arquivo _notebook.ipynb_ onde seu arquivo _ufo-model.pkl_ está localizado.

1. Dentro dessa pasta, crie mais três pastas: **static**, com uma pasta **css** dentro dela, e **templates**. Você deve ter agora os seguintes arquivos e diretórios:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulte a pasta de solução para ver o aplicativo finalizado

1. O primeiro arquivo a ser criado na pasta _web-app_ é o arquivo **requirements.txt**. Como o _package.json_ em um aplicativo JavaScript, este arquivo lista as dependências necessárias para o aplicativo. No **requirements.txt** adicione as linhas:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Agora, execute este arquivo navegando até _web-app_:

    ```bash
    cd web-app
    ```

1. No seu terminal, digite `pip install` para instalar as bibliotecas listadas em _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Agora, você está pronto para criar mais três arquivos para finalizar o aplicativo:

    1. Crie **app.py** na raiz.
    2. Crie **index.html** no diretório _templates_.
    3. Crie **styles.css** no diretório _static/css_.

1. Desenvolva o arquivo _styles.css_ com alguns estilos:

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

1. Em seguida, desenvolva o arquivo _index.html_:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    Observe o uso de templates neste arquivo. Note a sintaxe 'mustache' em torno das variáveis que serão fornecidas pelo aplicativo, como o texto de previsão: `{{}}`. Há também um formulário que envia uma previsão para a rota `/predict`.

    Finalmente, você está pronto para construir o arquivo Python que dirige o consumo do modelo e a exibição das previsões:

1. No `app.py` adicione:

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

    > 💡 Dica: quando você adiciona [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) ao executar o aplicativo web usando Flask, quaisquer alterações feitas no seu aplicativo serão refletidas imediatamente sem a necessidade de reiniciar o servidor. Atenção! Não habilite este modo em um aplicativo de produção.

Se você executar `python app.py` ou `python3 app.py` - seu servidor web será iniciado localmente, e você poderá preencher um formulário simples para obter uma resposta à sua pergunta sobre onde os OVNIs foram avistados!

Antes de fazer isso, dê uma olhada nas partes do `app.py`:

1. Primeiro, as dependências são carregadas e o aplicativo é iniciado.
1. Em seguida, o modelo é importado.
1. Depois, o index.html é renderizado na rota inicial.

Na rota `/predict`, várias coisas acontecem quando o formulário é enviado:

1. As variáveis do formulário são coletadas e convertidas em um array numpy. Elas são então enviadas ao modelo e uma previsão é retornada.
2. Os países que queremos exibir são re-renderizados como texto legível a partir de seu código de país previsto, e esse valor é enviado de volta ao index.html para ser renderizado no template.

Usar um modelo dessa forma, com Flask e um modelo pickled, é relativamente simples. A parte mais difícil é entender qual é o formato dos dados que devem ser enviados ao modelo para obter uma previsão. Isso depende de como o modelo foi treinado. Este modelo requer três pontos de dados como entrada para gerar uma previsão.

Em um ambiente profissional, você pode ver como é importante ter uma boa comunicação entre as pessoas que treinam o modelo e aquelas que o consomem em um aplicativo web ou móvel. No nosso caso, é apenas uma pessoa: você!

---

## 🚀 Desafio

Em vez de trabalhar em um notebook e importar o modelo para o aplicativo Flask, você poderia treinar o modelo diretamente dentro do aplicativo Flask! Tente converter seu código Python no notebook, talvez após limpar seus dados, para treinar o modelo dentro do aplicativo em uma rota chamada `train`. Quais são os prós e contras de seguir esse método?

## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Revisão e Autoestudo

Existem várias maneiras de construir um aplicativo web para consumir modelos de ML. Faça uma lista das maneiras que você poderia usar JavaScript ou Python para construir um aplicativo web que aproveite o aprendizado de máquina. Considere a arquitetura: o modelo deve permanecer no aplicativo ou viver na nuvem? Se for o último caso, como você o acessaria? Desenhe um modelo arquitetural para uma solução web aplicada de ML.

## Tarefa

[Experimente um modelo diferente](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.