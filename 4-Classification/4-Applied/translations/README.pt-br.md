# Construindo uma aplicação Web para recomendar culinária

Nesta lição, você construirá um modelo de classificação usando algumas das técnicas que aprendeu nas lições anteriores e com o _dataset_ de cozinhas deliciosas usado ao longo desta série. Além disso, você construirá uma pequena aplicação Web para usar um modelo salvo, aproveitando o tempo de execução da web do Onnx.

Um dos usos práticos mais úteis do aprendizado de máquina é criar sistemas de recomendação, e você pode dar o primeiro passo nessa direção hoje!

[![Introdução a Sistemas de Recomendação](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Clique na imagem acima para ver um vídeo

## [Questionário inicial](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25?loc=ptbr)

Nesta lição você aprenderá:

- Como construir um modelo e salvá-lo como um modelo Onnx
- Como usar o Netron para visualizar o modelo
- Como usar seu modelo em uma aplicação Web para inferência

## Construindo seu modelo

Construir sistemas aplicados de ML é uma parte importante para o aproveitamento dessas tecnologias voltadas para sistemas de negócios. Você pode usar modelos dentro de aplicações Web (e, portanto, usá-los em um contexto offline, se necessário) usando Onnx.

Em uma [lição anterior](../../../3-Web-App/1-Web-App/translations/README.pt-br.md), você construiu um modelo de regressão sobre avistamentos de OVNIs, aplicou "pickle" e o usou em uma aplicação Flask. Embora seja muito útil conhecer essa arquitetura, é uma aplicação full-stack Python e seus requisitos podem incluir o uso de JavaScript.

Nesta lição, você pode construir um sistema básico baseado em JavaScript para inferência. Mas primeiro, você precisa treinar um modelo e convertê-lo para uso através do Onnx.

## Exercício - treinando um modelo de classificação

Primeiro, treine um modelo de classificação usando o _dataset_ que usamos anteriormente. 

1. Comece importando algumas bibliotecas:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Você precisará da '[skl2onnx](https://onnx.ai/sklearn-onnx/)' para ajudar a converter seu modelo Scikit-learn para o formato Onnx.

1. Trabalhe com seus dados da mesma maneira que você fez nas lições anteriores, lendo um arquivo CSV usando `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Remova as duas primeiras colunas desnecessárias e salve os dados restantes na variável 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Salve as categorias na variável 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Começando a rotina de treinamento

Usaremos a biblioteca 'SVC' que tem boa acurácia.

1. Importe as bibliotecas apropriadas do Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Separe os dados em dados de treinamento e teste:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Construa um modelo de classificação SVC como você fez na lição anterior:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Agora teste seu modelo, chamando o método `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Imprima um relatório de classificação para verificar a qualidade do modelo:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Como vimos antes, a acurácia é boa:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### Convertendo seu modelo para o formato Onnx

Certifique-se de fazer a conversão adequada do número Tensor. Este _dataset_ tem 380 ingredientes listados, então você precisa anotar esse número usando a `FloatTensorType`:

1. Converta usando um número tensor de 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Crie a variável onx e armazene como um arquivo chamado **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Observe, você pode passar um dicionário em [options](https://onnx.ai/sklearn-onnx/parameterized.html) no seu script de conversão. Nesse caso, passamos 'nocl' como True e 'zipmap' como False. Por ser um modelo de classificação, você tem a opção de remover o ZipMap, que produz uma lista de dicionários (não é necessário). `nocl` refere-se às informações de classe incluídas no modelo. Reduza o tamanho do seu modelo configurando `nocl` para 'True'. 

Executando o _notebook_ inteiro agora irá construir um modelo Onnx e salvá-lo nesta pasta.

## Visualizando seu modelo

Os modelos Onnx não são muito visíveis no código do Visual Studio, mas existe um software livre muito bom que muitos pesquisadores usam para visualizar o modelo e garantir que ele seja construído corretamente. Baixe o programa [Netron](https://github.com/lutzroeder/Netron) e abra seu arquivo model.onnx com ele. Você pode ver seu modelo simples com suas 380 entradas e suas possíveis saídas:

![Visualização Netron](../images/netron.png)

Netron é uma ferramenta útil para visualizar seus modelos.

Estamos prontos para usar este modelo bacana em uma aplicação Web. Vamos construir uma aplicação que será útil pra quando você olhar em sua geladeira e tentar descobrir qual combinação de seus ingredientes você pode usar para cozinhar um prato de uma determinada culinária específica, conforme determinado por seu modelo.

## Criando uma aplicação Web de recomendação

Você pode usar seu modelo diretamente em uma aplicação Web. Essa arquitetura também permite executá-lo localmente e até mesmo offline, se necessário. Comece criando um arquivo `index.html` na mesma pasta onde você armazenou seu arquivo` model.onnx`.

1. No arquivo _index.html_, adicione o seguinte _markup_:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. Trabalhando com a tag `body`, adicione um pequeno _markup_ para mostrar uma lista de caixas de seleção (input) refletindo alguns ingredientes:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    Observe que cada caixa de seleção recebe um valor. Este valor reflete o índice onde o ingrediente é encontrado de acordo com o _dataset_. O ingrediente "apple", por exemplo, ocupa a quinta coluna, então seu valor é '4' já que começamos a contar em 0. Você pode consultar a [planilha de ingredientes](../../data/ingredient_indexes.csv) para descobrir um determinado índice do ingrediente.

    Continuando seu trabalho no arquivo _index.html_, vamos adicionar um bloco de script onde o modelo é chamado após o fechamento final `</div>`. 

1. Primeiro, importe o [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.09/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime é usado para permitir a execução de seus modelos Onnx em uma ampla gama de plataformas de hardware, incluindo otimizações e uma API.

1. Assim que o Runtime estiver pronto, você pode chamá-lo:

    ```javascript
    <script>
                const ingredients = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
                const checks = [].slice.call(document.querySelectorAll('.checkbox'));
    
                // use an async context to call onnxruntime functions.
                function init() {
                    
                    checks.forEach(function (checkbox, index) {
                        checkbox.onchange = function () {
                            if (this.checked) {
                                var index = checkbox.value;
    
                                if (index !== -1) {
                                    ingredients[index] = 1;
                                }
                                console.log(ingredients)
                            }
                            else {
                                var index = checkbox.value;
    
                                if (index !== -1) {
                                    ingredients[index] = 0;
                                }
                                console.log(ingredients)
                            }
                        }
                    })
                }
    
                function testCheckboxes() {
                        for (var i = 0; i < checks.length; i++)
                            if (checks[i].type == "checkbox")
                                if (checks[i].checked)
                                    return true;
                        return false;
                }
    
                async function startInference() {
    
                    let checked = testCheckboxes()
    
                    if (checked) {
    
                    try {
                        // create a new session and load the model.
                        
                        const session = await ort.InferenceSession.create('./model.onnx');
    
                        const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                        const feeds = { float_input: input };
    
                        // feed inputs and run
    
                        const results = await session.run(feeds);
    
                        // read from results
                        alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')
    
                    } catch (e) {
                        console.log(`failed to inference ONNX model: ${e}.`);
                    }
                }
                else alert("Please check an ingredient")
                    
                }
        init();
               
            </script>
    ```

Neste código, várias coisas acontecem:

1. Existe uma matriz de 380 valores possíveis (1 ou 0) para serem definidos e enviados ao modelo para inferência, dependendo se a caixa de seleção de um ingrediente está marcada.
2. Existe um array de caixas de seleção e uma maneira de determinar se elas foram verificadas é usando a função `init` que é chamada quando a aplicação é iniciada. Quando uma caixa de seleção é marcada, o array `ingredients` é atualizado para refletir o ingrediente escolhido.
3. Existe uma função `testCheckboxes` que verifica se alguma caixa de seleção foi marcada.
4. Você usa essa função quando o botão é pressionado e, se alguma caixa de seleção estiver marcada, você inicia a inferência.
5. A rotina de inferência inclui:
   1. Carregar o modelo de forma assíncrona
   2. Criar uma estrutura de Tensores para enviar ao modelo
   3. Criar 'feeds' que refletem a entrada `float_input` que você criou ao treinar seu modelo (você pode usar o Netron para verificar esse nome)
   4. Enviar esses 'feeds' para a modelo e aguardar uma resposta

## Testando sua aplicação

Abra uma sessão de terminal (prompt, cmd) no Visual Studio Code na pasta onde está o arquivo _index.html_. Certifique-se de ter o pacote [http-server](https://www.npmjs.com/package/http-server) instalado globalmente e digite `http-server` no prompt. Um _localhost_ será aberto e você pode visualizar sua aplicação. Verifique qual cozinha é recomendada com base nos ingredientes:

![Aplicação Web de Ingredientes](../images/web-app.png)

Parabéns, você criou uma aplicação Web de 'recomendação' com alguns campos. Dedique algum tempo para aprimorar o sistema!

## 🚀Desafio

Sua aplicação é simples, portanto, adicione outros ingredientes observando seus índices na [planilha de ingredientes](../../data/ingredient_indexes.csv). Que combinações de sabores funcionam para criar um determinado prato?

## [Questionário para fixação](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26?loc=ptbr)

## Revisão e Auto Aprendizagem

Embora esta lição tenha apenas abordado sobre a criação de um sistema de recomendação para ingredientes alimentícios, esta área de aplicações de ML é muito rica em exemplos. Leia mais sobre como esses sistemas são construídos:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Tarefa 

[Construindo um recomendador](assignment.pt-br.md).
