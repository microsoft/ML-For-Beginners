<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T00:48:18+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "fi"
}
-->
# Rakenna ruokasuositusverkkosovellus

Tässä oppitunnissa rakennat luokittelumallin käyttäen joitakin aiemmissa oppitunneissa opittuja tekniikoita sekä herkullista ruokadatasettiä, jota on käytetty läpi tämän sarjan. Lisäksi rakennat pienen verkkosovelluksen, joka hyödyntää tallennettua mallia Onnxin verkkokäyttöliittymän avulla.

Yksi koneoppimisen hyödyllisimmistä käytännön sovelluksista on suositusjärjestelmien rakentaminen, ja voit ottaa ensimmäisen askeleen siihen suuntaan jo tänään!

[![Esittely tästä verkkosovelluksesta](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi videon: Jen Looper rakentaa verkkosovelluksen luokitellun ruokadatan avulla

## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

Tässä oppitunnissa opit:

- Kuinka rakentaa malli ja tallentaa se Onnx-muodossa
- Kuinka käyttää Netronia mallin tarkasteluun
- Kuinka käyttää malliasi verkkosovelluksessa ennustamiseen

## Rakenna mallisi

Soveltuvien koneoppimisjärjestelmien rakentaminen on tärkeä osa näiden teknologioiden hyödyntämistä liiketoimintajärjestelmissäsi. Voit käyttää malleja verkkosovelluksissasi (ja siten käyttää niitä myös offline-tilassa tarvittaessa) Onnxin avulla.

[Edellisessä oppitunnissa](../../3-Web-App/1-Web-App/README.md) rakensit regressiomallin UFO-havainnoista, "picklasit" sen ja käytit sitä Flask-sovelluksessa. Vaikka tämä arkkitehtuuri on erittäin hyödyllinen, se on täysimittainen Python-sovellus, ja vaatimuksesi saattavat sisältää JavaScript-sovelluksen käytön.

Tässä oppitunnissa voit rakentaa yksinkertaisen JavaScript-pohjaisen järjestelmän ennustamista varten. Ensin sinun täytyy kuitenkin kouluttaa malli ja muuntaa se Onnxin käyttöön.

## Harjoitus - kouluta luokittelumalli

Ensiksi kouluta luokittelumalli käyttäen puhdistettua ruokadatasettiä, jota olemme käyttäneet.

1. Aloita tuomalla hyödylliset kirjastot:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Tarvitset '[skl2onnx](https://onnx.ai/sklearn-onnx/)'-kirjaston, joka auttaa muuntamaan Scikit-learn-mallisi Onnx-muotoon.

1. Työskentele datasi kanssa samalla tavalla kuin aiemmissa oppitunneissa, lukemalla CSV-tiedosto `read_csv()`-funktiolla:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Poista kaksi ensimmäistä tarpeetonta saraketta ja tallenna jäljelle jäävä data nimellä 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Tallenna etiketit nimellä 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Aloita koulutusrutiini

Käytämme 'SVC'-kirjastoa, joka tarjoaa hyvän tarkkuuden.

1. Tuo tarvittavat kirjastot Scikit-learnista:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Jaa data koulutus- ja testijoukkoihin:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Rakenna SVC-luokittelumalli kuten teit edellisessä oppitunnissa:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Testaa malliasi kutsumalla `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Tulosta luokitteluraportti tarkistaaksesi mallin laadun:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Kuten aiemmin näimme, tarkkuus on hyvä:

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

### Muunna mallisi Onnx-muotoon

Varmista, että muunnat mallin oikealla tensorimäärällä. Tässä datasetissä on 380 ainesosaa, joten sinun täytyy merkitä tämä määrä `FloatTensorType`-parametriin:

1. Muunna käyttäen tensorimäärää 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Luo onx-tiedosto ja tallenna se nimellä **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Huomaa, että voit välittää [asetuksia](https://onnx.ai/sklearn-onnx/parameterized.html) muunnosskriptiisi. Tässä tapauksessa välitimme 'nocl'-asetuksen arvoksi True ja 'zipmap'-asetuksen arvoksi False. Koska tämä on luokittelumalli, sinulla on mahdollisuus poistaa ZipMap, joka tuottaa listan sanakirjoja (ei tarpeellinen). `nocl` viittaa luokkainformaation sisällyttämiseen malliin. Pienennä mallisi kokoa asettamalla `nocl` arvoksi 'True'.

Kun suoritat koko notebookin, se rakentaa Onnx-mallin ja tallentaa sen tähän kansioon.

## Tarkastele malliasi

Onnx-mallit eivät ole kovin näkyviä Visual Studio Codessa, mutta on olemassa erittäin hyvä ilmainen ohjelmisto, jota monet tutkijat käyttävät mallien visualisointiin varmistaakseen, että ne on rakennettu oikein. Lataa [Netron](https://github.com/lutzroeder/Netron) ja avaa model.onnx-tiedostosi. Voit nähdä yksinkertaisen mallisi visualisoituna, jossa on 380 syötettä ja luokittelija listattuna:

![Netron visual](../../../../4-Classification/4-Applied/images/netron.png)

Netron on hyödyllinen työkalu mallien tarkasteluun.

Nyt olet valmis käyttämään tätä siistiä mallia verkkosovelluksessa. Rakennetaan sovellus, joka on kätevä, kun katsot jääkaappiasi ja yrität selvittää, mitä yhdistelmää jäljellä olevista ainesosista voit käyttää tietyn keittiön ruokalajin valmistamiseen mallisi perusteella.

## Rakenna suositusverkkosovellus

Voit käyttää malliasi suoraan verkkosovelluksessa. Tämä arkkitehtuuri mahdollistaa sen käytön paikallisesti ja jopa offline-tilassa tarvittaessa. Aloita luomalla `index.html`-tiedosto samaan kansioon, jossa tallensit `model.onnx`-tiedostosi.

1. Lisää tähän tiedostoon _index.html_ seuraava merkintä:

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

1. Työskentele nyt `body`-tagien sisällä ja lisää hieman merkintää, joka näyttää listan valintaruuduista, jotka heijastavat joitakin ainesosia:

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

    Huomaa, että jokaiselle valintaruudulle on annettu arvo. Tämä heijastaa indeksin, jossa ainesosa löytyy datasetistä. Esimerkiksi omena, tässä aakkosjärjestyksessä, sijaitsee viidennessä sarakkeessa, joten sen arvo on '4', koska laskenta alkaa nollasta. Voit tarkistaa [ainesosien taulukon](../../../../4-Classification/data/ingredient_indexes.csv) löytääksesi tietyn ainesosan indeksin.

    Jatka työtäsi index.html-tiedostossa ja lisää skriptilohko, jossa malli kutsutaan viimeisen sulkevan `</div>`-tagin jälkeen.

1. Tuo ensin [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime mahdollistaa Onnx-mallien suorittamisen laajalla valikoimalla laitteistoalustoja, mukaan lukien optimoinnit ja API:n käyttö.

1. Kun Runtime on paikallaan, voit kutsua sen:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
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
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

Tässä koodissa tapahtuu useita asioita:

1. Loit 380 mahdollisen arvon (1 tai 0) taulukon, joka asetetaan ja lähetetään mallille ennustamista varten riippuen siitä, onko ainesosan valintaruutu valittu.
2. Loit valintaruutujen taulukon ja tavan määrittää, onko niitä valittu `init`-funktiossa, joka kutsutaan sovelluksen käynnistyessä. Kun valintaruutu valitaan, `ingredients`-taulukko muuttuu heijastamaan valittua ainesosaa.
3. Loit `testCheckboxes`-funktion, joka tarkistaa, onko mitään valintaruutua valittu.
4. Käytät `startInference`-funktiota, kun painiketta painetaan, ja jos jokin valintaruutu on valittu, aloitat ennustamisen.
5. Ennustusrutiini sisältää:
   1. Asynkronisen mallin latauksen asettamisen
   2. Tensor-rakenteen luomisen, joka lähetetään mallille
   3. 'Feeds'-rakenteen luomisen, joka heijastaa `float_input`-syötettä, jonka loit kouluttaessasi malliasi (voit käyttää Netronia varmistaaksesi nimen)
   4. Näiden 'feeds'-rakenteiden lähettämisen mallille ja odottamisen vastaukselle

## Testaa sovellustasi

Avaa terminaali Visual Studio Codessa kansiossa, jossa index.html-tiedostosi sijaitsee. Varmista, että sinulla on [http-server](https://www.npmjs.com/package/http-server) asennettuna globaalisti, ja kirjoita `http-server` kehotteeseen. Paikallinen palvelin pitäisi avautua, ja voit tarkastella verkkosovellustasi. Tarkista, mitä keittiötä suositellaan eri ainesosien perusteella:

![ainesosien verkkosovellus](../../../../4-Classification/4-Applied/images/web-app.png)

Onnittelut, olet luonut 'suositus' verkkosovelluksen muutamalla kentällä. Käytä aikaa tämän järjestelmän kehittämiseen!

## 🚀Haaste

Verkkosovelluksesi on hyvin yksinkertainen, joten jatka sen kehittämistä käyttämällä ainesosia ja niiden indeksejä [ainesosien indeksit](../../../../4-Classification/data/ingredient_indexes.csv) -datasta. Mitkä makuyhdistelmät toimivat tietyn kansallisen ruokalajin luomisessa?

## [Jälkikysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itseopiskelu

Vaikka tämä oppitunti vain sivusi ruokasuositusjärjestelmän luomisen hyödyllisyyttä, tämä koneoppimisen sovellusalue on erittäin rikas esimerkeissä. Lue lisää siitä, kuinka näitä järjestelmiä rakennetaan:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Tehtävä 

[Rakenna uusi suositusjärjestelmä](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulee pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskääntämistä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.