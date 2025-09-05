<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T15:24:10+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "sw"
}
-->
# Jenga mfano wa regression kwa kutumia Scikit-learn: andaa na onyesha data

![Picha ya infographic ya uonyeshaji wa data](../../../../2-Regression/2-Data/images/data-visualization.png)

Infographic na [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Jaribio la kabla ya somo](https://ff-quizzes.netlify.app/en/ml/)

> ### [Somo hili linapatikana kwa R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Utangulizi

Sasa kwa kuwa umejiandaa na zana unazohitaji kuanza kujenga mifano ya kujifunza kwa mashine kwa kutumia Scikit-learn, uko tayari kuanza kuuliza maswali kuhusu data yako. Unapofanya kazi na data na kutumia suluhisho za ML, ni muhimu sana kuelewa jinsi ya kuuliza swali sahihi ili kufungua uwezo wa dataset yako ipasavyo.

Katika somo hili, utajifunza:

- Jinsi ya kuandaa data yako kwa ajili ya kujenga mifano.
- Jinsi ya kutumia Matplotlib kwa uonyeshaji wa data.

## Kuuliza swali sahihi kuhusu data yako

Swali unalotaka kujibiwa litaamua ni aina gani ya algorithimu za ML utatumia. Na ubora wa jibu unalopata utategemea sana asili ya data yako.

Angalia [data](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) iliyotolewa kwa somo hili. Unaweza kufungua faili hili la .csv katika VS Code. Ukilitazama haraka utaona kuwa kuna nafasi tupu na mchanganyiko wa maandishi na data ya nambari. Pia kuna safu ya ajabu inayoitwa 'Package' ambapo data ni mchanganyiko wa 'sacks', 'bins' na maadili mengine. Kwa kweli, data ni ya fujo kidogo.

[![ML kwa wanaoanza - Jinsi ya Kuchambua na Kusafisha Dataset](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML kwa wanaoanza - Jinsi ya Kuchambua na Kusafisha Dataset")

> 🎥 Bofya picha hapo juu kwa video fupi inayofanya kazi ya kuandaa data kwa somo hili.

Kwa kweli, si kawaida kupewa dataset ambayo iko tayari kabisa kutumika kuunda mfano wa ML moja kwa moja. Katika somo hili, utajifunza jinsi ya kuandaa dataset mbichi kwa kutumia maktaba za kawaida za Python. Pia utajifunza mbinu mbalimbali za kuonyesha data.

## Uchunguzi wa kesi: 'soko la malenge'

Katika folda hii utapata faili la .csv katika folda ya mizizi `data` inayoitwa [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) ambalo lina mistari 1757 ya data kuhusu soko la malenge, limepangwa katika vikundi kulingana na mji. Hii ni data mbichi iliyotolewa kutoka kwa [Ripoti za Kawaida za Masoko ya Mazao Maalum](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) zinazotolewa na Idara ya Kilimo ya Marekani.

### Kuandaa data

Data hii iko katika uwanja wa umma. Inaweza kupakuliwa katika faili nyingi tofauti, kwa kila mji, kutoka tovuti ya USDA. Ili kuepuka faili nyingi tofauti, tumeunganisha data yote ya miji katika lahajedwali moja, kwa hivyo tayari tume _andaa_ data kidogo. Sasa, hebu tuangalie kwa karibu data hiyo.

### Data ya malenge - hitimisho la awali

Unaona nini kuhusu data hii? Tayari umeona kuwa kuna mchanganyiko wa maandishi, nambari, nafasi tupu na maadili ya ajabu ambayo unahitaji kuelewa.

Ni swali gani unaweza kuuliza kuhusu data hii, kwa kutumia mbinu ya Regression? Vipi kuhusu "Tabiri bei ya malenge yanayouzwa katika mwezi fulani". Ukiangalia tena data, kuna mabadiliko unayohitaji kufanya ili kuunda muundo wa data unaohitajika kwa kazi hiyo.

## Zoezi - chunguza data ya malenge

Tutumie [Pandas](https://pandas.pydata.org/), (jina linamaanisha `Python Data Analysis`) chombo chenye manufaa sana kwa kuunda data, kuchambua na kuandaa data hii ya malenge.

### Kwanza, angalia tarehe zinazokosekana

Utahitaji kuchukua hatua za kuangalia tarehe zinazokosekana:

1. Badilisha tarehe kuwa muundo wa mwezi (hizi ni tarehe za Marekani, kwa hivyo muundo ni `MM/DD/YYYY`).
2. Toa mwezi kwenye safu mpya.

Fungua faili _notebook.ipynb_ katika Visual Studio Code na uingize lahajedwali katika dataframe mpya ya Pandas.

1. Tumia kazi ya `head()` kuona mistari mitano ya kwanza.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Ni kazi gani ungetumia kuona mistari mitano ya mwisho?

1. Angalia kama kuna data inayokosekana katika dataframe ya sasa:

    ```python
    pumpkins.isnull().sum()
    ```

    Kuna data inayokosekana, lakini labda haitakuwa muhimu kwa kazi inayofanyika.

1. Ili kufanya dataframe yako iwe rahisi kufanya kazi nayo, chagua tu safu unazohitaji, kwa kutumia kazi ya `loc` ambayo huchukua kutoka dataframe ya awali kikundi cha mistari (kinachopitishwa kama parameter ya kwanza) na safu (zinazopitishwa kama parameter ya pili). Usemi `:` katika kesi hapa chini unamaanisha "mistari yote".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Pili, amua bei ya wastani ya malenge

Fikiria jinsi ya kuamua bei ya wastani ya malenge katika mwezi fulani. Ni safu gani ungezichagua kwa kazi hii? Kidokezo: utahitaji safu 3.

Suluhisho: chukua wastani wa safu za `Low Price` na `High Price` kujaza safu mpya ya Price, na ubadilishe safu ya Date kuonyesha tu mwezi. Kwa bahati nzuri, kulingana na ukaguzi hapo juu, hakuna data inayokosekana kwa tarehe au bei.

1. Ili kuhesabu wastani, ongeza msimbo ufuatao:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Jisikie huru kuchapisha data yoyote unayotaka kuangalia kwa kutumia `print(month)`.

2. Sasa, nakili data yako iliyobadilishwa katika dataframe mpya ya Pandas:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Kuchapisha dataframe yako kutakuonyesha dataset safi na nadhifu ambayo unaweza kujenga mfano wako mpya wa regression.

### Lakini subiri! Kuna kitu cha ajabu hapa

Ukiangalia safu ya `Package`, malenge yanauzwa katika mipangilio mingi tofauti. Baadhi yanauzwa kwa kipimo cha '1 1/9 bushel', na mengine kwa '1/2 bushel', mengine kwa kila malenge, mengine kwa paundi, na mengine katika masanduku makubwa yenye upana tofauti.

> Malenge yanaonekana kuwa magumu sana kupimwa kwa usawa

Ukiangalia data ya awali, ni ya kuvutia kwamba chochote kilicho na `Unit of Sale` sawa na 'EACH' au 'PER BIN' pia kina aina ya `Package` kwa inchi, kwa bin, au 'each'. Malenge yanaonekana kuwa magumu sana kupimwa kwa usawa, kwa hivyo hebu tuyachuje kwa kuchagua tu malenge yenye neno 'bushel' katika safu yao ya `Package`.

1. Ongeza kichujio juu ya faili, chini ya uingizaji wa awali wa .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Ukichapisha data sasa, utaona kuwa unapata tu mistari 415 au zaidi ya data inayojumuisha malenge kwa bushel.

### Lakini subiri! Kuna jambo lingine la kufanya

Je, uliona kuwa kiasi cha bushel kinatofautiana kwa kila mstari? Unahitaji kuweka bei sawa ili uonyeshe bei kwa bushel, kwa hivyo fanya hesabu ili kuisawazisha.

1. Ongeza mistari hii baada ya block inayounda dataframe mpya ya malenge:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Kulingana na [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), uzito wa bushel unategemea aina ya mazao, kwani ni kipimo cha ujazo. "Bushel ya nyanya, kwa mfano, inapaswa kuwa na uzito wa paundi 56... Majani na mboga huchukua nafasi zaidi na uzito mdogo, kwa hivyo bushel ya spinachi ni paundi 20 tu." Ni ngumu sana! Hebu tusijisumbue na kufanya ubadilishaji wa bushel hadi paundi, na badala yake tuweke bei kwa bushel. Utafiti huu wote wa bushel za malenge, hata hivyo, unaonyesha jinsi ilivyo muhimu sana kuelewa asili ya data yako!

Sasa, unaweza kuchambua bei kwa kipimo kulingana na bushel yao. Ukichapisha data mara moja zaidi, utaona jinsi ilivyowekwa sawa.

✅ Je, uliona kuwa malenge yanayouzwa kwa nusu bushel ni ghali sana? Je, unaweza kuelewa kwa nini? Kidokezo: malenge madogo ni ghali sana kuliko makubwa, labda kwa sababu kuna mengi zaidi yao kwa bushel, ikizingatiwa nafasi isiyotumika inayochukuliwa na malenge moja kubwa la pie.

## Mikakati ya Uonyeshaji

Sehemu ya jukumu la mwanasayansi wa data ni kuonyesha ubora na asili ya data wanayofanya kazi nayo. Ili kufanya hivyo, mara nyingi huunda uonyeshaji wa kuvutia, au grafu, chati, na michoro, inayoonyesha vipengele tofauti vya data. Kwa njia hii, wanaweza kuonyesha kwa macho uhusiano na mapungufu ambayo vinginevyo ni vigumu kugundua.

[![ML kwa wanaoanza - Jinsi ya Kuonyesha Data kwa Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML kwa wanaoanza - Jinsi ya Kuonyesha Data kwa Matplotlib")

> 🎥 Bofya picha hapo juu kwa video fupi inayofanya kazi ya kuonyesha data kwa somo hili.

Uonyeshaji pia unaweza kusaidia kuamua mbinu ya kujifunza kwa mashine inayofaa zaidi kwa data. Scatterplot inayofuata mstari, kwa mfano, inaonyesha kuwa data ni mgombea mzuri kwa zoezi la regression ya mstari.

Maktaba moja ya uonyeshaji wa data inayofanya kazi vizuri katika daftari za Jupyter ni [Matplotlib](https://matplotlib.org/) (ambayo pia uliiona katika somo la awali).

> Pata uzoefu zaidi na uonyeshaji wa data katika [mafunzo haya](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Zoezi - jaribu Matplotlib

Jaribu kuunda grafu za msingi kuonyesha dataframe mpya uliyoijenga. Grafu ya mstari wa msingi ingeonyesha nini?

1. Ingiza Matplotlib juu ya faili, chini ya uingizaji wa Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Rerun daftari lote ili kusasisha.
1. Mwisho wa daftari, ongeza seli kuonyesha data kama sanduku:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Scatterplot inayoonyesha uhusiano wa bei na mwezi](../../../../2-Regression/2-Data/images/scatterplot.png)

    Je, grafu hii ni ya manufaa? Je, kuna kitu kinachokushangaza kuhusu grafu hii?

    Haifai sana kwani inachofanya ni kuonyesha data yako kama mchanganyiko wa pointi katika mwezi fulani.

### Ifanye iwe ya manufaa

Ili grafu zionyeshe data ya manufaa, kwa kawaida unahitaji kuunda vikundi vya data kwa namna fulani. Hebu jaribu kuunda grafu ambapo mhimili wa y unaonyesha miezi na data inaonyesha usambazaji wa data.

1. Ongeza seli kuunda grafu ya vikundi vya bar:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Grafu ya bar inayoonyesha uhusiano wa bei na mwezi](../../../../2-Regression/2-Data/images/barchart.png)

    Hii ni uonyeshaji wa data wa manufaa zaidi! Inaonekana kuonyesha kuwa bei ya juu zaidi ya malenge hutokea Septemba na Oktoba. Je, hilo linakubaliana na matarajio yako? Kwa nini au kwa nini siyo?

---

## 🚀Changamoto

Chunguza aina tofauti za uonyeshaji ambazo Matplotlib inatoa. Ni aina gani zinazofaa zaidi kwa matatizo ya regression?

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Angalia njia nyingi za kuonyesha data. Tengeneza orodha ya maktaba mbalimbali zinazopatikana na eleza ni zipi bora kwa aina fulani za kazi, kwa mfano uonyeshaji wa 2D dhidi ya uonyeshaji wa 3D. Unagundua nini?

## Kazi

[Kuchunguza uonyeshaji](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.