<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T15:32:26+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "sw"
}
-->
# Utangulizi wa utabiri wa mfululizo wa muda

![Muhtasari wa mfululizo wa muda katika sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote na [Tomomi Imura](https://www.twitter.com/girlie_mac)

Katika somo hili na linalofuata, utajifunza kidogo kuhusu utabiri wa mfululizo wa muda, sehemu ya kuvutia na yenye thamani katika ujuzi wa mwanasayansi wa ML ambayo haijulikani sana ikilinganishwa na mada nyingine. Utabiri wa mfululizo wa muda ni kama 'kioo cha uchawi': kwa kuzingatia utendaji wa zamani wa kigezo kama bei, unaweza kutabiri thamani yake ya baadaye.

[![Utangulizi wa utabiri wa mfululizo wa muda](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Utangulizi wa utabiri wa mfululizo wa muda")

> 🎥 Bofya picha hapo juu kwa video kuhusu utabiri wa mfululizo wa muda

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

Ni uwanja wa manufaa na wa kuvutia wenye thamani halisi kwa biashara, kutokana na matumizi yake ya moja kwa moja katika masuala ya bei, hesabu, na masuala ya mnyororo wa ugavi. Ingawa mbinu za kujifunza kwa kina zimeanza kutumika kupata maarifa zaidi ili kutabiri utendaji wa baadaye, utabiri wa mfululizo wa muda bado ni uwanja unaotegemea sana mbinu za kawaida za ML.

> Mtaala wa mfululizo wa muda wa Penn State unaweza kupatikana [hapa](https://online.stat.psu.edu/stat510/lesson/1)

## Utangulizi

Fikiria unadhibiti safu ya mita za maegesho za kisasa zinazotoa data kuhusu mara ngapi zinatumiwa na kwa muda gani kwa kipindi fulani.

> Je, ungeweza kutabiri, kwa kuzingatia utendaji wa zamani wa mita, thamani yake ya baadaye kulingana na sheria za usambazaji na mahitaji?

Kutabiri kwa usahihi wakati wa kuchukua hatua ili kufanikisha lengo lako ni changamoto inayoweza kushughulikiwa na utabiri wa mfululizo wa muda. Ingawa haitawafurahisha watu kutozwa zaidi wakati wa nyakati za shughuli nyingi wanapotafuta nafasi ya maegesho, itakuwa njia ya uhakika ya kuzalisha mapato ya kusafisha mitaa!

Hebu tuchunguze baadhi ya aina za algoriti za mfululizo wa muda na tuanze daftari la kusafisha na kuandaa data. Data utakayochambua imetolewa kutoka kwa mashindano ya utabiri ya GEFCom2014. Inajumuisha miaka 3 ya mzigo wa umeme wa kila saa na thamani za joto kati ya 2012 na 2014. Kwa kuzingatia mifumo ya kihistoria ya mzigo wa umeme na joto, unaweza kutabiri thamani za baadaye za mzigo wa umeme.

Katika mfano huu, utajifunza jinsi ya kutabiri hatua moja mbele ya muda, kwa kutumia data ya mzigo wa kihistoria pekee. Kabla ya kuanza, hata hivyo, ni muhimu kuelewa kinachoendelea nyuma ya pazia.

## Baadhi ya ufafanuzi

Unapokutana na neno 'mfululizo wa muda' unahitaji kuelewa matumizi yake katika muktadha tofauti.

🎓 **Mfululizo wa muda**

Katika hisabati, "mfululizo wa muda ni mfululizo wa alama za data zilizoorodheshwa (au zilizoorodheshwa au kuchorwa) kwa mpangilio wa muda. Mara nyingi zaidi, mfululizo wa muda ni mlolongo uliochukuliwa kwa vipindi vya muda vilivyopangwa sawa." Mfano wa mfululizo wa muda ni thamani ya kufunga ya kila siku ya [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Matumizi ya michoro ya mfululizo wa muda na uundaji wa takwimu mara nyingi hukutana katika usindikaji wa ishara, utabiri wa hali ya hewa, utabiri wa matetemeko ya ardhi, na nyanja nyingine ambapo matukio hutokea na alama za data zinaweza kuchorwa kwa muda.

🎓 **Uchambuzi wa mfululizo wa muda**

Uchambuzi wa mfululizo wa muda ni uchambuzi wa data ya mfululizo wa muda iliyotajwa hapo juu. Data ya mfululizo wa muda inaweza kuchukua aina tofauti, ikiwa ni pamoja na 'mfululizo wa muda uliokatizwa' ambao hugundua mifumo katika mabadiliko ya mfululizo wa muda kabla na baada ya tukio linalokatiza. Aina ya uchambuzi inayohitajika kwa mfululizo wa muda inategemea asili ya data. Data ya mfululizo wa muda yenyewe inaweza kuchukua fomu ya mfululizo wa namba au herufi.

Uchambuzi unaofanywa hutumia mbinu mbalimbali, ikiwa ni pamoja na uwanja wa masafa na uwanja wa muda, mstari na usio mstari, na zaidi. [Jifunze zaidi](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) kuhusu njia nyingi za kuchambua aina hii ya data.

🎓 **Utabiri wa mfululizo wa muda**

Utabiri wa mfululizo wa muda ni matumizi ya mfano kutabiri thamani za baadaye kulingana na mifumo inayoonyeshwa na data iliyokusanywa hapo awali kama ilivyotokea zamani. Ingawa inawezekana kutumia mifano ya regression kuchunguza data ya mfululizo wa muda, na fahirisi za muda kama vigezo vya x kwenye mchoro, data kama hiyo inachambuliwa vyema kwa kutumia aina maalum za mifano.

Data ya mfululizo wa muda ni orodha ya uchunguzi ulioamriwa, tofauti na data inayoweza kuchambuliwa kwa regression ya mstari. Mfano wa kawaida ni ARIMA, kifupi cha "Autoregressive Integrated Moving Average".

[Mifano ya ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "inaunganisha thamani ya sasa ya mfululizo na thamani za zamani na makosa ya utabiri ya zamani." Zinafaa zaidi kwa kuchambua data ya uwanja wa muda, ambapo data imepangwa kwa muda.

> Kuna aina kadhaa za mifano ya ARIMA, ambayo unaweza kujifunza kuhusu [hapa](https://people.duke.edu/~rnau/411arim.htm) na ambayo utagusia katika somo linalofuata.

Katika somo linalofuata, utajenga mfano wa ARIMA kwa kutumia [Mfululizo wa Muda wa Kigezo Kimoja](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), ambao unazingatia kigezo kimoja kinachobadilisha thamani yake kwa muda. Mfano wa aina hii ya data ni [seti ya data hii](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) inayorekodi mkusanyiko wa CO2 wa kila mwezi katika Maobservatori ya Mauna Loa:

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Tambua kigezo kinachobadilika kwa muda katika seti hii ya data.

## Tabia za data ya mfululizo wa muda za kuzingatia

Unapochunguza data ya mfululizo wa muda, unaweza kugundua kuwa ina [tabia fulani](https://online.stat.psu.edu/stat510/lesson/1/1.1) unazohitaji kuzingatia na kupunguza ili kuelewa vyema mifumo yake. Ukizingatia data ya mfululizo wa muda kama inayoweza kutoa 'ishara' unayotaka kuchambua, tabia hizi zinaweza kufikiriwa kama 'kelele'. Mara nyingi utahitaji kupunguza 'kelele' hii kwa kupunguza baadhi ya tabia hizi kwa kutumia mbinu za takwimu.

Hapa kuna dhana unazopaswa kujua ili kuweza kufanya kazi na mfululizo wa muda:

🎓 **Mwelekeo**

Mwelekeo hufafanuliwa kama ongezeko na upungufu unaoweza kupimwa kwa muda. [Soma zaidi](https://machinelearningmastery.com/time-series-trends-in-python). Katika muktadha wa mfululizo wa muda, ni kuhusu jinsi ya kutumia na, ikiwa ni lazima, kuondoa mwelekeo kutoka kwa mfululizo wako wa muda.

🎓 **[Msimu](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Msimu hufafanuliwa kama mabadiliko ya mara kwa mara, kama vile msimu wa sikukuu ambao unaweza kuathiri mauzo, kwa mfano. [Angalia](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) jinsi aina tofauti za michoro zinavyoonyesha msimu katika data.

🎓 **Vipimo vya mbali**

Vipimo vya mbali viko mbali sana na tofauti ya kawaida ya data.

🎓 **Mzunguko wa muda mrefu**

Huru na msimu, data inaweza kuonyesha mzunguko wa muda mrefu kama vile kushuka kwa uchumi kunakodumu zaidi ya mwaka mmoja.

🎓 **Tofauti ya mara kwa mara**

Kwa muda, data fulani huonyesha mabadiliko ya mara kwa mara, kama vile matumizi ya nishati kwa siku na usiku.

🎓 **Mabadiliko ya ghafla**

Data inaweza kuonyesha mabadiliko ya ghafla ambayo yanaweza kuhitaji uchambuzi zaidi. Kufungwa kwa ghafla kwa biashara kutokana na COVID, kwa mfano, kulisababisha mabadiliko katika data.

✅ Hapa kuna [mchoro wa mfululizo wa muda wa mfano](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) unaoonyesha matumizi ya sarafu ya ndani ya mchezo kwa siku kwa miaka michache. Je, unaweza kutambua yoyote ya tabia zilizoorodheshwa hapo juu katika data hii?

![Matumizi ya sarafu ya ndani ya mchezo](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Zoezi - kuanza na data ya matumizi ya umeme

Hebu tuanze kuunda mfano wa mfululizo wa muda ili kutabiri matumizi ya umeme ya baadaye kwa kuzingatia matumizi ya zamani.

> Data katika mfano huu imetolewa kutoka kwa mashindano ya utabiri ya GEFCom2014. Inajumuisha miaka 3 ya mzigo wa umeme wa kila saa na thamani za joto kati ya 2012 na 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli na Rob J. Hyndman, "Utabiri wa nishati wa uwezekano: Mashindano ya Utabiri wa Nishati ya Kimataifa 2014 na zaidi", Jarida la Kimataifa la Utabiri, vol.32, no.3, uk. 896-913, Julai-Septemba, 2016.

1. Katika folda ya `working` ya somo hili, fungua faili _notebook.ipynb_. Anza kwa kuongeza maktaba zitakazokusaidia kupakia na kuona data

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Kumbuka, unatumia faili kutoka folda ya `common` iliyojumuishwa ambayo inaweka mazingira yako na kushughulikia upakuaji wa data.

2. Kisha, chunguza data kama dataframe kwa kuita `load_data()` na `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Unaweza kuona kuwa kuna safu mbili zinazowakilisha tarehe na mzigo:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Sasa, chora data kwa kuita `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![mchoro wa nishati](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Sasa, chora wiki ya kwanza ya Julai 2014, kwa kuipatia kama ingizo kwa `energy` katika muundo wa `[kutoka tarehe]: [hadi tarehe]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![julai](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Mchoro mzuri! Angalia michoro hii na uone kama unaweza kubaini yoyote ya tabia zilizoorodheshwa hapo juu. Tunaweza kusema nini kwa kuona data?

Katika somo linalofuata, utaunda mfano wa ARIMA ili kuunda baadhi ya utabiri.

---

## 🚀Changamoto

Tengeneza orodha ya sekta zote na maeneo ya uchunguzi unayoweza kufikiria ambayo yangefaidika na utabiri wa mfululizo wa muda. Je, unaweza kufikiria matumizi ya mbinu hizi katika sanaa? Katika Uchumi? Ikolojia? Uuzaji? Sekta? Fedha? Wapi kwingine?

## [Maswali ya baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Ingawa hatutazungumzia hapa, mitandao ya neva wakati mwingine hutumika kuboresha mbinu za kawaida za utabiri wa mfululizo wa muda. Soma zaidi kuhusu hayo [katika makala hii](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Kazi

[Chora mfululizo wa muda zaidi](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.