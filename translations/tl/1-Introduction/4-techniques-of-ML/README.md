<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T18:18:13+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "tl"
}
-->
# Mga Teknik ng Machine Learning

Ang proseso ng pagbuo, paggamit, at pagpapanatili ng mga modelo ng machine learning at ang datos na ginagamit nito ay ibang-iba kumpara sa maraming iba pang mga workflow ng pag-develop. Sa araling ito, lilinawin natin ang proseso at ilalahad ang mga pangunahing teknik na kailangan mong malaman. Ikaw ay:

- Mauunawaan ang mga proseso sa likod ng machine learning sa mataas na antas.
- Susuriin ang mga pangunahing konsepto tulad ng 'mga modelo', 'mga prediksyon', at 'training data'.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML para sa mga baguhan - Mga Teknik ng Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML para sa mga baguhan - Mga Teknik ng Machine Learning")

> 🎥 I-click ang imahe sa itaas para sa isang maikling video na tumatalakay sa araling ito.

## Panimula

Sa mataas na antas, ang sining ng paglikha ng mga proseso ng machine learning (ML) ay binubuo ng ilang mga hakbang:

1. **Magpasya sa tanong**. Karamihan sa mga proseso ng ML ay nagsisimula sa pagtatanong ng tanong na hindi masagot ng simpleng conditional program o rules-based engine. Ang mga tanong na ito ay madalas na umiikot sa mga prediksyon batay sa koleksyon ng datos.
2. **Kolektahin at ihanda ang datos**. Upang masagot ang iyong tanong, kailangan mo ng datos. Ang kalidad at, minsan, dami ng iyong datos ang magtatakda kung gaano kahusay mong masasagot ang iyong tanong. Ang pag-visualize ng datos ay mahalagang aspeto ng yugtong ito. Kasama rin dito ang paghahati ng datos sa training at testing group upang makabuo ng modelo.
3. **Pumili ng paraan ng training**. Depende sa iyong tanong at sa likas na katangian ng iyong datos, kailangan mong pumili kung paano mo gustong i-train ang modelo upang pinakamahusay na maipakita ang datos at makagawa ng tumpak na prediksyon. Ang bahaging ito ng proseso ng ML ay nangangailangan ng tiyak na kadalubhasaan at, madalas, maraming eksperimento.
4. **I-train ang modelo**. Gamit ang iyong training data, gagamit ka ng iba't ibang algorithm upang i-train ang modelo na makilala ang mga pattern sa datos. Ang modelo ay maaaring gumamit ng internal weights na maaaring i-adjust upang bigyang-priyoridad ang ilang bahagi ng datos kaysa sa iba upang makabuo ng mas mahusay na modelo.
5. **Suriin ang modelo**. Gagamit ka ng datos na hindi pa nakikita (ang iyong testing data) mula sa nakolektang set upang makita kung paano gumagana ang modelo.
6. **Parameter tuning**. Batay sa performance ng iyong modelo, maaari mong ulitin ang proseso gamit ang iba't ibang parameter o variable na kumokontrol sa pag-uugali ng mga algorithm na ginamit upang i-train ang modelo.
7. **Mag-predict**. Gumamit ng bagong inputs upang subukan ang katumpakan ng iyong modelo.

## Anong tanong ang dapat itanong

Ang mga computer ay partikular na mahusay sa pagtuklas ng mga nakatagong pattern sa datos. Ang kakayahang ito ay napakahalaga para sa mga mananaliksik na may mga tanong tungkol sa isang partikular na larangan na hindi madaling masagot sa pamamagitan ng paggawa ng conditionally-based rules engine. Halimbawa, sa isang actuarial na gawain, maaaring makagawa ang isang data scientist ng mga handcrafted rules tungkol sa mortality ng mga naninigarilyo kumpara sa mga hindi naninigarilyo.

Kapag maraming iba pang variable ang isinama sa equation, gayunpaman, maaaring mas epektibo ang isang ML model sa pag-predict ng mga future mortality rates batay sa nakaraang health history. Isang mas masayang halimbawa ay ang paggawa ng mga prediksyon sa panahon para sa buwan ng Abril sa isang partikular na lokasyon batay sa datos na kinabibilangan ng latitude, longitude, climate change, proximity sa dagat, mga pattern ng jet stream, at iba pa.

✅ Ang [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) na ito tungkol sa mga weather models ay nagbibigay ng historical na perspektibo sa paggamit ng ML sa pagsusuri ng panahon.  

## Mga Gawain Bago Magbuo

Bago simulan ang pagbuo ng iyong modelo, may ilang mga gawain na kailangan mong tapusin. Upang masubukan ang iyong tanong at bumuo ng hypothesis batay sa mga prediksyon ng modelo, kailangan mong tukuyin at i-configure ang ilang elemento.

### Datos

Upang masagot ang iyong tanong nang may katiyakan, kailangan mo ng sapat na dami ng datos na may tamang uri. May dalawang bagay na kailangan mong gawin sa puntong ito:

- **Kolektahin ang datos**. Tandaan ang nakaraang aralin tungkol sa fairness sa data analysis, kolektahin ang iyong datos nang maingat. Maging aware sa mga pinagmulan ng datos na ito, anumang inherent biases na maaaring mayroon ito, at i-dokumento ang pinagmulan nito.
- **Ihanda ang datos**. May ilang hakbang sa proseso ng paghahanda ng datos. Maaaring kailanganin mong pagsamahin ang datos at i-normalize ito kung ito ay galing sa iba't ibang pinagmulan. Maaari mong pagandahin ang kalidad at dami ng datos sa pamamagitan ng iba't ibang paraan tulad ng pag-convert ng strings sa numbers (tulad ng ginagawa natin sa [Clustering](../../5-Clustering/1-Visualize/README.md)). Maaari ka ring bumuo ng bagong datos batay sa orihinal (tulad ng ginagawa natin sa [Classification](../../4-Classification/1-Introduction/README.md)). Maaari mong linisin at i-edit ang datos (tulad ng gagawin natin bago ang [Web App](../../3-Web-App/README.md) na aralin). Sa huli, maaaring kailanganin mo ring i-randomize at i-shuffle ito, depende sa iyong training techniques.

✅ Pagkatapos kolektahin at iproseso ang iyong datos, maglaan ng sandali upang tingnan kung ang hugis nito ay magpapahintulot sa iyo na tugunan ang iyong nilalayong tanong. Maaaring ang datos ay hindi mag-perform nang maayos sa iyong ibinigay na gawain, tulad ng natuklasan natin sa aming [Clustering](../../5-Clustering/1-Visualize/README.md) na mga aralin!

### Mga Feature at Target

Ang [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) ay isang masusukat na katangian ng iyong datos. Sa maraming datasets, ito ay ipinapahayag bilang isang column heading tulad ng 'date', 'size', o 'color'. Ang iyong feature variable, karaniwang kinakatawan bilang `X` sa code, ay kumakatawan sa input variable na gagamitin upang i-train ang modelo.

Ang target ay ang bagay na sinusubukan mong i-predict. Ang target, karaniwang kinakatawan bilang `y` sa code, ay kumakatawan sa sagot sa tanong na sinusubukan mong itanong sa iyong datos: sa Disyembre, anong **kulay** ng mga kalabasa ang magiging pinakamura? Sa San Francisco, anong mga kapitbahayan ang magkakaroon ng pinakamagandang presyo ng **real estate**? Minsan ang target ay tinutukoy din bilang label attribute.

### Pagpili ng iyong feature variable

🎓 **Feature Selection at Feature Extraction** Paano mo malalaman kung aling variable ang pipiliin kapag bumubuo ng modelo? Malamang na dadaan ka sa proseso ng feature selection o feature extraction upang piliin ang tamang mga variable para sa pinaka-performant na modelo. Hindi sila pareho: "Ang feature extraction ay lumilikha ng mga bagong feature mula sa mga function ng orihinal na mga feature, samantalang ang feature selection ay nagbabalik ng subset ng mga feature." ([source](https://wikipedia.org/wiki/Feature_selection))

### I-visualize ang iyong datos

Isang mahalagang aspeto ng toolkit ng data scientist ay ang kakayahang i-visualize ang datos gamit ang ilang magagaling na libraries tulad ng Seaborn o MatPlotLib. Ang pag-representa ng iyong datos nang biswal ay maaaring magbigay-daan sa iyo upang matuklasan ang mga nakatagong correlation na maaari mong magamit. Ang iyong mga visualization ay maaari ring makatulong sa iyo na matuklasan ang bias o hindi balanseng datos (tulad ng natuklasan natin sa [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Hatiin ang iyong dataset

Bago mag-training, kailangan mong hatiin ang iyong dataset sa dalawa o higit pang bahagi na may hindi pantay na laki ngunit mahusay na kumakatawan sa datos.

- **Training**. Ang bahaging ito ng dataset ay ginagamit upang i-fit ang iyong modelo para i-train ito. Ang set na ito ay bumubuo ng karamihan ng orihinal na dataset.
- **Testing**. Ang test dataset ay isang independiyenteng grupo ng datos, madalas na kinukuha mula sa orihinal na datos, na ginagamit mo upang kumpirmahin ang performance ng nabuo na modelo.
- **Validating**. Ang validation set ay isang mas maliit na independiyenteng grupo ng mga halimbawa na ginagamit mo upang i-tune ang hyperparameters o architecture ng modelo upang mapabuti ito. Depende sa laki ng iyong datos at sa tanong na iyong tinatanong, maaaring hindi mo kailangan bumuo ng pangatlong set (tulad ng nabanggit natin sa [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Pagbuo ng Modelo

Gamit ang iyong training data, ang layunin mo ay bumuo ng modelo, o isang statistical na representasyon ng iyong datos, gamit ang iba't ibang algorithm upang **i-train** ito. Ang pag-train ng modelo ay inilalantad ito sa datos at nagbibigay-daan dito upang gumawa ng mga assumption tungkol sa mga pattern na natuklasan, na-validate, at tinanggap o tinanggihan.

### Magpasya sa paraan ng training

Depende sa iyong tanong at sa likas na katangian ng iyong datos, pipili ka ng paraan upang i-train ito. Sa pamamagitan ng pagdaan sa [Scikit-learn's documentation](https://scikit-learn.org/stable/user_guide.html) - na ginagamit natin sa kursong ito - maaari mong tuklasin ang maraming paraan upang i-train ang modelo. Depende sa iyong karanasan, maaaring kailanganin mong subukan ang ilang iba't ibang paraan upang makabuo ng pinakamahusay na modelo. Malamang na dadaan ka sa proseso kung saan ang mga data scientist ay sinusuri ang performance ng modelo sa pamamagitan ng pagpapakain dito ng unseen data, pag-check ng accuracy, bias, at iba pang mga isyung nakakasira ng kalidad, at pagpili ng pinaka-angkop na paraan ng training para sa gawain.

### I-train ang modelo

Gamit ang iyong training data, handa ka nang 'i-fit' ito upang lumikha ng modelo. Mapapansin mo na sa maraming ML libraries, makikita mo ang code na 'model.fit' - sa oras na ito mo ipapadala ang iyong feature variable bilang isang array ng mga halaga (karaniwang 'X') at isang target variable (karaniwang 'y').

### Suriin ang modelo

Kapag natapos na ang proseso ng training (maaari itong tumagal ng maraming iterations, o 'epochs', upang i-train ang malaking modelo), magagawa mong suriin ang kalidad ng modelo sa pamamagitan ng paggamit ng test data upang sukatin ang performance nito. Ang datos na ito ay isang subset ng orihinal na datos na hindi pa nasuri ng modelo. Maaari kang mag-print ng isang table ng metrics tungkol sa kalidad ng iyong modelo.

🎓 **Model fitting**

Sa konteksto ng machine learning, ang model fitting ay tumutukoy sa katumpakan ng underlying function ng modelo habang sinusubukan nitong suriin ang datos na hindi nito pamilyar.

🎓 Ang **Underfitting** at **Overfitting** ay mga karaniwang problema na nakakasira sa kalidad ng modelo, kung saan ang modelo ay hindi sapat na mahusay o masyadong mahusay. Nagdudulot ito ng modelo na gumawa ng mga prediksyon na masyadong malapit o masyadong maluwag na nakahanay sa training data nito. Ang overfit na modelo ay masyadong mahusay sa pag-predict ng training data dahil natutunan nito ang mga detalye at ingay ng datos nang sobra. Ang underfit na modelo ay hindi tumpak dahil hindi nito maayos na masuri ang training data nito o ang datos na hindi pa nito 'nakikita'.

![overfitting model](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infographic ni [Jen Looper](https://twitter.com/jenlooper)

## Parameter tuning

Kapag natapos na ang iyong initial training, obserbahan ang kalidad ng modelo at isaalang-alang ang pagpapabuti nito sa pamamagitan ng pag-tweak ng 'hyperparameters' nito. Magbasa pa tungkol sa proseso [sa dokumentasyon](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediksyon

Ito ang sandali kung saan maaari mong gamitin ang ganap na bagong datos upang subukan ang katumpakan ng iyong modelo. Sa isang 'applied' ML setting, kung saan bumubuo ka ng mga web asset upang gamitin ang modelo sa production, maaaring kasangkot sa prosesong ito ang pagkolekta ng user input (halimbawa, isang pindot ng button) upang magtakda ng variable at ipadala ito sa modelo para sa inference o pagsusuri.

Sa mga araling ito, matutuklasan mo kung paano gamitin ang mga hakbang na ito upang maghanda, bumuo, mag-test, mag-evaluate, at mag-predict - lahat ng mga galaw ng isang data scientist at higit pa, habang ikaw ay umuusad sa iyong paglalakbay upang maging isang 'full stack' ML engineer.

---

## 🚀Hamunin

Gumuhit ng flow chart na nagpapakita ng mga hakbang ng isang ML practitioner. Nasaan ka ngayon sa proseso? Saan mo inaasahan na mahihirapan ka? Ano ang tila madali para sa iyo?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review at Pag-aaral sa Sarili

Maghanap online ng mga panayam sa mga data scientist na nag-uusap tungkol sa kanilang pang-araw-araw na trabaho. Narito ang [isa](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Takdang Aralin

[Magpanayam ng isang data scientist](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na pinagmulan. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.