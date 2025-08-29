<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6396d5d8617572cd2ac1de74fb0deb22",
  "translation_date": "2025-08-29T14:34:05+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "tl"
}
-->
# Pagsasalin at pagsusuri ng damdamin gamit ang ML

Sa mga nakaraang aralin, natutunan mo kung paano gumawa ng isang simpleng bot gamit ang `TextBlob`, isang library na gumagamit ng ML sa likod ng eksena upang magsagawa ng mga pangunahing gawain sa NLP tulad ng pagkuha ng mga parirala ng pangngalan. Isa pang mahalagang hamon sa computational linguistics ay ang tumpak na _pagsasalin_ ng isang pangungusap mula sa isang wika patungo sa isa pa.

## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

Ang pagsasalin ay isang napakahirap na problema dahil may libu-libong wika at bawat isa ay may iba't ibang mga patakaran sa gramatika. Isang paraan ay ang pag-convert ng mga pormal na patakaran ng gramatika ng isang wika, tulad ng Ingles, sa isang istrukturang hindi nakadepende sa wika, at pagkatapos ay isalin ito sa pamamagitan ng pag-convert pabalik sa ibang wika. Ang pamamaraang ito ay nangangahulugan na gagawin mo ang mga sumusunod na hakbang:

1. **Pagkilala**. Tukuyin o i-tag ang mga salita sa input na wika bilang mga pangngalan, pandiwa, atbp.
2. **Gumawa ng pagsasalin**. Gumawa ng direktang pagsasalin ng bawat salita sa format ng target na wika.

### Halimbawa ng pangungusap, Ingles sa Irish

Sa 'Ingles', ang pangungusap na _I feel happy_ ay binubuo ng tatlong salita sa pagkakasunod na:

- **paksa** (I)
- **pandiwa** (feel)
- **pang-uri** (happy)

Gayunpaman, sa wikang 'Irish', ang parehong pangungusap ay may ibang istrukturang gramatikal - ang mga damdamin tulad ng "*happy*" o "*sad*" ay ipinapahayag bilang *nasa ibabaw mo*.

Ang pariralang Ingles na `I feel happy` sa Irish ay magiging `Tá athas orm`. Ang isang *literal* na pagsasalin ay magiging `Happy is upon me`.

Ang isang nagsasalita ng Irish na nagsasalin sa Ingles ay magsasabi ng `I feel happy`, hindi `Happy is upon me`, dahil nauunawaan nila ang kahulugan ng pangungusap, kahit na magkaiba ang mga salita at istruktura ng pangungusap.

Ang pormal na pagkakasunod ng pangungusap sa Irish ay:

- **pandiwa** (Tá o is)
- **pang-uri** (athas, o happy)
- **paksa** (orm, o upon me)

## Pagsasalin

Ang isang simpleng programa sa pagsasalin ay maaaring magsalin lamang ng mga salita, hindi isinasaalang-alang ang istruktura ng pangungusap.

✅ Kung natutunan mo ang pangalawa (o pangatlo o higit pa) na wika bilang isang adulto, maaaring nagsimula ka sa pag-iisip sa iyong katutubong wika, isinalin ang isang konsepto nang salita-sa-salita sa iyong isipan sa pangalawang wika, at pagkatapos ay binibigkas ang iyong pagsasalin. Katulad ito ng ginagawa ng mga simpleng programa sa pagsasalin. Mahalagang malampasan ang yugtong ito upang makamit ang kahusayan!

Ang simpleng pagsasalin ay nagreresulta sa mga maling (at kung minsan ay nakakatawang) pagsasalin: `I feel happy` ay literal na isinasalin sa `Mise bhraitheann athas` sa Irish. Ang ibig sabihin nito (literal) ay `me feel happy` at hindi ito wastong pangungusap sa Irish. Kahit na ang Ingles at Irish ay mga wikang sinasalita sa dalawang magkalapit na isla, sila ay napakaibang mga wika na may iba't ibang istruktura ng gramatika.

> Maaari kang manood ng ilang mga video tungkol sa mga tradisyong lingguwistiko ng Irish tulad ng [ito](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Mga pamamaraan ng machine learning

Sa ngayon, natutunan mo ang tungkol sa pormal na mga patakaran sa natural language processing. Isa pang paraan ay ang huwag pansinin ang kahulugan ng mga salita, at _sa halip ay gumamit ng machine learning upang makita ang mga pattern_. Maaari itong gumana sa pagsasalin kung mayroon kang maraming teksto (isang *corpus*) o mga teksto (*corpora*) sa parehong pinagmulan at target na mga wika.

Halimbawa, isaalang-alang ang kaso ng *Pride and Prejudice*, isang kilalang nobelang Ingles na isinulat ni Jane Austen noong 1813. Kung susuriin mo ang libro sa Ingles at isang pagsasalin ng tao ng libro sa *Pranses*, maaari mong makita ang mga parirala sa isa na _idiomatically_ isinalin sa isa pa. Gagawin mo iyon sa ilang sandali.

Halimbawa, kapag ang isang pariralang Ingles tulad ng `I have no money` ay literal na isinalin sa Pranses, maaaring maging `Je n'ai pas de monnaie`. Ang "Monnaie" ay isang mapanlinlang na 'false cognate' sa Pranses, dahil ang 'money' at 'monnaie' ay hindi magkapareho. Ang mas mahusay na pagsasalin na maaaring gawin ng isang tao ay `Je n'ai pas d'argent`, dahil mas mahusay nitong naipapahayag ang kahulugan na wala kang pera (sa halip na 'loose change' na siyang kahulugan ng 'monnaie').

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.tl.png)

> Larawan ni [Jen Looper](https://twitter.com/jenlooper)

Kung ang isang modelo ng ML ay may sapat na pagsasalin ng tao upang makabuo ng isang modelo, maaari nitong mapabuti ang katumpakan ng mga pagsasalin sa pamamagitan ng pagtukoy ng mga karaniwang pattern sa mga tekstong dati nang isinalin ng mga dalubhasang tao na nagsasalita ng parehong wika.

### Ehersisyo - pagsasalin

Maaari mong gamitin ang `TextBlob` upang magsalin ng mga pangungusap. Subukan ang sikat na unang linya ng **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

Ang `TextBlob` ay gumagawa ng medyo mahusay na pagsasalin: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Maaaring sabihin na ang pagsasalin ng TextBlob ay mas eksakto, sa katunayan, kaysa sa pagsasalin noong 1932 sa Pranses ng libro nina V. Leconte at Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Sa kasong ito, ang pagsasalin na pinapagana ng ML ay gumagawa ng mas mahusay na trabaho kaysa sa tagasalin ng tao na hindi kinakailangang nagdadagdag ng mga salita sa bibig ng orihinal na may-akda para sa 'kalinawan'.

> Ano ang nangyayari dito? At bakit napakahusay ng TextBlob sa pagsasalin? Sa likod ng eksena, gumagamit ito ng Google translate, isang sopistikadong AI na kayang mag-parse ng milyun-milyong parirala upang mahulaan ang pinakamahusay na mga string para sa gawain. Walang manwal na nagaganap dito at kailangan mo ng koneksyon sa internet upang magamit ang `blob.translate`.

✅ Subukan ang ilang higit pang mga pangungusap. Alin ang mas mahusay, ML o pagsasalin ng tao? Sa anong mga kaso?

## Pagsusuri ng damdamin

Isa pang lugar kung saan mahusay na gumagana ang machine learning ay ang pagsusuri ng damdamin. Ang isang hindi ML na paraan sa damdamin ay ang tukuyin ang mga salita at parirala na 'positibo' at 'negatibo'. Pagkatapos, sa isang bagong piraso ng teksto, kalkulahin ang kabuuang halaga ng mga positibo, negatibo, at neutral na salita upang matukoy ang pangkalahatang damdamin.

Ang pamamaraang ito ay madaling malinlang tulad ng maaaring nakita mo sa gawain ni Marvin - ang pangungusap na `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` ay isang sarcastic, negatibong pangungusap, ngunit ang simpleng algorithm ay natutukoy ang 'great', 'wonderful', 'glad' bilang positibo at 'waste', 'lost' at 'dark' bilang negatibo. Ang pangkalahatang damdamin ay naiimpluwensyahan ng mga salungat na salitang ito.

✅ Huminto sandali at isipin kung paano natin ipinapahayag ang pangungutya bilang mga tao. Ang tono ng boses ay may malaking papel. Subukang sabihin ang pariralang "Well, that film was awesome" sa iba't ibang paraan upang matuklasan kung paano ipinapahayag ng iyong boses ang kahulugan.

### Mga pamamaraan ng ML

Ang pamamaraang ML ay manu-manong mangolekta ng mga negatibo at positibong teksto - mga tweet, o mga pagsusuri ng pelikula, o anumang bagay kung saan ang tao ay nagbigay ng marka *at* isang nakasulat na opinyon. Pagkatapos, maaaring ilapat ang mga pamamaraan ng NLP sa mga opinyon at marka, upang lumitaw ang mga pattern (halimbawa, ang mga positibong pagsusuri ng pelikula ay may posibilidad na magkaroon ng pariralang 'Oscar worthy' kaysa sa mga negatibong pagsusuri ng pelikula, o ang mga positibong pagsusuri sa restawran ay nagsasabing 'gourmet' nang higit kaysa sa 'disgusting').

> ⚖️ **Halimbawa**: Kung nagtatrabaho ka sa opisina ng isang politiko at may bagong batas na pinagtatalunan, maaaring sumulat ang mga nasasakupan ng mga email na sumusuporta o tumututol sa partikular na bagong batas. Sabihin nating ikaw ay inatasang basahin ang mga email at ayusin ang mga ito sa 2 tambak, *pabor* at *laban*. Kung maraming email, maaaring ma-overwhelm ka sa pagtatangkang basahin ang lahat. Hindi ba't mas maganda kung ang isang bot ang magbabasa ng lahat para sa iyo, nauunawaan ang mga ito, at sasabihin sa iyo kung saang tambak kabilang ang bawat email? 
> 
> Isang paraan upang makamit iyon ay ang paggamit ng Machine Learning. Sanayin mo ang modelo gamit ang bahagi ng mga email na *laban* at bahagi ng mga email na *pabor*. Ang modelo ay may posibilidad na iugnay ang mga parirala at salita sa panig na laban at panig na pabor, *ngunit hindi nito maiintindihan ang anumang nilalaman*, tanging ang ilang mga salita at pattern ay mas malamang na lumitaw sa isang email na *laban* o *pabor*. Maaari mo itong subukan gamit ang ilang mga email na hindi mo ginamit upang sanayin ang modelo, at tingnan kung pareho ang konklusyon nito sa iyo. Pagkatapos, kapag nasiyahan ka na sa katumpakan ng modelo, maaari mong iproseso ang mga email sa hinaharap nang hindi kinakailangang basahin ang bawat isa.

✅ Ang prosesong ito ba ay parang mga prosesong ginamit mo sa mga nakaraang aralin?

## Ehersisyo - mga pangungusap na may damdamin

Ang damdamin ay sinusukat gamit ang *polarity* mula -1 hanggang 1, kung saan ang -1 ay ang pinaka-negatibong damdamin, at 1 ang pinaka-positibo. Ang damdamin ay sinusukat din gamit ang iskor mula 0 - 1 para sa objectivity (0) at subjectivity (1).

Balikan ang *Pride and Prejudice* ni Jane Austen. Ang teksto ay makukuha dito sa [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Ang halimbawa sa ibaba ay nagpapakita ng isang maikling programa na sumusuri sa damdamin ng una at huling mga pangungusap mula sa libro at ipinapakita ang polarity ng damdamin at iskor ng subjectivity/objectivity nito.

Dapat mong gamitin ang library na `TextBlob` (inilarawan sa itaas) upang matukoy ang `sentiment` (hindi mo kailangang gumawa ng sarili mong calculator ng damdamin) sa sumusunod na gawain.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Makikita mo ang sumusunod na output:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Hamon - suriin ang polarity ng damdamin

Ang iyong gawain ay tukuyin, gamit ang polarity ng damdamin, kung ang *Pride and Prejudice* ay may mas maraming ganap na positibong pangungusap kaysa sa ganap na negatibo. Para sa gawaing ito, maaari mong ipalagay na ang iskor ng polarity na 1 o -1 ay ganap na positibo o negatibo ayon sa pagkakabanggit.

**Mga Hakbang:**

1. I-download ang isang [kopya ng Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) mula sa Project Gutenberg bilang isang .txt file. Alisin ang metadata sa simula at dulo ng file, iwanan lamang ang orihinal na teksto
2. Buksan ang file sa Python at kunin ang nilalaman bilang isang string
3. Gumawa ng TextBlob gamit ang string ng libro
4. Suriin ang bawat pangungusap sa libro sa isang loop
   1. Kung ang polarity ay 1 o -1, itabi ang pangungusap sa isang array o listahan ng mga positibo o negatibong mensahe
5. Sa dulo, i-print ang lahat ng positibong pangungusap at negatibong pangungusap (hiwalay) at ang bilang ng bawat isa.

Narito ang isang sample na [solusyon](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Kaalaman Check

1. Ang damdamin ay batay sa mga salitang ginamit sa pangungusap, ngunit naiintindihan ba ng code ang mga salita?
2. Sa tingin mo ba ang polarity ng damdamin ay tumpak, o sa madaling salita, sumasang-ayon ka ba sa mga iskor?
   1. Sa partikular, sumasang-ayon o hindi ka ba sumasang-ayon sa ganap na **positibong** polarity ng mga sumusunod na pangungusap?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Ang susunod na 3 pangungusap ay naitala na may ganap na positibong damdamin, ngunit sa masusing pagbabasa, hindi sila positibong pangungusap. Bakit sa tingin mo ang pagsusuri ng damdamin ay inakala nilang positibo ang mga pangungusap?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Sumasang-ayon o hindi ka ba sumasang-ayon sa ganap na **negatibong** polarity ng mga sumusunod na pangungusap?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Ang sinumang tagahanga ni Jane Austen ay mauunawaan na madalas niyang ginagamit ang kanyang mga libro upang punahin ang mas katawa-tawang aspeto ng lipunang English Regency. Si Elizabeth Bennett, ang pangunahing tauhan sa *Pride and Prejudice*, ay isang mahusay na tagamasid sa lipunan (tulad ng may-akda) at ang kanyang wika ay madalas na puno ng mga pahiwatig. Kahit si Mr. Darcy (ang love interest sa kwento) ay napansin ang mapaglaro at mapanuksong paggamit ni Elizabeth ng wika: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Hamon

Paano mo mapapahusay si Marvin sa pamamagitan ng pagkuha ng iba pang mga tampok mula sa input ng user?

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## Pagsusuri at Pag-aaral ng Sarili
Maraming paraan upang makuha ang damdamin mula sa teksto. Isipin ang mga aplikasyon sa negosyo na maaaring gumamit ng teknik na ito. Pag-isipan din kung paano ito maaaring magkamali. Magbasa pa tungkol sa mga sopistikadong sistemang handa para sa negosyo na nag-a-analyze ng damdamin tulad ng [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Subukan ang ilan sa mga pangungusap mula sa Pride and Prejudice sa itaas at tingnan kung kaya nitong matukoy ang mga maseselang detalye.

## Takdang-Aralin

[Lisensyang Pampanitikan](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.