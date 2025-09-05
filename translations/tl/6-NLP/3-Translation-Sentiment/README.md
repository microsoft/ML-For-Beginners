<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T18:27:19+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "tl"
}
-->
# Pagsasalin at Sentiment Analysis gamit ang ML

Sa mga nakaraang aralin, natutunan mo kung paano bumuo ng isang simpleng bot gamit ang `TextBlob`, isang library na gumagamit ng ML sa likod ng eksena upang magsagawa ng mga pangunahing gawain sa NLP tulad ng pagkuha ng mga parirala ng pangngalan. Isa pang mahalagang hamon sa computational linguistics ay ang tumpak na _pagsasalin_ ng isang pangungusap mula sa isang sinasalita o nakasulat na wika patungo sa isa pa.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Ang pagsasalin ay isang napakahirap na problema dahil sa libu-libong wika na may iba't ibang mga panuntunan sa gramatika. Isang paraan ay ang pag-convert ng mga pormal na panuntunan sa gramatika ng isang wika, tulad ng Ingles, sa isang istrukturang hindi nakadepende sa wika, at pagkatapos ay isalin ito sa pamamagitan ng pag-convert pabalik sa ibang wika. Ang pamamaraang ito ay nangangahulugan na gagawin mo ang mga sumusunod na hakbang:

1. **Pagkilala**. Tukuyin o i-tag ang mga salita sa input na wika bilang mga pangngalan, pandiwa, atbp.
2. **Gumawa ng pagsasalin**. Gumawa ng direktang pagsasalin ng bawat salita sa format ng target na wika.

### Halimbawa ng pangungusap, Ingles sa Irish

Sa 'Ingles', ang pangungusap na _I feel happy_ ay binubuo ng tatlong salita sa ganitong pagkakasunod-sunod:

- **subject** (I)
- **verb** (feel)
- **adjective** (happy)

Gayunpaman, sa wikang 'Irish', ang parehong pangungusap ay may ibang istrukturang gramatikal - ang mga damdamin tulad ng "*happy*" o "*sad*" ay ipinapahayag bilang *nasa ibabaw mo*.

Ang pariralang Ingles na `I feel happy` sa Irish ay magiging `Tá athas orm`. Ang *literal* na pagsasalin ay magiging `Happy is upon me`.

Ang isang nagsasalita ng Irish na nagsasalin sa Ingles ay magsasabi ng `I feel happy`, hindi `Happy is upon me`, dahil nauunawaan nila ang kahulugan ng pangungusap, kahit na magkaiba ang mga salita at istruktura ng pangungusap.

Ang pormal na pagkakasunod-sunod ng pangungusap sa Irish ay:

- **verb** (Tá o is)
- **adjective** (athas, o happy)
- **subject** (orm, o upon me)

## Pagsasalin

Ang isang simpleng programa sa pagsasalin ay maaaring magsalin lamang ng mga salita, na hindi isinasaalang-alang ang istruktura ng pangungusap.

✅ Kung natutunan mo ang pangalawa (o pangatlo o higit pa) na wika bilang isang adulto, maaaring nagsimula ka sa pag-iisip sa iyong katutubong wika, isinalin ang isang konsepto nang salita-sa-salita sa iyong isipan patungo sa pangalawang wika, at pagkatapos ay binibigkas ang iyong pagsasalin. Katulad ito ng ginagawa ng mga simpleng programa sa pagsasalin ng computer. Mahalagang malampasan ang yugtong ito upang makamit ang kasanayan!

Ang simpleng pagsasalin ay nagdudulot ng masama (at kung minsan nakakatawang) mga maling pagsasalin: `I feel happy` ay literal na isinasalin sa `Mise bhraitheann athas` sa Irish. Ang ibig sabihin nito (literal) ay `me feel happy` at hindi ito wastong pangungusap sa Irish. Kahit na ang Ingles at Irish ay mga wikang sinasalita sa dalawang magkalapit na isla, sila ay napaka-magkaibang wika na may iba't ibang istruktura ng gramatika.

> Maaari kang manood ng ilang mga video tungkol sa tradisyon ng lingguwistika ng Irish tulad ng [ito](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Mga Pamamaraan ng Machine Learning

Sa ngayon, natutunan mo ang tungkol sa pormal na pamamaraan sa natural language processing. Isa pang paraan ay ang huwag pansinin ang kahulugan ng mga salita, at _sa halip ay gumamit ng machine learning upang matukoy ang mga pattern_. Maaari itong gumana sa pagsasalin kung mayroon kang maraming teksto (isang *corpus*) o mga teksto (*corpora*) sa parehong pinagmulan at target na wika.

Halimbawa, isaalang-alang ang kaso ng *Pride and Prejudice*, isang kilalang nobelang Ingles na isinulat ni Jane Austen noong 1813. Kung susuriin mo ang libro sa Ingles at isang pagsasalin ng tao ng libro sa *French*, maaari mong matukoy ang mga parirala sa isa na _idiomatically_ na isinalin sa isa pa. Gagawin mo iyon sa ilang sandali.

Halimbawa, kapag ang pariralang Ingles tulad ng `I have no money` ay literal na isinalin sa French, maaaring maging `Je n'ai pas de monnaie`. Ang "Monnaie" ay isang mapanlinlang na 'false cognate' sa French, dahil ang 'money' at 'monnaie' ay hindi magkatulad. Ang mas mahusay na pagsasalin na maaaring gawin ng tao ay `Je n'ai pas d'argent`, dahil mas mahusay nitong naipapahayag ang kahulugan na wala kang pera (sa halip na 'loose change' na siyang kahulugan ng 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Larawan ni [Jen Looper](https://twitter.com/jenlooper)

Kung ang isang ML model ay may sapat na pagsasalin ng tao upang bumuo ng isang modelo, maaari nitong mapabuti ang katumpakan ng mga pagsasalin sa pamamagitan ng pagtukoy ng mga karaniwang pattern sa mga teksto na dati nang isinalin ng mga dalubhasang nagsasalita ng parehong wika.

### Ehersisyo - Pagsasalin

Maaari mong gamitin ang `TextBlob` upang magsalin ng mga pangungusap. Subukan ang sikat na unang linya ng **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

Ang `TextBlob` ay gumagawa ng medyo mahusay na pagsasalin: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Maaaring sabihin na ang pagsasalin ng TextBlob ay mas eksakto, sa katunayan, kaysa sa 1932 French translation ng libro nina V. Leconte at Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Sa kasong ito, ang pagsasalin na pinamamahalaan ng ML ay mas mahusay kaysa sa tagasalin ng tao na hindi kinakailangang nagdagdag ng mga salita sa bibig ng orihinal na may-akda para sa 'kalinawan'.

> Ano ang nangyayari dito? At bakit napakahusay ng TextBlob sa pagsasalin? Sa likod ng eksena, ginagamit nito ang Google translate, isang sopistikadong AI na kayang mag-parse ng milyun-milyong parirala upang mahulaan ang pinakamahusay na mga string para sa gawain. Walang manu-manong nangyayari dito at kailangan mo ng koneksyon sa internet upang magamit ang `blob.translate`.

✅ Subukan ang ilang pangungusap pa. Alin ang mas mahusay, ML o pagsasalin ng tao? Sa anong mga kaso?

## Sentiment Analysis

Isa pang lugar kung saan mahusay na gumagana ang machine learning ay ang sentiment analysis. Ang isang non-ML na paraan sa sentiment ay ang tukuyin ang mga salita at parirala na 'positibo' at 'negatibo'. Pagkatapos, sa isang bagong piraso ng teksto, kalkulahin ang kabuuang halaga ng positibo, negatibo, at neutral na mga salita upang matukoy ang pangkalahatang damdamin.

Ang pamamaraang ito ay madaling malinlang tulad ng nakita mo sa Marvin task - ang pangungusap na `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` ay isang sarcastic, negatibong damdamin na pangungusap, ngunit ang simpleng algorithm ay nakikita ang 'great', 'wonderful', 'glad' bilang positibo at 'waste', 'lost' at 'dark' bilang negatibo. Ang pangkalahatang damdamin ay naiimpluwensyahan ng mga salungat na salitang ito.

✅ Huminto sandali at isipin kung paano natin ipinapahayag ang sarcasm bilang mga nagsasalita ng tao. Ang tono ng boses ay may malaking papel. Subukang sabihin ang pariralang "Well, that film was awesome" sa iba't ibang paraan upang matuklasan kung paano ipinapahayag ng iyong boses ang kahulugan.

### Mga Pamamaraan ng ML

Ang pamamaraang ML ay manu-manong mangolekta ng negatibo at positibong mga teksto - tweets, o mga review ng pelikula, o anumang bagay kung saan ang tao ay nagbigay ng score *at* ng nakasulat na opinyon. Pagkatapos ay maaaring ilapat ang mga teknik ng NLP sa mga opinyon at score, upang lumitaw ang mga pattern (halimbawa, ang mga positibong review ng pelikula ay may tendensiyang magkaroon ng pariralang 'Oscar worthy' kaysa sa mga negatibong review ng pelikula, o ang mga positibong review ng restaurant ay nagsasabing 'gourmet' nang mas madalas kaysa sa 'disgusting').

> ⚖️ **Halimbawa**: Kung nagtatrabaho ka sa opisina ng isang politiko at may bagong batas na pinagtatalunan, maaaring magsulat ang mga constituent ng mga email na sumusuporta o laban sa partikular na bagong batas. Sabihin nating ikaw ang naatasang magbasa ng mga email at ayusin ang mga ito sa 2 tambak, *para* at *laban*. Kung maraming email, maaaring ma-overwhelm ka sa pagbabasa ng lahat ng ito. Hindi ba't mas maganda kung may bot na makakabasa ng lahat ng ito para sa iyo, mauunawaan ang mga ito, at sasabihin kung saang tambak dapat mapunta ang bawat email? 
> 
> Isang paraan upang makamit iyon ay ang paggamit ng Machine Learning. Sanayin mo ang modelo gamit ang bahagi ng mga *laban* na email at bahagi ng mga *para* na email. Ang modelo ay may tendensiyang iugnay ang mga parirala at salita sa panig ng laban at panig ng para, *ngunit hindi nito mauunawaan ang anumang nilalaman*, kundi ang ilang mga salita at pattern ay mas malamang na lumitaw sa isang *laban* o *para* na email. Maaari mo itong subukan gamit ang ilang email na hindi mo ginamit upang sanayin ang modelo, at tingnan kung pareho ang konklusyon nito sa iyo. Pagkatapos, kapag nasiyahan ka sa katumpakan ng modelo, maaari mong iproseso ang mga email sa hinaharap nang hindi kinakailangang basahin ang bawat isa.

✅ Ang prosesong ito ba ay katulad ng mga prosesong ginamit mo sa mga nakaraang aralin?

## Ehersisyo - Sentimental na Pangungusap

Ang damdamin ay sinusukat gamit ang *polarity* mula -1 hanggang 1, ibig sabihin ang -1 ay ang pinaka-negatibong damdamin, at ang 1 ay ang pinaka-positibo. Ang damdamin ay sinusukat din gamit ang score mula 0 - 1 para sa objectivity (0) at subjectivity (1).

Balikan ang *Pride and Prejudice* ni Jane Austen. Ang teksto ay makukuha dito sa [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Ang sample sa ibaba ay nagpapakita ng isang maikling programa na nag-a-analisa ng damdamin ng unang at huling mga pangungusap mula sa libro at ipinapakita ang sentiment polarity at subjectivity/objectivity score nito.

Dapat mong gamitin ang library na `TextBlob` (inilarawan sa itaas) upang matukoy ang `sentiment` (hindi mo kailangang magsulat ng sarili mong sentiment calculator) sa sumusunod na gawain.

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

## Hamon - Suriin ang Sentiment Polarity

Ang iyong gawain ay tukuyin, gamit ang sentiment polarity, kung ang *Pride and Prejudice* ay may mas maraming ganap na positibong pangungusap kaysa sa ganap na negatibo. Para sa gawaing ito, maaari mong ipalagay na ang polarity score na 1 o -1 ay ganap na positibo o negatibo ayon sa pagkakabanggit.

**Mga Hakbang:**

1. I-download ang [kopya ng Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) mula sa Project Gutenberg bilang isang .txt file. Alisin ang metadata sa simula at dulo ng file, iwan lamang ang orihinal na teksto
2. Buksan ang file sa Python at kunin ang nilalaman bilang isang string
3. Gumawa ng TextBlob gamit ang string ng libro
4. I-analisa ang bawat pangungusap sa libro sa isang loop
   1. Kung ang polarity ay 1 o -1, itago ang pangungusap sa isang array o listahan ng positibo o negatibong mga mensahe
5. Sa dulo, i-print ang lahat ng positibong pangungusap at negatibong pangungusap (hiwalay) at ang bilang ng bawat isa.

Narito ang isang sample na [solusyon](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Knowledge Check

1. Ang damdamin ay batay sa mga salitang ginamit sa pangungusap, ngunit nauunawaan ba ng code ang mga salita?
2. Sa tingin mo ba ang sentiment polarity ay tumpak, o sa madaling salita, sumasang-ayon ka ba sa mga score?
   1. Partikular, sumasang-ayon ka ba o hindi sa ganap na **positibong** polarity ng mga sumusunod na pangungusap?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Ang susunod na 3 pangungusap ay na-score na may ganap na positibong damdamin, ngunit sa mas malapit na pagbabasa, hindi sila positibong pangungusap. Bakit sa tingin mo ang sentiment analysis ay naisip na positibo ang mga pangungusap na ito?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Sumasang-ayon ka ba o hindi sa ganap na **negatibong** polarity ng mga sumusunod na pangungusap?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Ang sinumang tagahanga ni Jane Austen ay mauunawaan na madalas niyang ginagamit ang kanyang mga libro upang punahin ang mas katawa-tawang aspeto ng English Regency society. Si Elizabeth Bennett, ang pangunahing tauhan sa *Pride and Prejudice*, ay isang mahusay na tagamasid sa lipunan (tulad ng may-akda) at ang kanyang wika ay madalas na puno ng nuance. Kahit si Mr. Darcy (ang love interest sa kuwento) ay napansin ang mapaglaro at mapanuksong paggamit ni Elizabeth ng wika: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Hamon

Paano mo mapapabuti si Marvin sa pamamagitan ng pagkuha ng iba pang mga tampok mula sa input ng user?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study
Maraming paraan upang makuha ang damdamin mula sa teksto. Isipin ang mga aplikasyon sa negosyo na maaaring gumamit ng teknik na ito. Isipin kung paano ito maaaring magkamali. Magbasa pa tungkol sa mga sopistikadong sistema na handa para sa enterprise na nag-a-analyze ng damdamin tulad ng [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Subukan ang ilan sa mga pangungusap mula sa Pride and Prejudice sa itaas at tingnan kung kaya nitong matukoy ang mga masalimuot na damdamin.

## Takdang-Aralin

[Poetic license](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.