<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T16:02:46+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "sw"
}
-->
# Mbinu za Kujifunza Mashine

Mchakato wa kujenga, kutumia, na kudumisha mifano ya kujifunza mashine na data wanayotumia ni tofauti sana na mchakato wa maendeleo mengine. Katika somo hili, tutafafanua mchakato huo na kuelezea mbinu kuu unazohitaji kujua. Utajifunza:

- Kuelewa michakato inayosimamia kujifunza mashine kwa kiwango cha juu.
- Kuchunguza dhana za msingi kama 'mifano', 'utabiri', na 'data ya mafunzo'.

## [Jaribio la awali la somo](https://ff-quizzes.netlify.app/en/ml/)

[![ML kwa wanaoanza - Mbinu za Kujifunza Mashine](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML kwa wanaoanza - Mbinu za Kujifunza Mashine")

> 🎥 Bofya picha hapo juu kwa video fupi inayopitia somo hili.

## Utangulizi

Kwa kiwango cha juu, sanaa ya kuunda michakato ya kujifunza mashine (ML) inajumuisha hatua kadhaa:

1. **Amua swali**. Mchakato mwingi wa ML huanza kwa kuuliza swali ambalo haliwezi kujibiwa kwa programu rahisi ya masharti au injini ya sheria. Maswali haya mara nyingi huzunguka utabiri kulingana na mkusanyiko wa data.
2. **Kusanya na kuandaa data**. Ili uweze kujibu swali lako, unahitaji data. Ubora na, wakati mwingine, wingi wa data yako utaamua jinsi unavyoweza kujibu swali lako la awali. Kuonyesha data ni kipengele muhimu cha awamu hii. Awamu hii pia inajumuisha kugawanya data katika kikundi cha mafunzo na majaribio ili kujenga mfano.
3. **Chagua mbinu ya mafunzo**. Kulingana na swali lako na asili ya data yako, unahitaji kuchagua jinsi unavyotaka kufundisha mfano ili kuakisi data yako vizuri na kufanya utabiri sahihi dhidi yake. Hii ni sehemu ya mchakato wa ML inayohitaji utaalamu maalum na, mara nyingi, majaribio mengi.
4. **Fanya mafunzo ya mfano**. Kwa kutumia data yako ya mafunzo, utatumia algoriti mbalimbali kufundisha mfano kutambua mifumo katika data. Mfano unaweza kutumia uzito wa ndani ambao unaweza kubadilishwa ili kuzingatia sehemu fulani za data kuliko nyingine ili kujenga mfano bora.
5. **Tathmini mfano**. Unatumia data ambayo haijawahi kuonekana (data yako ya majaribio) kutoka seti yako iliyokusanywa ili kuona jinsi mfano unavyofanya kazi.
6. **Kuboresha vigezo**. Kulingana na utendaji wa mfano wako, unaweza kurudia mchakato kwa kutumia vigezo tofauti, au mabadiliko, yanayodhibiti tabia ya algoriti zinazotumika kufundisha mfano.
7. **Tabiri**. Tumia pembejeo mpya kujaribu usahihi wa mfano wako.

## Swali la kuuliza

Kompyuta zina ujuzi wa kipekee wa kugundua mifumo iliyofichwa katika data. Uwezo huu ni muhimu sana kwa watafiti wenye maswali kuhusu uwanja fulani ambayo hayawezi kujibiwa kwa urahisi kwa kuunda injini ya sheria za masharti. Kwa mfano wa kazi ya actuarial, mwanasayansi wa data anaweza kuunda sheria za mikono kuhusu vifo vya wavutaji sigara dhidi ya wasiovuta sigara.

Hata hivyo, wakati vigezo vingine vingi vinapojumuishwa katika hesabu, mfano wa ML unaweza kuwa bora zaidi katika kutabiri viwango vya vifo vya baadaye kulingana na historia ya afya ya zamani. Mfano wa kufurahisha zaidi unaweza kuwa kutabiri hali ya hewa kwa mwezi wa Aprili katika eneo fulani kulingana na data inayojumuisha latitudo, longitudo, mabadiliko ya hali ya hewa, ukaribu na bahari, mifumo ya mkondo wa ndege, na zaidi.

✅ Hii [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) kuhusu mifano ya hali ya hewa inatoa mtazamo wa kihistoria wa kutumia ML katika uchambuzi wa hali ya hewa.  

## Kazi za kabla ya kujenga

Kabla ya kuanza kujenga mfano wako, kuna kazi kadhaa unazohitaji kukamilisha. Ili kujaribu swali lako na kuunda dhana kulingana na utabiri wa mfano, unahitaji kutambua na kusanidi vipengele kadhaa.

### Data

Ili uweze kujibu swali lako kwa uhakika wowote, unahitaji kiasi kizuri cha data ya aina sahihi. Kuna mambo mawili unayohitaji kufanya katika hatua hii:

- **Kusanya data**. Ukizingatia somo la awali kuhusu usawa katika uchambuzi wa data, kusanya data yako kwa uangalifu. Kuwa na ufahamu wa vyanzo vya data hii, upendeleo wowote wa asili ambao inaweza kuwa nao, na andika asili yake.
- **Andaa data**. Kuna hatua kadhaa katika mchakato wa kuandaa data. Unaweza kuhitaji kuunganisha data na kuifanya kuwa ya kawaida ikiwa inatoka kwa vyanzo tofauti. Unaweza kuboresha ubora na wingi wa data kupitia mbinu mbalimbali kama kubadilisha maandishi kuwa namba (kama tunavyofanya katika [Clustering](../../5-Clustering/1-Visualize/README.md)). Unaweza pia kuzalisha data mpya, kulingana na ya awali (kama tunavyofanya katika [Classification](../../4-Classification/1-Introduction/README.md)). Unaweza kusafisha na kuhariri data (kama tutakavyofanya kabla ya somo la [Web App](../../3-Web-App/README.md)). Hatimaye, unaweza pia kuhitaji kuipangilia upya na kuichanganya, kulingana na mbinu zako za mafunzo.

✅ Baada ya kukusanya na kuchakata data yako, chukua muda kuona kama umbo lake litaruhusu kushughulikia swali lako lililokusudiwa. Inawezekana kwamba data haitafanya vizuri katika kazi yako uliyopewa, kama tunavyogundua katika masomo yetu ya [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Vipengele na Lengo

[Feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) ni mali inayoweza kupimika ya data yako. Katika seti nyingi za data, huonyeshwa kama kichwa cha safu kama 'tarehe', 'ukubwa', au 'rangi'. Kigezo cha kipengele, ambacho kwa kawaida huwakilishwa kama `X` katika msimbo, kinawakilisha kigezo cha pembejeo ambacho kitatumika kufundisha mfano.

Lengo ni kitu unachojaribu kutabiri. Lengo, ambalo kwa kawaida huwakilishwa kama `y` katika msimbo, linawakilisha jibu la swali unalojaribu kuuliza kuhusu data yako: katika Desemba, malenge ya **rangi** gani yatakuwa ya bei rahisi? Katika San Francisco, vitongoji gani vitakuwa na **bei** bora ya mali isiyohamishika? Wakati mwingine lengo pia hujulikana kama sifa ya lebo.

### Kuchagua kigezo cha kipengele

🎓 **Uchaguzi wa Kipengele na Uchimbaji wa Kipengele** Unajuaje ni kigezo gani cha kuchagua wakati wa kujenga mfano? Huenda ukapitia mchakato wa kuchagua kipengele au kuchimba kipengele ili kuchagua vigezo sahihi kwa mfano wenye utendaji bora. Hata hivyo, si sawa: "Uchimbaji wa kipengele huunda vipengele vipya kutoka kwa kazi za vipengele vya awali, wakati uchaguzi wa kipengele unarudisha sehemu ndogo ya vipengele." ([chanzo](https://wikipedia.org/wiki/Feature_selection))

### Onyesha data yako

Sehemu muhimu ya zana ya mwanasayansi wa data ni uwezo wa kuonyesha data kwa kutumia maktaba kadhaa bora kama Seaborn au MatPlotLib. Kuonyesha data yako kwa njia ya picha kunaweza kukuruhusu kugundua uhusiano uliofichwa ambao unaweza kutumia. Picha zako zinaweza pia kukusaidia kugundua upendeleo au data isiyo na uwiano (kama tunavyogundua katika [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Gawanya seti yako ya data

Kabla ya mafunzo, unahitaji kugawanya seti yako ya data katika sehemu mbili au zaidi za ukubwa usio sawa ambazo bado zinawakilisha data vizuri.

- **Mafunzo**. Sehemu hii ya seti ya data inafaa kwa mfano wako ili kuufundisha. Seti hii inajumuisha sehemu kubwa ya seti ya data ya awali.
- **Majaribio**. Seti ya majaribio ni kikundi huru cha data, mara nyingi hukusanywa kutoka data ya awali, unayotumia kuthibitisha utendaji wa mfano uliojengwa.
- **Uthibitishaji**. Seti ya uthibitishaji ni kikundi kidogo cha mifano huru unayotumia kurekebisha vigezo vya mfano, au usanifu, ili kuboresha mfano. Kulingana na ukubwa wa data yako na swali unalouliza, huenda usihitaji kujenga seti hii ya tatu (kama tunavyobainisha katika [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Kujenga mfano

Kwa kutumia data yako ya mafunzo, lengo lako ni kujenga mfano, au uwakilishi wa takwimu wa data yako, kwa kutumia algoriti mbalimbali ili **kufundisha**. Kufundisha mfano kunaupa data na kuruhusu kufanya dhana kuhusu mifumo inayotambua, kuthibitisha, na kukubali au kukataa.

### Amua mbinu ya mafunzo

Kulingana na swali lako na asili ya data yako, utachagua mbinu ya kuifundisha. Ukipitia [nyaraka za Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - tunazotumia katika kozi hii - unaweza kuchunguza njia nyingi za kufundisha mfano. Kulingana na uzoefu wako, huenda ukajaribu mbinu kadhaa tofauti ili kujenga mfano bora. Unaweza kupitia mchakato ambapo wanasayansi wa data wanatathmini utendaji wa mfano kwa kuupa data ambayo haijawahi kuonekana, kuangalia usahihi, upendeleo, na masuala mengine yanayopunguza ubora, na kuchagua mbinu ya mafunzo inayofaa zaidi kwa kazi iliyopo.

### Fanya mafunzo ya mfano

Ukiwa na data yako ya mafunzo, uko tayari 'kuifaa' ili kuunda mfano. Utagundua kwamba katika maktaba nyingi za ML utapata msimbo 'model.fit' - ni wakati huu ambapo unatuma kigezo chako cha kipengele kama safu ya thamani (kwa kawaida 'X') na kigezo cha lengo (kwa kawaida 'y').

### Tathmini mfano

Mara mchakato wa mafunzo unapokamilika (inaweza kuchukua marudio mengi, au 'epochs', kufundisha mfano mkubwa), utaweza kutathmini ubora wa mfano kwa kutumia data ya majaribio kupima utendaji wake. Data hii ni sehemu ndogo ya data ya awali ambayo mfano haujawahi kuchambua. Unaweza kuchapisha jedwali la vipimo kuhusu ubora wa mfano wako.

🎓 **Kufaa kwa mfano**

Katika muktadha wa kujifunza mashine, kufaa kwa mfano kunahusu usahihi wa kazi ya msingi ya mfano unavyojaribu kuchambua data ambayo hauijui.

🎓 **Kutofaa** na **kufaa kupita kiasi** ni matatizo ya kawaida yanayopunguza ubora wa mfano, kwani mfano unafaa aidha si vizuri vya kutosha au vizuri kupita kiasi. Hii husababisha mfano kufanya utabiri aidha kwa ukaribu sana au kwa umbali sana na data yake ya mafunzo. Mfano uliokaa kupita kiasi hutabiri data ya mafunzo vizuri sana kwa sababu umejifunza maelezo na kelele za data vizuri sana. Mfano usiofaa si sahihi kwani hauwezi kuchambua data yake ya mafunzo wala data ambayo bado haujaiona kwa usahihi.

![mfano uliokaa kupita kiasi](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Picha ya taarifa na [Jen Looper](https://twitter.com/jenlooper)

## Kuboresha vigezo

Mara mafunzo yako ya awali yanapokamilika, angalia ubora wa mfano na fikiria kuuboresha kwa kurekebisha 'vigezo vya juu'. Soma zaidi kuhusu mchakato huu [katika nyaraka](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Utabiri

Huu ndio wakati ambapo unaweza kutumia data mpya kabisa kujaribu usahihi wa mfano wako. Katika muktadha wa ML 'iliyotumika', ambapo unajenga mali za wavuti kutumia mfano katika uzalishaji, mchakato huu unaweza kuhusisha kukusanya pembejeo za mtumiaji (mfano wa kubonyeza kitufe) kuweka kigezo na kukituma kwa mfano kwa ajili ya uchambuzi au tathmini.

Katika masomo haya, utagundua jinsi ya kutumia hatua hizi kuandaa, kujenga, kujaribu, kutathmini, na kutabiri - hatua zote za mwanasayansi wa data na zaidi, unavyosonga mbele katika safari yako ya kuwa mhandisi wa ML 'full stack'.

---

## 🚀Changamoto

Chora mchoro wa mtiririko unaoonyesha hatua za mtaalamu wa ML. Unaona uko wapi sasa katika mchakato? Unadhani utapata ugumu wapi? Nini kinaonekana rahisi kwako?

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Tafuta mtandaoni mahojiano na wanasayansi wa data wanaojadili kazi zao za kila siku. Hapa kuna [moja](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Kazi

[Mahojiano na mwanasayansi wa data](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.