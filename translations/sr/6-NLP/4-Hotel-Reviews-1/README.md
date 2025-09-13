<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T13:57:56+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "sr"
}
-->
# Анализа сентимента са рецензијама хотела - обрада података

У овом делу ћете користити технике из претходних лекција за истраживачку анализу великог скупа података. Када стекнете добро разумевање корисности различитих колона, научићете:

- како да уклоните непотребне колоне
- како да израчунате нове податке на основу постојећих колона
- како да сачувате резултујући скуп података за употребу у завршном изазову

## [Квиз пре предавања](https://ff-quizzes.netlify.app/en/ml/)

### Увод

До сада сте научили да текстуални подаци значајно одступају од нумеричких типова података. Ако је текст написан или изговорен од стране човека, може се анализирати ради проналажења образаца, учесталости, сентимента и значења. Ова лекција вас уводи у прави скуп података са правим изазовом: **[515K рецензија хотела у Европи](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, који укључује [CC0: Јавна доменска лиценца](https://creativecommons.org/publicdomain/zero/1.0/). Подаци су прикупљени са Booking.com са јавних извора. Креатор скупа података је Џијашен Лиу.

### Припрема

Потребно вам је:

* Могућност покретања .ipynb нотебука користећи Python 3
* pandas
* NLTK, [који треба да инсталирате локално](https://www.nltk.org/install.html)
* Скуп података који је доступан на Kaggle [515K рецензија хотела у Европи](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Скуп података је величине око 230 MB када се распакује. Преузмите га у коренски `/data` директоријум повезан са овим лекцијама о NLP-у.

## Истраживачка анализа података

Овај изазов претпоставља да правите бота за препоруке хотела користећи анализу сентимента и оцене гостију. Скуп података који ћете користити укључује рецензије 1493 различита хотела у 6 градова.

Користећи Python, скуп података о рецензијама хотела и NLTK анализу сентимента, могли бисте открити:

* Које су најчешће коришћене речи и фразе у рецензијама?
* Да ли званичне *ознаке* које описују хотел корелирају са оценама рецензија (нпр. да ли су негативније рецензије за одређени хотел од *Породица са малом децом* него од *Самостални путник*, што можда указује да је бољи за *Самосталне путнике*)?
* Да ли NLTK оцене сентимента 'сагласне' са нумеричком оценом рецензента хотела?

#### Скуп података

Хајде да истражимо скуп података који сте преузели и сачували локално. Отворите датотеку у уређивачу као што је VS Code или чак Excel.

Заглавља у скупу података су следећа:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Ево их груписаних на начин који може бити лакши за преглед:  
##### Колоне хотела

* `Hotel_Name`, `Hotel_Address`, `lat` (географска ширина), `lng` (географска дужина)
  * Користећи *lat* и *lng* могли бисте направити мапу у Python-у која приказује локације хотела (можда обојене у складу са негативним и позитивним рецензијама)
  * Hotel_Address нам није очигледно користан, и вероватно ћемо га заменити државом ради лакшег сортирања и претраге

**Колоне мета-рецензија хотела**

* `Average_Score`
  * Према креатору скупа података, ова колона представља *Просечну оцену хотела, израчунату на основу најновијег коментара у последњој години*. Ово изгледа као необичан начин за израчунавање оцене, али то су подаци који су прикупљени, па их за сада можемо узети здраво за готово.

  ✅ На основу осталих колона у овим подацима, можете ли смислити други начин за израчунавање просечне оцене?

* `Total_Number_of_Reviews`
  * Укупни број рецензија које је хотел добио - није јасно (без писања кода) да ли се ово односи на рецензије у скупу података.
* `Additional_Number_of_Scoring`
  * Ово значи да је дата оцена, али није написана позитивна или негативна рецензија од стране рецензента.

**Колоне рецензија**

- `Reviewer_Score`
  - Ово је нумеричка вредност са највише једним децималним местом између минималне и максималне вредности 2.5 и 10
  - Није објашњено зашто је 2.5 најнижа могућа оцена
- `Negative_Review`
  - Ако рецензент није ништа написао, ово поље ће имати "**No Negative**"
  - Имајте на уму да рецензент може написати позитивну рецензију у колони Negative review (нпр. "нема ништа лоше у овом хотелу")
- `Review_Total_Negative_Word_Counts`
  - Већи број негативних речи указује на нижу оцену (без провере сентимента)
- `Positive_Review`
  - Ако рецензент није ништа написао, ово поље ће имати "**No Positive**"
  - Имајте на уму да рецензент може написати негативну рецензију у колони Positive review (нпр. "нема ништа добро у овом хотелу")
- `Review_Total_Positive_Word_Counts`
  - Већи број позитивних речи указује на вишу оцену (без провере сентимента)
- `Review_Date` и `days_since_review`
  - Може се применити мера свежине или застарелости рецензије (старије рецензије можда нису толико тачне као новије јер се менаџмент хотела променио, или су извршене реновације, или је додат базен итд.)
- `Tags`
  - Ово су кратки описи које рецензент може изабрати да опише тип госта који је био (нпр. самосталан или породичан), тип собе коју је имао, дужину боравка и начин на који је рецензија поднета.
  - Нажалост, коришћење ових ознака је проблематично, погледајте одељак испод који разматра њихову корисност.

**Колоне рецензента**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Ово би могло бити фактор у моделу препоруке, на пример, ако можете утврдити да су продуктивнији рецензенти са стотинама рецензија вероватније негативни него позитивни. Међутим, рецензент било које конкретне рецензије није идентификован јединственим кодом, и стога се не може повезати са сетом рецензија. Постоји 30 рецензената са 100 или више рецензија, али је тешко видети како ово може помоћи моделу препоруке.
- `Reviewer_Nationality`
  - Неки људи могу мислити да су одређене националности склоније дају позитивне или негативне рецензије због националне склоности. Будите опрезни приликом уградње таквих анегдотских ставова у своје моделе. Ово су национални (а понекад и расни) стереотипи, а сваки рецензент је био појединац који је написао рецензију на основу свог искуства. То искуство је могло бити филтрирано кроз многе призме као што су њихови претходни боравци у хотелима, пређена удаљеност и њихов лични темперамент. Тешко је оправдати мишљење да је њихова националност била разлог за оцену рецензије.

##### Примери

| Просечна оцена | Укупно рецензија | Оцена рецензента | Негативна <br />рецензија                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Позитивна рецензија                 | Ознаке                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Ово тренутно није хотел већ градилиште. Терорисан сам од раног јутра и током целог дана неприхватљивом буком грађевинских радова док сам се одмарао након дугог путовања и радио у соби. Људи су радили током целог дана, нпр. са бушилицама у суседним собама. Тражио сам промену собе, али није било доступне тихе собе. Да ствар буде гора, наплатили су ми више. Одјавио сам се увече јер сам морао да кренем рано на лет и добио одговарајући рачун. Дан касније хотел је направио још једну наплату без мог пристанка, која је премашила договорену цену. То је ужасно место. Немојте се кажњавати резервисањем овде. | Ништа. Ужасно место. Држите се даље. | Пословно путовање. Пар. Стандардна двокреветна соба. Боравак 2 ноћи. |

Као што видите, овај гост није имао срећан боравак у овом хотелу. Хотел има добру просечну оцену од 7.8 и 1945 рецензија, али овај рецензент му је дао 2.5 и написао 115 речи о томе колико је његов боравак био негативан. Ако није написао ништа у колони Positive_Review, могли бисте закључити да није било ничег позитивног, али ипак је написао 7 речи упозорења. Ако бисмо само бројали речи уместо значења или сентимента речи, могли бисмо добити искривљену слику намере рецензента. Чудно, њихова оцена од 2.5 је збуњујућа, јер ако је боравак у хотелу био толико лош, зашто му уопште дати било какве бодове? Истражујући скуп података пажљиво, видећете да је најнижа могућа оцена 2.5, а не 0. Највиша могућа оцена је 10.

##### Ознаке

Као што је горе поменуто, на први поглед, идеја да се користе `Tags` за категоризацију података има смисла. Нажалост, ове ознаке нису стандардизоване, што значи да у датом хотелу опције могу бити *Single room*, *Twin room* и *Double room*, али у следећем хотелу, оне су *Deluxe Single Room*, *Classic Queen Room* и *Executive King Room*. Ово могу бити исте ствари, али постоји толико варијација да избор постаје:

1. Покушај да се сви термини промене у један стандард, што је веома тешко, јер није јасно какав би био пут конверзије у сваком случају (нпр. *Classic single room* се мапира на *Single room*, али *Superior Queen Room with Courtyard Garden or City View* је много теже мапирати)

1. Можемо применити NLP приступ и измерити учесталост одређених термина као што су *Solo*, *Business Traveller* или *Family with young kids* како се односе на сваки хотел, и то укључити у препоруку  

Ознаке су обично (али не увек) једно поље које садржи листу од 5 до 6 вредности одвојених зарезима које се односе на *Тип путовања*, *Тип гостију*, *Тип собе*, *Број ноћи* и *Тип уређаја на којем је рецензија поднета*. Међутим, пошто неки рецензенти не попуњавају свако поље (могу оставити једно празно), вредности нису увек у истом редоследу.

На пример, узмите *Тип групе*. Постоји 1025 јединствених могућности у овом пољу у колони `Tags`, и нажалост само неке од њих се односе на групу (неке су тип собе итд.). Ако филтрирате само оне које помињу породицу, резултати садрже много резултата типа *Family room*. Ако укључите термин *with*, тј. бројите вредности *Family with*, резултати су бољи, са преко 80,000 од 515,000 резултата који садрже фразу "Family with young children" или "Family with older children".

Ово значи да колона ознака није потпуно бескорисна за нас, али ће бити потребно мало рада да би била корисна.

##### Просечна оцена хотела

Постоји низ необичности или неслагања са скупом података које не могу да схватим, али су овде илустроване како бисте били свесни њих приликом изградње својих модела. Ако их схватите, молимо вас да нас обавестите у делу за дискусију!

Скуп података има следеће колоне које се односе на просечну оцену и број рецензија:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Хотел са највише рецензија у овом скупу података је *Britannia International Hotel Canary Wharf* са 4789 рецензија од 515,000. Али ако погледамо вредност `Total_Number_of_Reviews` за овај хотел, она је 9086. Могли бисте закључити да постоји много више оцена без рецензија, па можда треба додати вредност колоне `Additional_Number_of_Scoring`. Та вредност је 2682, и додајући је на 4789 добијамо 7471, што је и даље 1615 мање од `Total_Number_of_Reviews`.

Ако узмете колону `Average_Score`, могли бисте закључити да је то просек рецензија у скупу података, али опис са Kaggle-а је "*Просечна оцена хотела, израчуната на основу најновијег коментара у последњој години*". То не изгледа баш корисно, али можемо израчунати сопствени просек на основу оцена рецензија у скупу података. Узимајући исти хотел као пример, просечна оцена хотела је дата као 7.1, али израчуната оцена (просечна оцена рецензента *у* скупу података) је 6.8. Ово је близу, али није иста вредност, и можемо само претпоставити да су оцене дате у рецензијама `Additional_Number_of_Scoring` повећале просек на 7.1. Нажалост, без начина да се тестира или докаже та тврдња, тешко је користити или веровати `Average_Score`, `Additional_Number_of_Scoring` и `Total_Number_of_Reviews` када се заснивају на, или се односе на, податке које немамо.

Да додатно закомпликујемо ствари, хотел са другим највећим бројем рецензија има израчунату просечну оцену од 8.12, а `Average_Score` у скупу података је 
> 🚨 Напомена о опрезу  
>  
> Када радите са овим скупом података, писаћете код који израчунава нешто из текста без потребе да сами читате или анализирате текст. Ово је суштина обраде природног језика (NLP), тумачење значења или сентимента без потребе да то ради човек. Међутим, могуће је да ћете прочитати неке од негативних рецензија. Саветовао бих вам да то не радите, јер нема потребе. Неки од њих су бесмислени или небитни негативни коментари о хотелима, као што је „Време није било добро“, нешто што је ван контроле хотела, или било кога другог. Али постоји и мрачна страна неких рецензија. Понекад су негативне рецензије расистичке, сексистичке или дискриминаторне према старосној доби. Ово је нажалост очекивано у скупу података који је прикупљен са јавног веб-сајта. Неки рецензенти остављају коментаре који би вам могли бити непријатни, узнемирујући или неприхватљиви. Боље је дозволити коду да измери сентимент него да их сами читате и узнемирите се. Ипак, мањина је оних који пишу такве ствари, али они ипак постоје.
## Вежба - Истраживање података
### Учитавање података

Довољно је визуелно прегледати податке, сада ћете написати код и добити одговоре! Овај део користи библиотеку pandas. Ваш први задатак је да се уверите да можете учитати и прочитати CSV податке. Библиотека pandas има брз CSV учитач, а резултат се смешта у dataframe, као у претходним лекцијама. CSV који учитавамо има преко пола милиона редова, али само 17 колона. Pandas вам пружа много моћних начина за интеракцију са dataframe-ом, укључујући могућност извршавања операција на сваком реду.

Од овог тренутка у лекцији, биће укључени делови кода, објашњења кода и дискусија о томе шта резултати значе. Користите приложени _notebook.ipynb_ за ваш код.

Хајде да почнемо са учитавањем датотеке коју ћете користити:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Сада када су подаци учитани, можемо извршити неке операције над њима. Држите овај код на врху вашег програма за следећи део.

## Истраживање података

У овом случају, подаци су већ *чисти*, што значи да су спремни за рад и да не садрже карактере на другим језицима који би могли да збуне алгоритме који очекују само енглеске карактере.

✅ Можда ћете морати да радите са подацима који захтевају почетну обраду како би се форматирали пре примене NLP техника, али не овог пута. Ако бисте морали, како бисте се носили са карактерима који нису на енглеском?

Одвојите тренутак да се уверите да, када се подаци учитају, можете их истражити помоћу кода. Веома је лако фокусирати се на колоне `Negative_Review` и `Positive_Review`. Оне су испуњене природним текстом за ваше NLP алгоритме. Али сачекајте! Пре него што ускочите у NLP и анализу сентимента, требало би да следите код испод како бисте утврдили да ли вредности дате у датасету одговарају вредностима које израчунате помоћу pandas-а.

## Операције над dataframe-ом

Први задатак у овој лекцији је да проверите да ли су следеће тврдње тачне тако што ћете написати код који испитује dataframe (без промене истог).

> Као и код многих програмских задатака, постоји више начина да се ово уради, али добар савет је да то урадите на најједноставнији, најлакши начин, посебно ако ће бити лакше разумети када се вратите на овај код у будућности. Са dataframe-ом, постоји свеобухватан API који ће често имати начин да ефикасно урадите оно што желите.

Третирајте следећа питања као програмске задатке и покушајте да их решите без гледања у решење.

1. Испишите *облик* dataframe-а који сте управо учитали (облик је број редова и колона).
2. Израчунати учесталост националности рецензената:
   1. Колико различитих вредности постоји за колону `Reviewer_Nationality` и које су то вредности?
   2. Која националност рецензената је најчешћа у датасету (испишите земљу и број рецензија)?
   3. Које су следећих 10 најчешћих националности и њихова учесталост?
3. Који је хотел најчешће рецензиран за сваку од 10 најчешћих националности рецензената?
4. Колико рецензија има по хотелу (учесталост рецензија по хотелу) у датасету?
5. Иако постоји колона `Average_Score` за сваки хотел у датасету, можете израчунати просечну оцену (узимајући просек свих оцена рецензената у датасету за сваки хотел). Додајте нову колону у ваш dataframe са заглављем колоне `Calc_Average_Score` која садржи израчунати просек.
6. Да ли неки хотели имају исту (заокружену на 1 децималу) `Average_Score` и `Calc_Average_Score`?
   1. Покушајте да напишете Python функцију која узима Series (ред) као аргумент и упоређује вредности, исписујући поруку када вредности нису једнаке. Затим користите `.apply()` метод да обрадите сваки ред помоћу функције.
7. Израчунати и исписати колико редова има вредност колоне `Negative_Review` "No Negative".
8. Израчунати и исписати колико редова има вредност колоне `Positive_Review` "No Positive".
9. Израчунати и исписати колико редова има вредност колоне `Positive_Review` "No Positive" **и** вредност колоне `Negative_Review` "No Negative".

### Одговори у коду

1. Испишите *облик* dataframe-а који сте управо учитали (облик је број редова и колона).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Израчунати учесталост националности рецензената:

   1. Колико различитих вредности постоји за колону `Reviewer_Nationality` и које су то вредности?
   2. Која националност рецензената је најчешћа у датасету (испишите земљу и број рецензија)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. Које су следећих 10 најчешћих националности и њихова учесталост?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. Који је хотел најчешће рецензиран за сваку од 10 најчешћих националности рецензената?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. Колико рецензија има по хотелу (учесталост рецензија по хотелу) у датасету?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   Можда ћете приметити да резултати *израчунати у датасету* не одговарају вредности у `Total_Number_of_Reviews`. Није јасно да ли ова вредност у датасету представља укупан број рецензија које је хотел имао, али нису све биле scraped, или нека друга рачуница. `Total_Number_of_Reviews` се не користи у моделу због ове нејасноће.

5. Иако постоји колона `Average_Score` за сваки хотел у датасету, можете израчунати просечну оцену (узимајући просек свих оцена рецензената у датасету за сваки хотел). Додајте нову колону у ваш dataframe са заглављем колоне `Calc_Average_Score` која садржи израчунати просек. Испишите колоне `Hotel_Name`, `Average_Score` и `Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Можда ћете се запитати о вредности `Average_Score` и зашто је понекад различита од израчунатог просека. Како не можемо знати зашто неке вредности одговарају, а друге имају разлику, најсигурније је у овом случају користити оцене рецензената које имамо за израчунавање просека. Ипак, разлике су обично веома мале, ево хотела са највећим одступањем између просека из датасета и израчунатог просека:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   Са само 1 хотелом који има разлику у оцени већу од 1, то значи да вероватно можемо игнорисати разлику и користити израчунати просек.

6. Израчунати и исписати колико редова има вредност колоне `Negative_Review` "No Negative".

7. Израчунати и исписати колико редова има вредност колоне `Positive_Review` "No Positive".

8. Израчунати и исписати колико редова има вредност колоне `Positive_Review` "No Positive" **и** вредност колоне `Negative_Review` "No Negative".

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Други начин

Други начин за бројање ставки без Lambdas, и коришћење sum за бројање редова:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Можда сте приметили да постоји 127 редова који имају и "No Negative" и "No Positive" вредности за колоне `Negative_Review` и `Positive_Review` респективно. То значи да је рецензент дао хотелу нумеричку оцену, али је одбио да напише позитивну или негативну рецензију. Срећом, ово је мали број редова (127 од 515738, или 0.02%), тако да вероватно неће утицати на наш модел или резултате у било ком правцу, али можда нисте очекивали да датасет рецензија има редове без рецензија, па је вредно истражити податке како бисте открили овакве редове.

Сада када сте истражили датасет, у следећој лекцији ћете филтрирати податке и додати анализу сентимента.

---
## 🚀Изазов

Ова лекција показује, као што смо видели у претходним лекцијама, колико је критично важно разумети ваше податке и њихове недостатке пре него што извршите операције над њима. Подаци засновани на тексту, посебно, захтевају пажљиво испитивање. Претражите различите датасете богате текстом и видите да ли можете открити области које би могле увести пристрасност или искривљен сентимент у модел.

## [Квиз након предавања](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостално учење

Пређите [овај пут учења о NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) да бисте открили алате које можете испробати приликом изградње модела заснованих на говору и тексту.

## Задатак 

[NLTK](assignment.md)

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако тежимо тачности, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на изворном језику треба сматрати ауторитативним извором. За критичне информације препоручује се професионални превод од стране људи. Не сносимо одговорност за било каква неспоразумевања или погрешна тумачења настала услед коришћења овог превода.