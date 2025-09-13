<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T14:03:26+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "uk"
}
-->
# Аналіз настроїв за відгуками про готелі - обробка даних

У цьому розділі ви використаєте техніки з попередніх уроків для проведення дослідницького аналізу великого набору даних. Після того, як ви добре зрозумієте корисність різних колонок, ви навчитеся:

- як видаляти непотрібні колонки
- як обчислювати нові дані на основі існуючих колонок
- як зберігати отриманий набір даних для використання у фінальному завданні

## [Тест перед лекцією](https://ff-quizzes.netlify.app/en/ml/)

### Вступ

До цього моменту ви дізналися, що текстові дані значно відрізняються від числових типів даних. Якщо це текст, написаний або сказаний людиною, його можна аналізувати для пошуку шаблонів, частот, настроїв і значень. Цей урок знайомить вас із реальним набором даних і реальним завданням: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, який має [ліцензію CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). Дані були зібрані з Booking.com із публічних джерел. Автором набору даних є Jiashen Liu.

### Підготовка

Вам знадобиться:

* Можливість запускати .ipynb ноутбуки з використанням Python 3
* pandas
* NLTK, [який слід встановити локально](https://www.nltk.org/install.html)
* Набір даних, доступний на Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Розмір файлу після розпакування становить близько 230 МБ. Завантажте його в кореневу папку `/data`, пов’язану з цими уроками NLP.

## Дослідницький аналіз даних

Це завдання передбачає, що ви створюєте бота для рекомендацій готелів, використовуючи аналіз настроїв і оцінки гостей. Набір даних, який ви будете використовувати, включає відгуки про 1493 різні готелі в 6 містах.

Використовуючи Python, набір даних відгуків про готелі та аналіз настроїв NLTK, ви можете дізнатися:

* Які слова та фрази найчастіше використовуються у відгуках?
* Чи корелюють офіційні *теги*, що описують готель, з оцінками відгуків (наприклад, чи є більше негативних відгуків для певного готелю від *Сімей з маленькими дітьми*, ніж від *Соло-мандрівників*, можливо, це вказує на те, що готель більше підходить для *Соло-мандрівників*)?
* Чи збігаються оцінки настроїв NLTK з числовою оцінкою рецензента?

#### Набір даних

Давайте дослідимо набір даних, який ви завантажили та зберегли локально. Відкрийте файл у редакторі, наприклад, VS Code або навіть Excel.

Заголовки в наборі даних такі:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Ось вони згруповані таким чином, щоб їх було легше аналізувати: 
##### Колонки готелів

* `Hotel_Name`, `Hotel_Address`, `lat` (широта), `lng` (довгота)
  * Використовуючи *lat* і *lng*, ви можете побудувати карту за допомогою Python, яка показує розташування готелів (можливо, з кольоровим кодуванням для негативних і позитивних відгуків)
  * Hotel_Address не є очевидно корисним для нас, і ми, ймовірно, замінимо його на країну для зручнішого сортування та пошуку

**Колонки мета-відгуків готелів**

* `Average_Score`
  * За словами автора набору даних, ця колонка містить *Середню оцінку готелю, розраховану на основі останнього коментаря за останній рік*. Це здається незвичайним способом розрахунку оцінки, але це дані, які були зібрані, тому ми можемо прийняти їх за основу на даний момент.
  
  ✅ Виходячи з інших колонок у цьому наборі даних, чи можете ви придумати інший спосіб розрахунку середньої оцінки?

* `Total_Number_of_Reviews`
  * Загальна кількість відгуків, які отримав цей готель - не зрозуміло (без написання коду), чи стосується це відгуків у наборі даних.
* `Additional_Number_of_Scoring`
  * Це означає, що оцінка була надана, але позитивний або негативний відгук не був написаний рецензентом

**Колонки відгуків**

- `Reviewer_Score`
  - Це числове значення з максимумом 1 десятковим знаком між мінімальним і максимальним значеннями 2.5 і 10
  - Не пояснюється, чому найнижча можлива оцінка становить 2.5
- `Negative_Review`
  - Якщо рецензент нічого не написав, це поле матиме "**No Negative**"
  - Зверніть увагу, що рецензент може написати позитивний відгук у колонці Negative review (наприклад, "немає нічого поганого в цьому готелі")
- `Review_Total_Negative_Word_Counts`
  - Більша кількість негативних слів вказує на нижчу оцінку (без перевірки настрою)
- `Positive_Review`
  - Якщо рецензент нічого не написав, це поле матиме "**No Positive**"
  - Зверніть увагу, що рецензент може написати негативний відгук у колонці Positive review (наприклад, "у цьому готелі немає нічого хорошого")
- `Review_Total_Positive_Word_Counts`
  - Більша кількість позитивних слів вказує на вищу оцінку (без перевірки настрою)
- `Review_Date` і `days_since_review`
  - Можна застосувати міру свіжості або застарілості до відгуку (старіші відгуки можуть бути менш точними, ніж новіші, оскільки управління готелем змінилося, були проведені ремонти або додано басейн тощо)
- `Tags`
  - Це короткі дескриптори, які рецензент може вибрати для опису типу гостя (наприклад, соло або сім'я), типу кімнати, тривалості перебування та способу подання відгуку.
  - На жаль, використання цих тегів є проблематичним, див. розділ нижче, який обговорює їх корисність

**Колонки рецензентів**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Це може бути фактором у моделі рекомендацій, наприклад, якщо ви можете визначити, що більш продуктивні рецензенти з сотнями відгуків частіше були негативними, ніж позитивними. Однак рецензент будь-якого конкретного відгуку не ідентифікується за унікальним кодом, і тому його не можна пов’язати з набором відгуків. Є 30 рецензентів із 100 або більше відгуками, але важко зрозуміти, як це може допомогти моделі рекомендацій.
- `Reviewer_Nationality`
  - Дехто може думати, що певні національності частіше дають позитивні або негативні відгуки через національну схильність. Будьте обережні, будуючи такі анекдотичні погляди у своїх моделях. Це національні (а іноді й расові) стереотипи, і кожен рецензент був індивідуальним, який написав відгук на основі свого досвіду. Він міг бути відфільтрований через багато лінз, таких як їхні попередні перебування в готелях, пройдена відстань і їхній особистий темперамент. Вважати, що їхня національність була причиною оцінки відгуку, важко виправдати.

##### Приклади

| Середня оцінка | Загальна кількість відгуків | Оцінка рецензента | Негативний <br />відгук                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Позитивний відгук                 | Теги                                                                                      |
| -------------- | -------------------------- | ----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                       | 2.5               | Це зараз не готель, а будівельний майданчик. Мене тероризували з раннього ранку і весь день неприйнятним будівельним шумом, поки я відпочивав після довгої подорожі та працював у кімнаті. Люди працювали весь день, тобто з відбійними молотками в сусідніх кімнатах. Я попросив змінити кімнату, але тихої кімнати не було. Щоб погіршити ситуацію, мене переплатили. Я виїхав увечері, оскільки мав дуже ранній рейс, і отримав відповідний рахунок. Через день готель зробив ще один платіж без моєї згоди, перевищивши заброньовану ціну. Це жахливе місце. Не карайте себе, бронюючи тут. | Нічого. Жахливе місце. Тримайтеся подалі | Ділова поїздка                                Пара Стандартний двомісний номер Перебування 2 ночі |

Як бачите, цей гість не мав щасливого перебування в цьому готелі. Готель має хорошу середню оцінку 7.8 і 1945 відгуків, але цей рецензент дав йому 2.5 і написав 115 слів про те, наскільки негативним було його перебування. Якщо вони нічого не написали в колонці Positive_Review, можна припустити, що нічого позитивного не було, але вони написали 7 слів попередження. Якщо ми просто рахували слова, а не їхнє значення чи настрій, ми могли б отримати спотворене уявлення про наміри рецензента. Дивно, що їхня оцінка 2.5 збиває з пантелику, адже якщо перебування в готелі було настільки поганим, чому взагалі давати якісь бали? Досліджуючи набір даних уважно, ви побачите, що найнижча можлива оцінка становить 2.5, а не 0. Найвища можлива оцінка — 10.

##### Теги

Як згадувалося вище, на перший погляд, ідея використання `Tags` для категоризації даних здається логічною. На жаль, ці теги не стандартизовані, що означає, що в одному готелі варіанти можуть бути *Single room*, *Twin room* і *Double room*, а в іншому готелі — *Deluxe Single Room*, *Classic Queen Room* і *Executive King Room*. Це можуть бути ті самі речі, але існує так багато варіацій, що вибір стає:

1. Спроба змінити всі терміни на єдиний стандарт, що дуже складно, оскільки не зрозуміло, яким буде шлях конверсії в кожному випадку (наприклад, *Classic single room* відповідає *Single room*, але *Superior Queen Room with Courtyard Garden or City View* набагато складніше зіставити)

1. Ми можемо застосувати підхід NLP і виміряти частоту певних термінів, таких як *Solo*, *Business Traveller* або *Family with young kids*, як вони застосовуються до кожного готелю, і врахувати це в рекомендації  

Теги зазвичай (але не завжди) є одним полем, що містить список із 5-6 значень, розділених комами, які відповідають *Типу подорожі*, *Типу гостей*, *Типу кімнати*, *Кількості ночей* і *Типу пристрою, на якому був поданий відгук*. Однак через те, що деякі рецензенти не заповнюють кожне поле (вони можуть залишити одне порожнім), значення не завжди знаходяться в тому самому порядку.

Наприклад, візьмемо *Тип групи*. У цьому полі в колонці `Tags` є 1025 унікальних можливостей, і, на жаль, лише деякі з них стосуються групи (деякі стосуються типу кімнати тощо). Якщо ви відфільтруєте лише ті, що згадують сім’ю, результати містять багато результатів типу *Family room*. Якщо ви включите термін *with*, тобто врахуєте значення *Family with*, результати будуть кращими, з понад 80,000 із 515,000 результатів, що містять фразу "Family with young children" або "Family with older children".

Це означає, що колонка тегів не є повністю марною для нас, але її потрібно доопрацювати, щоб зробити корисною.

##### Середня оцінка готелю

У наборі даних є кілька дивностей або розбіжностей, які я не можу зрозуміти, але вони ілюструються тут, щоб ви знали про них під час створення своїх моделей. Якщо ви розберетеся, будь ласка, повідомте нам у розділі обговорення!

Набір даних має такі колонки, що стосуються середньої оцінки та кількості відгуків: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Готель із найбільшою кількістю відгуків у цьому наборі даних — *Britannia International Hotel Canary Wharf* із 4789 відгуками з 515,000. Але якщо ми подивимося на значення `Total_Number_of_Reviews` для цього готелю, воно становить 9086. Можна припустити, що є набагато більше оцінок без відгуків, тому, можливо, нам слід додати значення колонки `Additional_Number_of_Scoring`. Це значення становить 2682, і додавання його до 4789 дає нам 7471, що все ще на 1615 менше, ніж `Total_Number_of_Reviews`. 

Якщо взяти колонку `Average_Score`, можна припустити, що це середнє значення відгуків у наборі даних, але опис із Kaggle говорить: "*Середня оцінка готелю, розрахована на основі останнього коментаря за останній рік*". Це здається не дуже корисним, але ми можемо розрахувати власне середнє значення на основі оцінок відгуків у наборі даних. Використовуючи той самий готель як приклад, середня оцінка готелю становить 7.1, але розрахована оцінка (середня оцінка рецензента *у* наборі даних) становить 6.8. Це близько, але не те саме значення, і ми можемо лише припустити, що оцінки, наведені в `Additional_Number_of_Scoring`, збільшили середнє значення до 7.1. На жаль, без можливості перевірити або довести це твердження важко використовувати або довіряти `Average_Score`, `Additional_Number_of_Scoring` і `Total_Number_of_Reviews`, коли вони базуються на даних, яких у нас немає.

Щоб ускладнити ситуацію, готель із другою найбільшою кількістю відгуків має розраховану середню оцінку 8.12, а `Average_Score` у наборі даних становить 8.1. Чи є ця правильна оцінка збігом, чи перший готель є розбіжністю? 

На можливість того, що ці готелі можуть бути винятками, і що, можливо, більшість значень збігаються (але деякі з якихось причин не збігаються), ми напишемо коротку програму, щоб дослідити значення в наборі даних і визначити правильне використання (або невикористання) значень.
> 🚨 Зауваження щодо обережності
>
> Працюючи з цим набором даних, ви будете писати код, який обчислює щось на основі тексту, не читаючи або аналізуючи сам текст. Це суть обробки природної мови (NLP) — інтерпретувати значення або настрій без участі людини. Однак можливо, що ви натрапите на деякі негативні відгуки. Я закликаю вас цього не робити, тому що в цьому немає необхідності. Деякі з них є безглуздими або неактуальними негативними відгуками про готелі, наприклад: "Погода була не дуже гарною", що знаходиться поза контролем готелю або будь-кого взагалі. Але є й темна сторона деяких відгуків. Іноді негативні відгуки містять расизм, сексизм або ейджизм. Це прикро, але очікувано для набору даних, зібраного з публічного вебсайту. Деякі рецензенти залишають відгуки, які можуть здатися вам неприємними, дискомфортними або такими, що засмучують. Краще дозволити коду оцінити настрій, ніж читати їх самостійно і засмучуватися. Тим не менш, це меншість, яка пише такі речі, але вони все ж існують.
## Вправа - Дослідження даних
### Завантаження даних

Досить візуально оглядати дані, тепер ви напишете код і отримаєте відповіді! У цьому розділі використовується бібліотека pandas. Ваше перше завдання — переконатися, що ви можете завантажити та прочитати дані CSV. Бібліотека pandas має швидкий завантажувач CSV, і результат розміщується у dataframe, як у попередніх уроках. CSV, який ми завантажуємо, містить понад півмільйона рядків, але лише 17 стовпців. Pandas надає багато потужних способів взаємодії з dataframe, включаючи можливість виконувати операції над кожним рядком.

Далі в цьому уроці будуть фрагменти коду, пояснення до коду та обговорення того, що означають результати. Використовуйте включений _notebook.ipynb_ для вашого коду.

Почнемо із завантаження файлу даних, який ви будете використовувати:

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

Тепер, коли дані завантажено, ми можемо виконувати деякі операції над ними. Залиште цей код у верхній частині вашої програми для наступної частини.

## Дослідження даних

У цьому випадку дані вже *чисті*, тобто вони готові до роботи і не містять символів іншими мовами, які можуть завадити алгоритмам, що очікують лише англійські символи.

✅ Можливо, вам доведеться працювати з даними, які потребують початкової обробки для форматування перед застосуванням технік NLP, але не цього разу. Як би ви обробляли символи іншими мовами, якщо б це було необхідно?

Переконайтеся, що після завантаження даних ви можете досліджувати їх за допомогою коду. Дуже легко захотіти зосередитися на стовпцях `Negative_Review` і `Positive_Review`. Вони заповнені природним текстом для обробки вашими алгоритмами NLP. Але зачекайте! Перш ніж переходити до NLP і аналізу настроїв, слід виконати код нижче, щоб переконатися, що значення, наведені в наборі даних, відповідають значенням, які ви обчислюєте за допомогою pandas.

## Операції з dataframe

Перше завдання в цьому уроці — перевірити, чи правильні наступні твердження, написавши код, який досліджує dataframe (без його зміни).

> Як і багато завдань програмування, є кілька способів виконати це, але хороша порада — робити це найпростішим і найзручнішим способом, особливо якщо це буде легше зрозуміти, коли ви повернетеся до цього коду в майбутньому. У dataframe є всеосяжний API, який часто має спосіб зробити те, що вам потрібно, ефективно.

Розглядайте наступні питання як завдання з програмування та спробуйте відповісти на них, не дивлячись на рішення.

1. Виведіть *форму* dataframe, який ви щойно завантажили (форма — це кількість рядків і стовпців).
2. Обчисліть частоту для національностей рецензентів:
   1. Скільки унікальних значень є у стовпці `Reviewer_Nationality` і які вони?
   2. Яка національність рецензента є найпоширенішою в наборі даних (виведіть країну та кількість рецензій)?
   3. Які наступні 10 найчастіше зустрічаються національності та їх частота?
3. Який готель отримав найбільше рецензій для кожної з 10 найпоширеніших національностей рецензентів?
4. Скільки рецензій є на кожен готель (частота рецензій на готель) у наборі даних?
5. Хоча в наборі даних є стовпець `Average_Score` для кожного готелю, ви також можете обчислити середній бал (отримавши середнє значення всіх оцінок рецензентів у наборі даних для кожного готелю). Додайте новий стовпець до вашого dataframe із заголовком стовпця `Calc_Average_Score`, який містить цей обчислений середній бал.
6. Чи є готелі, які мають однакові (округлені до 1 десяткового знака) значення `Average_Score` і `Calc_Average_Score`?
   1. Спробуйте написати функцію Python, яка приймає Series (рядок) як аргумент і порівнює значення, виводячи повідомлення, коли значення не рівні. Потім використовуйте метод `.apply()` для обробки кожного рядка за допомогою функції.
7. Обчисліть і виведіть кількість рядків, у яких значення стовпця `Negative_Review` дорівнює "No Negative".
8. Обчисліть і виведіть кількість рядків, у яких значення стовпця `Positive_Review` дорівнює "No Positive".
9. Обчисліть і виведіть кількість рядків, у яких значення стовпця `Positive_Review` дорівнює "No Positive" **і** значення стовпця `Negative_Review` дорівнює "No Negative".

### Відповіді у коді

1. Виведіть *форму* dataframe, який ви щойно завантажили (форма — це кількість рядків і стовпців).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Обчисліть частоту для національностей рецензентів:

   1. Скільки унікальних значень є у стовпці `Reviewer_Nationality` і які вони?
   2. Яка національність рецензента є найпоширенішою в наборі даних (виведіть країну та кількість рецензій)?

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

   3. Які наступні 10 найчастіше зустрічаються національності та їх частота?

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

3. Який готель отримав найбільше рецензій для кожної з 10 найпоширеніших національностей рецензентів?

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

4. Скільки рецензій є на кожен готель (частота рецензій на готель) у наборі даних?

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

   Ви можете помітити, що результати *підраховані в наборі даних* не відповідають значенню в `Total_Number_of_Reviews`. Незрозуміло, чи це значення в наборі даних представляє загальну кількість рецензій, які мав готель, але не всі були зібрані, чи це якесь інше обчислення. `Total_Number_of_Reviews` не використовується в моделі через цю неясність.

5. Хоча в наборі даних є стовпець `Average_Score` для кожного готелю, ви також можете обчислити середній бал (отримавши середнє значення всіх оцінок рецензентів у наборі даних для кожного готелю). Додайте новий стовпець до вашого dataframe із заголовком стовпця `Calc_Average_Score`, який містить цей обчислений середній бал. Виведіть стовпці `Hotel_Name`, `Average_Score` і `Calc_Average_Score`.

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

   Ви також можете задатися питанням про значення `Average_Score` і чому воно іноді відрізняється від обчисленого середнього балу. Оскільки ми не можемо знати, чому деякі значення збігаються, а інші мають різницю, найкраще в цьому випадку використовувати оцінки рецензентів, які ми маємо, щоб обчислити середнє самостійно. Тим не менш, різниця зазвичай дуже мала, ось готелі з найбільшим відхиленням між середнім балом із набору даних і обчисленим середнім балом:

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

   Лише 1 готель має різницю в оцінці більше 1, це означає, що ми, ймовірно, можемо ігнорувати різницю та використовувати обчислений середній бал.

6. Обчисліть і виведіть кількість рядків, у яких значення стовпця `Negative_Review` дорівнює "No Negative".

7. Обчисліть і виведіть кількість рядків, у яких значення стовпця `Positive_Review` дорівнює "No Positive".

8. Обчисліть і виведіть кількість рядків, у яких значення стовпця `Positive_Review` дорівнює "No Positive" **і** значення стовпця `Negative_Review` дорівнює "No Negative".

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

## Інший спосіб

Інший спосіб підрахувати елементи без Lambdas і використовувати sum для підрахунку рядків:

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

   Ви могли помітити, що є 127 рядків, які мають значення "No Negative" і "No Positive" для стовпців `Negative_Review` і `Positive_Review` відповідно. Це означає, що рецензент дав готелю числову оцінку, але відмовився писати як позитивний, так і негативний відгук. На щастя, це невелика кількість рядків (127 із 515738, або 0,02%), тому це, ймовірно, не вплине на нашу модель чи результати в будь-якому конкретному напрямку, але ви могли не очікувати, що набір даних рецензій міститиме рядки без рецензій, тому варто дослідити дані, щоб виявити такі рядки.

Тепер, коли ви дослідили набір даних, у наступному уроці ви відфільтруєте дані та додасте аналіз настроїв.

---
## 🚀Виклик

Цей урок демонструє, як ми бачили в попередніх уроках, наскільки важливо розуміти ваші дані та їх особливості перед виконанням операцій над ними. Текстові дані, зокрема, потребують ретельного аналізу. Перегляньте різні набори даних із великою кількістю тексту та спробуйте виявити області, які можуть вводити упередженість або спотворений настрій у модель.

## [Тест після лекції](https://ff-quizzes.netlify.app/en/ml/)

## Огляд і самостійне навчання

Пройдіть [цей навчальний шлях з NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott), щоб дізнатися про інструменти, які можна спробувати під час створення моделей для роботи з текстом і мовою.

## Завдання

[NLTK](assignment.md)

---

**Відмова від відповідальності**:  
Цей документ було перекладено за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, зверніть увагу, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ мовою оригіналу слід вважати авторитетним джерелом. Для критично важливої інформації рекомендується професійний людський переклад. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникли внаслідок використання цього перекладу.