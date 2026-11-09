# បង្កើតម៉ូដែលកំណត់ត្រា regression ប្រើប្រាស់ Scikit-learn: regression ៤ របៀប

## សម្គាល់សម្រាប់អ្នកចាប់ផ្ដើម

ការកំណត់ត្រា Linear regression ត្រូវបានប្រើពេលយើងចង់ទាយទោល **តម្លៃជាលេខ** (ឧទាហរណ៍, តម្លៃផ្ទះ, សីតុណ្ហភាព, ឬការលក់)។ វាធ្វើការដោយស្វែងរកខ្សែស្របមួយដែលតំណាងឱ្យទំនាក់ទំនងរវាងលក្ខណៈបញ្ចូល និងលទ្ធផលបានល្អបំផុត។

នៅក្នុងមេរៀននេះ យើងផ្តោតលើការយល់ដឹងពីគំនិតមុនពេលស្វែងយល់បច្ចេកទេស regression ដែលមានភាពស្មុគស្មាញជាងនេះ។
![Linear vs polynomial regression infographic](../../../../translated_images/km/linear-polynomial.5523c7cb6576ccab.webp)
> រូបតំណាងដោយ [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [តេស្តមុខមាត់មេរៀន](https://ff-quizzes.netlify.app/en/ml/)

> ### [មេរៀននេះមានជាភាសា R ផងដែរ!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### ការណែនាំ

រហូតដល់ពេលនេះ អ្នកបានស្វែងយល់អំពីអ្វីទៅជាកំណត់ត្រា regression ជាមួយទិន្នន័យគំរូពី dataset តម្លៃផ្លែគុជ ដែលយើងនឹងប្រើកន្លងមកក្នុងមេរៀននេះ។ អ្នកក៏បានបង្ហាញវាដោយប្រើ Matplotlib ផងដែរ។

ឥឡូវនេះ អ្នករួចរាល់ក្នុងការចូលដល់ regression ជ្រាលជ្រៅសម្រាប់ ML។ ខណៈពេលការបង្ហាញអនុញ្ញាតឲ្យអ្នកយល់ដឹងទិន្នន័យ កម្លាំងពិតរបស់ Machine Learning មកពី _ការ​បណ្តុះបណ្តាលម៉ូដែល_។ ម៉ូដែលទាំងនេះត្រូវបានបណ្តុះបណ្តាលលើទិន្នន័យប្រវត្តិ ដើម្បីចាប់យកក្បួនពាក់ព័ន្ធរវាងទិន្នន័យបានដោយស្វ័យប្រវត្តិ ហើយវាអនុញ្ញាតឲ្យអ្នកទាយទ្ឋានលទ្ធផលសម្រាប់ទិន្នន័យថ្មី ដែលម៉ូដែលមិនទាន់ឃើញពីមុន។

នៅក្នុងមេរៀននេះ អ្នកនឹងស្វែងយល់បន្ថែមពីប្រភេទនៃ regression ២ ប្រភេទ ៖ _linear regression ជាមូលដ្ឋាន_ និង _polynomial regression_, រួមជាមួយគណិតវិទ្យាមួយចំនួននៅពីក្រោយបច្ចេកទេសទាំងនេះ។ ម៉ូដែលទាំងនេះនឹងអនុញ្ញាតឲ្យយើងទាយតម្លៃផ្លែគុជដោយផ្អែកលើទិន្នន័យបញ្ចូលដ៏ខុសគ្នា។

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 ចុចរូបភាពខាងលើសម្រាប់មើលវីដេអូសង្ខេបពី linear regression។

> ក្នុងកម្រិតសិក្សាទាំងនេះ យើងសន្មតថាជំនាញគណិតវិទ្យាមិនខ្ពស់ ព្រមទាំងព្យាយាមធ្វើឲ្យវាអាចចូលដល់បានសម្រាប់សិស្សពីវិស័យផ្សេងៗ ដូច្នេះសូមមើលកំណត់សម្គាល់, 🧮 ការហៅ, រូបតំណាង និងឧបករណ៍រៀនផ្សេងទៀតសម្រាប់ជំនួយក្នុងការយល់ដឹង។

### តម្រូវការមុន

អ្នកគួរតែលេខពេញចិត្តទៅនឹងរចនាសម្ព័ន្ធទិន្នន័យផ្លែគុជដែលយើងកំពុងពិនិត្យ។ អ្នកអាចរកឃើញវាត្រូវបានបញ្ចូលរួច និងបានបើកស្អាតក្នុងឯកសារ _notebook.ipynb_ របស់មេរៀននេះ។ ក្នុងឯកសារ តម្លៃផ្លែគុជត្រូវបានបង្ហាញជាតម្លៃភាគតំណាងមួយក្នុង DataFrame ថ្មី។ សូមប្រាកដថាអ្នកអាចរត់ notebooks ទាំងនេះនៅក្នុង kernel នៃ Visual Studio Code។

### ការរៀបចំ

ដើម្បីរំលឹក អ្នកកំពុងផ្ទុកទិន្នន័យនេះដើម្បីសួរចំលើយពីវា។

- ពេលណាជាពេលល្អបំផុតក្នុងការទិញផ្លែគុជ?
- តម្លៃដែលខ្ញុំអាចរំពឹងទុក្ខប្រអប់ផ្លែគុជតូចមួយជាអ្វី?
- តើខ្ញុំគួរទិញវានៅក្នុងធុងកន្លែងកន្លះបាសែល ឬប្រអប់ ១ ១/៩ បាសែល?
យើងសូមបន្តស្រាវជ្រាវក្នុងទិន្នន័យនេះ។

ក្នុងមេរៀនមុន អ្នកបានបង្កើត DataFrame របស់ Pandas ហើយបញ្ចូលវាជាមួយផ្នែកមួយនៃ dataset ដើម បម្រែបម្រួលតម្លៃដោយបាសែល។ ប៉ុន្តែដោយធ្វើការនេះ អ្នកអាចទទួលបានតែប្រហែល៤០០ចំណុចទិន្នន័យ និងសម្រាប់ខែរដូវស្លឹកឈើជ្រុះតែប៉ុណ្ណោះ។

សូមមើលទៅទិន្នន័យដែលយើងបានបញ្ចូលរួចក្នុង notebook រួមជាមួយមេរៀននេះ។ ទិន្នន័យបានបញ្ចូលរួចហើយ និងមានការ scatterplot ដំបូងបង្ហាញតាមខែ។ ប្រហែលជាយើងអាចទទួលបានព័ត៌មានលម្អិតមួយចំនួនពីធម្មជាតិនៃទិន្នន័យដោយការសម្អាតវាបន្ថែមទៀត។

## ខ្សែ regression ស្រប

ដូចដែលអ្នកបានរៀនក្នុងមេរៀនទី ១ គោលដៅនៃលំហាត់ linear regression គឺដើម្បីអាចគូសខ្សែដូចខាងក្រោម៖

- **បង្ហាញទំនាក់ទំនងអថេរ**។ បង្ហាញទំនាក់ទំនងរវាងអថេរ
- **ធ្វើការទាយទោល**។ ធ្វើការទាយទោលបានត្រឹមត្រូវលើទីតាំងនៃចំណុចទិន្នន័យថ្មីមួយ ទាក់ទងទៅខ្សែស្របនោះ។

វាមានទំាងជាគន្លងនៃ **Least-Squares Regression** ដើម្បីគូសខ្សែបែបនេះ។ ពាក្យ "Least-Squares" មានន័យថា ជំហាននៃការកាត់បន្ថយកំហុសសរុបនៅក្នុងម៉ូដែល។ សម្រាប់ចំណុចទិន្នន័យរាល់ចំណុច យើងវាស់ចម្ងាយបញ្ឈរ (ហៅថា residual) រវាងចំណុចពិត និងខ្សែ regression។

យើងធ្វើការ​លោតបន្ធោងចម្ងាយនេះសម្រាប់មូលហេតុសំខាន់ ២៖

1. **ទំហំជាងទិសដៅ៖** យើងចង់ចាត់ទុកកំហុស -៥ ដូចគ្នានឹងកំហុស +៥។ ការ​លោតបន្ធោងធ្វើឲ្យតម្លៃទាំងអស់ទៅជាអវិជ្ជមាន។  

2. **ដាក់ពិន័យចំពោះចំណុចខុសប្លែក៖** ការលោតបន្ធោងធ្វើឲ្យកំហុសធំហើយមានទំងន់បន្ថែម ហើយជំរុញឲ្យខ្សែស្របនៅជិតចំណុចដែលឆ្ងាយ។

ក្រោយមក យើងនឹងបូករួមតម្លៃលោតបន្ធោងទាំងអស់ទាំងនេះ។គោលដៅរបស់យើងគឺរកឃើញខ្សែនូវកន្លែងដែលសរុបចុងក្រោយកើតមានតិចបំផុត (តម្លៃតិចបំផុតដែលអាចមាន)—អាស្រ័យពីឈ្មោះ "Least-Squares"។

> **🧮 បង្ហាញគណិតវិទ្យា**
> 
> ខ្សែនេះហៅថា _line of best fit_ អាចបង្ហាញដោយ [សមីការ](https://en.wikipedia.org/wiki/Simple_linear_regression):
> 
> ```
> Y = a + bX
> ```
>
> `X` គឺជា "អថេរពន្យល់"។ `Y` គឺជា "អថេរអាស្រ័យ"។ ចំហាយខ្សែគឺ `b` ហើយ `a` គឺជា y-intercept ដែលមានន័យថាតម្លៃ `Y` នៅពេល `X = 0`។
>
>![calculate the slope](../../../../translated_images/km/slope.f3c9d5910ddbfcf9.webp)
>
> ជំហានដំបូង គណនាចំហាយ `b`។ រូបតំណាងដោយ [Jen Looper](https://twitter.com/jenlooper)
>
> ក្នុងពាក្យផ្សេងៗ ហើយយោងទៅលើសំណួរដើមរបស់ទិន្នន័យផ្លែគុជ៖ "ទាយតម្លៃផ្លែគុជតាមបាសែលក្នុងមួយខែ", `X` នឹងបញ្ចេញតម្លៃតម្លៃ និង `Y` នឹងបញ្ចេញខែការលក់។
>
>![complete the equation](../../../../translated_images/km/calculation.a209813050a1ddb1.webp)
>
> គណនាតម្លៃ Y។ បើអ្នកកំពុងបង់ប្រាក់ប្រហែល $4 វា​គួរតែជាខែមេសា! រូបតំណាងដោយ [Jen Looper](https://twitter.com/jenlooper)
>
> គណិតវិទ្យាដែលគណនាចំពោះខ្សែនេះត្រូវបង្ហាញចំហាយខ្សែ ហើយវាក៏អាស្រ័យលើ intercept ឬទីតាំង `Y` នៅពេលដែល `X = 0`។
>
> អ្នកអាចមើលវិធីសាស្ត្រគណនាតម្លៃទាំងនេះក្រៅពី [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html)។ សូមទៅសំណេះសំណាល [Least-squares calculator](https://www.mathsisfun.com/data/least-squares-calculator.html) ដើម្បីមើលការប៉ះពាល់នៃតម្លៃចំនួនទៅលើខ្សែ។

## ការចងក្រង

ពាក្យមួយទៀតដែលត្រូវយល់គឺ **អនុផលចងក្រង (Correlation Coefficient)** រវាងអថេរ X និង Y ប្រាប់។ ដោយប្រើ scatterplot អ្នកអាចឃើញអនុផលចងក្រងនេះបានរហ័ស។ ពេល scatterplot មានចំណុចកន្លងទៅតាមខ្សែស្រប មានអនុផលចងក្រងខ្ពស់ ប៉ុន្តែពេល scatterplot មានចំណុចចែកចាយជុំវិញលើទីលានសរុប មានអនុផលចងក្រងទាប។

ម៉ូដែល linear regression ដែលល្អ គឺម៉ូដែលមានអនុផលចងក្រងខ្ពស់ (ជិត 1 មិនមែន 0) ដោយប្រើវិធី Least-Squares Regression ជាមួយខ្សែ regression ។

✅ រាជធានី notebook ដែលបញ្ជាក់មេរៀននេះ និងមើល scatterplot រវាង ខែក្រោមតម្លៃ។ តើទិន្នន័យដែលភ្ជាប់ខែក្រោមតម្លៃសម្រាប់ការលក់ផ្លែគុជមានអនុផលចងក្រងខ្ពស់ ឬទាប តាមការបកស្រាយរូបភាពរបស់អ្នក? តើវាប្រែប្រួលបើប្រើវាស់វែងលម្អិតជាងខែដូចជា *ថ្ងៃឆ្នាំ* (ជាចំនួនថ្ងៃចាប់ពីដើមឆ្នាំ)?

ក្នុងកូដខាងក្រោមវាយើងនឹងសន្មតថាយើងបានបន្ថែមសម្អាតទិន្នន័យ ហើយទទួលបាន DataFrame ឈ្មោះ `new_pumpkins` ដូចខាងក្រោម៖

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> កូដសម្រាប់សម្អាតទិន្នន័យអាចរកបានក្នុង [`notebook.ipynb`](notebook.ipynb)។ យើងបានអនុវត្តជំហានសម្អាតដដែលដូចមេរៀនមុន ហើយបានគណនាគូបន្ទាត់ `DayOfYear` ដោយប្រើនិយមន័យ​ខាងក្រោម៖ 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

ឥឡូវនេះដែលអ្នកមានការយល់ដឹងក្នងគណិតវិទ្យាក្រោយ linear regression យើងសាកល្បងបង្កើតម៉ូដែល Regression ដើម្បីមើលថាតើយើងអាចទាយថាតើកញ្ចប់ផ្លែគុជណាអាចមានតម្លៃល្អបំផុត។ មនុស្សដែលទិញផ្លែគុជសម្រាប់បរិវេទបុណ្យអាចចង់បានព័ត៌មាននេះដើម្បីអាចធ្វើអោយការទិញកញ្ចប់ផ្លែគុជមានប្រសិទ្ធភាពល្អបំផុតសម្រាប់បរិវេទ។

## ការស្វែងរកការចងក្រង

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 ចុចរូបភាពខាងលើសម្រាប់មើលវីដេអូសង្ខេបពីការស្វែងរកការចងក្រង។

ពីមេរៀនមុន អ្នកប្រហែលជាបានឃើញថាតម្លៃមធ្យមរបស់ខែផ្សេងៗមើលទៅដូចខាងក្រោម៖

<img alt="Average price by month" src="../../../../translated_images/km/barchart.a833ea9194346d76.webp" width="50%"/>

វាសម្លឹងបង្ហាញថា គួរតែមានការចងក្រងខ្លះ ហើយយើងអាចព្យាយាមបណ្តុះម៉ូដែល linear regression ដើម្បីទាយទំនាក់ទំនងរវាង `Month` និង `Price` ឬ `DayOfYear` និង `Price`។ នេះគឺជា scatter plot បង្ហាញទំនាក់ទំនងចុងក្រោយ៖

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/km/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

មកមើលថាតើមានការចងក្រងជាមួយ `corr` function មែនទេ៖

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

វាបង្ហាញថាការចងក្រងមានតិច (-0.15 សម្រាប់ `Month` និង -0.17 សម្រាប់ `DayOfYear`) ប៉ុន្តែអាចមានទំនាក់ទំនងសំខាន់ផ្សេងទៀត។ វា​បង្ហាញថាមានក្រុមតម្លៃផ្សេងៗដែលទាក់ទងទៅនឹងប្រភេទផ្លែគុជផ្សេងគ្នា។ ដើម្បីបញ្ជាក់សំណងនេះ សូមគូសផ្ទាំងផ្តោតលើប្រភេទផ្លែគុជដោយពណ៌ផ្សេងៗ។ ដោយផ្តល់ផារាម៉ែត្រ `ax` ទៅ `scatter` អ្នកអាចគូសចំណុចទាំងអស់នៅលើក្រាហ្វមួយដូចគ្នា៖

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/km/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

ការស៊ើបអង្កេតរបស់យើងបង្ហាញថាប្រភេទផ្លែគុជមានឥទ្ធិពលច្រើនជាងលើតម្លៃសរុបជាងថ្ងៃដែលលក់។ យើងអាចឃើញវាជាមួយក្រាហ្វបារ៖

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Bar graph of price vs variety" src="../../../../translated_images/km/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

សូមផ្តោតសំខាន់បច្ចុប្បន្នលើប្រភេទផ្លែគុជតែមួយគត់ 'pie type' ហើយមើលថាតើថ្ងៃមានឥទ្ធិពលដូចម្ដេចចំពោះតម្លៃ៖

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/km/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

បើយើងគណនាការចងក្រងរវាង `Price` និង `DayOfYear` ដោយប្រើ `corr` function ឥឡូវនេះ អ្នកនឹងបានតម្លៃប្រហែល `-0.27` - មានន័យថាការបណ្តុះម៉ូដែលទាយទោលរួចហើយមានហត្ថពលន៍។

> មុនពេលបណ្តុះម៉ូដែល linear regression វាម៉ត់ចត់ឲ្យប្រាកដថាទិន្នន័យស្អាត។ Linear regression មិនប្រសើរនៅពេលមានតម្លៃទំនេរ ដូច្នេះវាអាចមានអត្ថប្រយោជន៍ក្នុងការបោះបង់ជាសំណុំចំលងទាំងអស់៖

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

វិធីមួយផ្សេងទៀតគឺបំពេញតម្លៃទំនេរទាំងនោះជាមួយតម្លៃមធ្យមពីជួរឈរដែលទាក់ទង។

## Regression ស្រួលៗ linear

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 ចុចរូបភាពខាងលើសម្រាប់មើលវីដេអូសង្ខេបពី linear និង polynomial regression។

ដើម្បីបណ្តុះម៉ូដែល Linear Regression របស់យើង យើងនឹងប្រើបណ្ណាល័យ **Scikit-learn** ។

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

យើងចាប់ផ្ដើមដោយបំបែកកម្រិតបញ្ចូល (features) និងលទ្ធផលដែលរំពឹងទុក (label) ទៅក្នុង array numpy ផ្សេងៗ៖

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> សូមចំណាំថាយើងត្រូវបានធ្វើ `reshape` លើទិន្នន័យបញ្ចូល ដើម្បីឲ្យ package Linear Regression យល់បានត្រឹមត្រូវ។ Linear Regression រងចាំ array 2D ជាកម្រិតបញ្ចូល ដែលជារៀងរាល់ជួរឈរនៃ array តំណាងថាលក្ខណៈបញ្ចូលមួយ vector។ សម្រាប់ករណីយើង មានតែបញ្ចូលមួយ ចាំបាច់ត្រូវមាន array ទំហំ N×1 ដែល N ជា​ចំនួនទិន្នន័យ។

បន្ទាប់មក យើងត្រូវបំបែកទិន្នន័យជាក្រុមបណ្តុះបណ្តាលនិងសាកល្បង ដើម្បីពិនិត្យម៉ូដែលបន្ទាប់ពីបណ្តុះបណ្តាល៖

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

ចុងក្រោយ ការបណ្តុះណែនាំម៉ូដែល Linear Regression ពិតប្រាកដត្រូវការចំនួនកូដប៉ុន្មានជួរប៉ុណ្ណោះ។ យើងកំណត់អំពើ `LinearRegression` ហើយតភ្ជាប់វាជាមួយទិន្នន័យដោយវិធីសាស្ត្រ `fit`៖

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

វត្ថុ `LinearRegression` បន្ទាប់ពី​បានធ្វើការបណ្តុះ​បណ្តាល (`fit`-ting)​ មានគោលបំណងសមាមាត្រទាំងអស់នៃសមីការបញ្ចេញលទ្ធផល ដែល​អាច​ចូលដំណើរការបាន​ដោយ​ប្រើ​អចលនវត្ថុ `.coef_`។ នៅក្នុងករណីរបស់​យើង មានគោលបំណងតែមួយ ប៉ុន្តែគួរតែប្រហែលជា `-0.017`។ នេះមានន័យថាកម្លៃ​តម្លៃ​ព្រៃដូចជានឹងធ្លាក់បន្តិចបន្តួចជាមួយ​ពេលវេលា ប៉ុន្តែមិនច្រើនពេក យ៉ាងតិចប្រហែល 2 សេន្ទក្នុងមួយថ្ងៃ។ យើងក៏អាចចូលដំណើរការចំណុច​ឆ្លុះកាត់​នៃសមីការបញ្ចេញលទ្ធផលជាមួយអ័ក្ស Y បានដោយ​ប្រើ `lin_reg.intercept_` - វានឹងប្រហែលជា `21` ក្នុងករណីយើង បង្ហាញពី​តម្លៃ​នៅដើម​ឆ្នាំ។

ក្នុងការមើលថាតើ​ម៉ូឌែលរបស់យើងមានភាពត្រឹមត្រូវប៉ុណ្ណា យើងអាចទាយតម្លៃលើទិន្នន័យសាកល្បង ហើយបន្ទាប់មកវាស់ឈរភាពជិតស្និទ្ធរបស់ការទាយរបស់យើងនឹងតម្លៃដែលរំពឹងទុក។ នេះអាចធ្វើបានដោយប្រើមាត្រដ្ឋាន root mean square error (RMSE) ដែលជាចម្ងាយរបស់មធ្យមនៃភាពខុសគ្នាប្រក់គ្នាជារូបបូកមួយរវាងតម្លៃរំពឹងនិងតម្លៃទាយ។

```python
pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')
```
  
កំហុសរបស់យើងហាក់ដូចជាប្រហែល 2 ពិន្ទុ ដែលប្រហែលជាគឺ ~17%។ មិនល្អធ្វើទេ។ មាត្រដ្ឋានជួនកាលទៀតនៃគុណភាពម៉ូឌែលគឺ **coefficient of determination** ដែលអាចទទួលបានដូចខាងក្រោម៖

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
  
បើតម្លៃគឺ 0 មានន័យថា ម៉ូឌែលមិនយកទិន្នន័យបញ្ចូលចូលក្នុងការគិតឡើយ ហើយដំណើរការជា *អ្នកទាយបន្ទាត់អាក្រក់បំផុត* ដែលគ្រាន់តែជាមធ្យមនៃលទ្ធផល។ តម្លៃ 1 មានន័យថាយើងអាចទាយបានអ្វីគ្រប់យ៉ាងយ៉ាងត្រឹមត្រូវ។ ក្នុងករណីយើង គុណហានិភ័យគឺប្រហែល 0.06 ដែលច្រើនទាប។

យើងក៏អាចគូរបង្ហាញទិន្នន័យសាកល្បងរួមជាមួយបន្ទាត់សមីការបញ្ចេញលទ្ធផល ដើម្បីមើលថាតើសមីការបញ្ចេញលទ្ធផលដំណើរការយ៉ាងដូចម្តេច៖

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```
  
<img alt="Linear regression" src="../../../../translated_images/km/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## សមីការបញ្ចេញលទ្ធផល Polynomial Regression

ប្រភេទមួយទៀតនៃសមីការបញ្ចេញលទ្ធផលបន្ទាត់គឺ សមីការ Polynomial Regression។ ខណៈពេលដែលខ្លះមានទំនាក់ទំនងបន្ទាត់រវាងអថេរផ្សេងៗ - ទំហំប៉ូមពេញមួយធំពីលទ្ធផលកាន់តែលើកម្លៃ យ៉ាងណាក៏ដោយ ទំនាក់ទំនងទាំងនេះមិនអាចគូរជាបន្ទាត់ផ្លាត់ ឬបន្ទាត់ត្រង់បានទេ។

✅ នៅទីនេះមាន [ឧទាហរណ៍បន្ថែម](https://online.stat.psu.edu/stat501/Lesson09#polynomial-regression-examples) នៃទិន្នន័យដែលអាចប្រើ Polynomial Regression

សូមមើលទំនាក់ទំនងរវាង Date និង Price ជាទៀត។ តើ scatterplot នេះមើលទៅគួរត្រូវបានវិភាគដោយបន្ទាត់ត្រង់តែមួយ? តើតម្លៃអាចប្រែប្រួលបានទេ? ក្នុងករណីនេះ អ្នកអាចសាកល្បងសមីការ polynomial regression។

✅ Polynomials គឺជាអប្បបរមាណ គណិតវិទ្យា ដែលប្រហែលជាមានអថេរមួយ ឬច្រើន និងគោលបំណងជាសំណុំ

Polynomial regression បង្កើតបន្ទាត់ពោងដើម្បីផ្គួតផ្គងទិន្នន័យមិនបន្ទាត់បានល្អជាងមុន។ ក្នុងករណីរបស់យើង ប្រសិនបើបញ្ចូលអថេរ `DayOfYear` ការប្រកួតកំណត់ក្នុងទិន្នន័យចូលទាំងមូល យើងគួរតែអាចធ្វើការបណ្តុះក្នុងទិន្នន័យជាមួយព្រីលារតិសន្ទិលក្រោមរូបមន្ត Parabolic ដែលមានអប្បបរមា​នៅចំណុចមួយក្នុងឆ្នាំ។

Scikit-learn មាន API [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) ប្រយោជន៍ក្នុងការបញ្ចូលជំហ៊ានផ្សេងៗនៃការបញ្ចូលទិន្នន័យជាមួយគ្នា។ ​**pipeline** គឺជាច្រវាក់នៃ **estimators**។ ក្នុងករណីយើង យើងនឹងបង្កើត pipeline ដែលដំបូងបន្ថែមលក្ខណៈ polynomial ទៅម៉ូឌែល ហើយបន្ទាប់បណ្តុះ regression៖

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```
  
ការប្រើ `PolynomialFeatures(2)` មានន័យថាយើងនឹងបញ្ចូល polynomials កំរិតទី ២ ទាំងអស់ពីទិន្នន័យបញ្ចូល។ ក្នុងករណីយើង នឹងមានតែមួយគឺគឺ `DayOfYear`<sup>2</sup> ប៉ុន្តែជិតជាមួយអថេរចំនួនពីរ X និង Y វានឹងបន្ថែម X<sup>2</sup>, XY និង Y<sup>2</sup>។ យើងអាចប្រើ polynomials កំរិតខ្ពស់ជាងនេះបានផងដែរ ប្រសិនបើយើងចង់។

Pipeline អាចប្រើបានយ៉ាងដូចគ្នាទៅនឹងវត្ថុ `LinearRegression` ដើម, គឺ៖ យើងអាច `fit` pipeline ហើយបន្ទាប់មកប្រើ `predict` ដើម្បីទទួលលទ្ធផលទាយ៖

```python
pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```
  
ដើម្បីគូរបន្ទាត់ពោងស្រទាប់ម៉ាត់ អ្នកប្រើ `np.linspace` ដើម្បីបង្កើតជួរចាប់ផ្តើមតម្លៃបញ្ចូលស្មើគ្នា ជាងការគូរតាមទិន្នន័យសាកល្បងដែលមិនមានលំដាប់ (ដែលនឹងបង្កើតបន្ទាត់ zigzag):

```python
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
y_range = pipeline.predict(X_range)

plt.scatter(X_test, y_test)
plt.plot(X_range, y_range)
```
  
នេះគឺជាបន្ទាត់បង្ហាញទិន្នន័យសាកល្បង និងបន្ទាត់សន្និដ្ឋាន៖

<img alt="Polynomial regression" src="../../../../translated_images/km/poly-results.ee587348f0f1f60b.webp" width="50%" />

ប្រើ Polynomial Regression អាចទទួលបានកម្លាំង RMSE ទាបជាង និងឈរភាពកំណត់ខ្ពស់ជាង ប៉ុន្តែមិនច្រើនពេកទេ។ យើងត្រូវយកចិត្តទុកដាក់លើលក្ខណៈផ្សេងទៀត!

> អ្នកអាចមើលឃើញថាតម្លៃបន្ទះប៉ូមគឺអប្បបរមា ខណៈពេលដែលនៅជិតបុណ្យ Halloween។ តើអ្នកអាចពន្យល់បែបណា?

🎃 អបអរសាទរ! អ្នកបានបង្កើតម៉ូឌែលដែលអាចជួយទាយតម្លៃបន្ទះប៉ូមសម្រាប់នំបុ័ង។ អ្នកប្រហែលជាអាចធ្វើឡើងវិញសម្រាប់ប្រភេទប៉ូមទាំងអស់ ប៉ុន្តែវាជារឿងធុញនឿយ។ យើងត្រូវស្យៀនការយល់ពីរបៀបយកភាពខុសគ្នាប្រភេទប៉ូមចូលក្នុងម៉ូឌែលរបស់យើង!

## លក្ខណៈកាតេហ្គរី

នៅក្នុងពិភពដ៏ល្អបំផុត យើងចង់អាចទាយតម្លៃរបស់ប៉ូមប្រភេទផ្សេងៗ ដោយប្រើម៉ូឌែលដដែល។ ទោះយ៉ាងណា ជួរឈរដែលមានឈ្មោះ `Variety` មានភាពខុសពីជួរឈរ‌ដូចជា `Month` ពីព្រោះវាមានតម្លៃមិនមែនជាលេខ។ ជួរឈរដូចនេះហៅថា **categorical**។

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 ចុចលើរូបភាពខាងលើសម្រាប់វីដេអូសង្ខេបអំពីការប្រើលក្ខណៈកាតេហ្គរី។

នេះអ្នកអាចមើលឃើញថាតម្លៃមធ្យមពឹងផ្អែកលើប្រភេទ៖

<img alt="Average price by variety" src="../../../../translated_images/km/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

ដើម្បីយកប្រភេទចូលក្នុងការគិតយើងត្រូវបំលែងវាទៅជាទម្រង់លេខ ឬ **encode** វា។ មានវិធីជាច្រើនដែលអាចធ្វើបាន៖

* ការបំលែងលេខសាមញ្ញ (**numeric encoding**) នឹងបង្កើតតារាងនៃប្រភេទផ្សេងៗ ហើយបន្ទាប់មកប្តូរឈ្មោះប្រភេទជាកន្លែងក្នុងតារាង។ វាមិនមែនជាគំនិតល្អសម្រាប់សមីការបញ្ចេញលទ្ធផលបន្ទាត់ ទេ ព្រោះសមីការបញ្ចេញលទ្ធផលបន្ទាត់យកតម្លៃលេខនៃសន្ទស្សន៍ចូលរួមក្នុងការគណនា ហើយបូកបន្ថែមចូល ធ្វើការបូកគុណជាមួយគោលបំណងខ្លះ។ ក្នុងករណីយើង ទំនាក់ទំនងរវាងលេខសន្ទស្សន៍ និងតម្លៃពិតមិនបន្ទាត់ទេ ទោះបីយើងធ្វើអោយតំណាងលេខត្រូវត្រួតត្រាក្នុងលំដាប់ច្បាស់ក៏ដោយ។
* ការបំលែងប្រភេទជា **one-hot encoding** នឹងប្តូរជួរឈរ `Variety` ទៅជាជួរឈរបួនផ្សេងៗ មួយសម្រាប់ប្រភេទនីមួយៗ។ ត្រង់ជួរឈរនីមួយនឹងមានតម្លៃ `1` ប្រសិនបើជួរដេកពាក់ព័ន្ធជាប្រភេទនោះ ហើយ `0` ផ្សេងៗគ្នា។ នេះមានន័យថានឹងមានគោលបំណងបួនក្នុងសមីការបញ្ចេញលទ្ធផលឲ្យតម្រូវតម្លៃចាប់ផ្តើម (ឬ “តម្លៃបន្ថែម”) សម្រាប់ប្រភេទប៉ូមនីមួយៗ។

កូដខាងក្រោមបង្ហាញពីរបៀបដែលយើងអាចបំលែងប្រភេទជាទម្រង់ one-hot encoded៖

```python
pd.get_dummies(new_pumpkins['Variety'])
```
  
 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE  
----|-----------|-----------|--------------------------|----------  
70 | 0 | 0 | 0 | 1  
71 | 0 | 0 | 0 | 1  
... | ... | ... | ... | ...  
1738 | 0 | 1 | 0 | 0  
1739 | 0 | 1 | 0 | 0  
1740 | 0 | 1 | 0 | 0  
1741 | 0 | 1 | 0 | 0  
1742 | 0 | 1 | 0 | 0  

ដើម្បីបណ្តុះសមីការបញ្ចេញលទ្ធផលបន្ទាត់ប្រើបណ្តុះប្រភេទ one-hot encoded ជា input យើងគ្រាន់តែត្រូវផ្តល់ `X` និង `y` ដោយត្រឹមត្រូវ៖

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```
  
Remaining កូដដូចគ្នានឹងអ្វីដែលយើងប្រើខាងលើសម្រាប់បណ្តុះ Linear Regression។ ប្រសិនបើអ្នកសាកល្បង អ្នកនឹងឃើញថាកំហុសគន្លងគំនូសមធ្យមនៃ squared error ជិតតំលៃដដែល ប៉ុន្តាយើងទទួលបាន coefficient of determination ខ្ពស់ជាង (~77%)។ ដើម្បីទទួលបានការទាយខ្ពស់ជាងនេះទៀត យើងអាចយកលក្ខណៈ categorical ជាច្រើនជាមួយ លក្ខណៈលេខ ដូចជា `Month` ឬ `DayOfYear`។ ដើម្បីធ្វើអោយជាសំណុំលក្ខណៈធំមួយ យើងអាចប្រើ `join`៖

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```
  
នៅទីនេះយើងក៏យកចិត្តទុកដាក់លើ `City` និងប្រភេទ `Package` ផងដែរ ដែលផ្តល់ RMSE 2.84 (១០.៥%) និង coefficient determination 0.94!

## សម្រង់ទាំងអស់ជាមួយគ្នា

ដើម្បីបង្កើតម៉ូឌែលល្អបំផុត យើងអាចប្រើទិន្នន័យរួម (categorical one-hot encoded + numeric) ពីឧទាហរណ៍ខាងលើ រួមជាមួយ Polynomial Regression។ នេះជាកូដពេញលេញសម្រាប់សម្រួលអ្នក៖

```python
# បង្កើតទិន្នន័យសម្រាប់បណ្ដុះបណ្ដាល
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# បំបែកទិន្នន័យចេញជាក្រុមបណ្ដុះបណ្ដាល និងសំណួរប្រលង
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# កំណត់ និងបណ្ដុះបណ្ដាលបាយឡុង
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# ប្រោសចេញលទ្ធផលសម្រាប់ទិន្នន័យសាកល្បង
pred = pipeline.predict(X_test)

# គណនាតម្លៃ RMSE និងកំណត់ការសំរេច
rmse = mean_squared_error(y_test, pred, squared=False)
print(f'RMSE: {rmse:3.3} ({rmse/pred.mean()*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```
  
នេះគួរតែផ្តល់ឲ្យយើងបាន coefficient determination ល្អបំផុតប្រហែល ៩៧% និង RMSE=2.23 (~៨% កំហុស).

| ម៉ូឌែល | RMSE | coefficient determination |  
|-------|-----|-------------------------|  
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |  
| `DayOfYear` Polynomial | 2.73 (17.0%) | 0.08 |  
| `Variety` Linear | 5.24 (19.7%) | 0.77 |  
| លក្ខណៈទាំងអស់ Linear | 2.84 (10.5%) | 0.94 |  
| លក្ខណៈទាំងអស់ Polynomial | 2.23 (8.25%) | 0.97 |

🏆 សូមអបអរសាទរ! អ្នកបានបង្កើតម៉ូឌែល Regression បួនប្រភេទក្នុងមេរៀនមួយ ហើយកែលម្អគុណភាពម៉ូឌែលដល់ ៩៧%។ នៅផ្នែកចុងក្រោយនៃមេរៀន Regression អ្នកនឹងរៀនអំពី Logistic Regression ដើម្បីកំណត់ប្រភេទ។

---  
## 🚀ការប្រកួតប្រជែង

សាកល្បងអថេរផ្សេងៗក្នុង notebook នេះ ដើម្បីមើលថាតើ​ធរណីមាត្រភ្ជាប់​សម្រាប់ការទាយម៉ូឌែលមានភាពទាក់ទងយ៉ាងដូចម្តេច។

## [សំណួរប្រឡងបន្ទាប់មេរៀន](https://ff-quizzes.netlify.app/en/ml/)

## ការពិនិត្យឡើងវិញ និងរៀនដោយខ្លួនឯង

នៅក្នុងមេរៀននេះ យើងបានរៀនអំពី Linear Regression។ មានប្រភេទសមីការបញ្ចេញលទ្ធផលសំខាន់ផ្សេងទៀត។ សូមអានអំពី Stepwise, Ridge, Lasso និង Elasticnet techniques។ មេរៀនល្អសម្រាប់អនុវត្តន៍បន្ថែមគឺ [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## បេសកកម្ម

[បង្កើតម៉ូឌែល](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ការបដិសេធ**:  
ឯកសារនេះត្រូវបានបកប្រែដោយប្រើសេវាកម្មបកប្រែ AI [Co-op Translator](https://github.com/Azure/co-op-translator)។ ខណៈពេលដែលយើងខិតខំប្រឹងប្រែងសម្រាប់ភាពត្រឹមត្រូវ សូមយល់ឲ្យបានថាការបកប្រែដោយស្វ័យប្រវត្តិអាចមានកំហុស ឬភាពមិនត្រឹមត្រូវ។ ឯកសារដើមក្នុងភាសាមាតុភូមិគួរត្រូវបានគេយកជាជាតំណរ ម៉ោងសម្រាប់ព័ត៌មានសំខាន់ៗ គេណែនាំឲ្យប្រើការបកប្រែដោយអ្នកជំនាញ។ យើងមិនទទួលខុសត្រូវចំពោះការយល់ច្រឡំ ឬការបកស្រាយខុសកើតឡើងពីការប្រើប្រាស់ការបកប្រែនេះឡើយ។
<!-- CO-OP TRANSLATOR DISCLAIMER END -->