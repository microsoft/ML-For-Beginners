<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-05T15:08:49+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "sw"
}
-->
# Jenga Mfano wa Regression kwa kutumia Scikit-learn: Njia Nne za Regression

![Picha ya habari ya regression ya mstari dhidi ya polynomial](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Picha ya habari na [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Jaribio la kabla ya somo](https://ff-quizzes.netlify.app/en/ml/)

> ### [Somo hili linapatikana kwa R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Utangulizi 

Hadi sasa umechunguza regression ni nini kwa kutumia data ya mfano iliyokusanywa kutoka kwenye seti ya data ya bei ya maboga ambayo tutatumia katika somo hili. Pia umeiona kwa kutumia Matplotlib.

Sasa uko tayari kuchunguza zaidi regression kwa ML. Wakati uonekanaji wa data unakusaidia kuelewa data, nguvu halisi ya Kujifunza kwa Mashine inatokana na _kufundisha mifano_. Mifano hufundishwa kwa data ya kihistoria ili kunasa utegemezi wa data kiotomatiki, na hukuwezesha kutabiri matokeo kwa data mpya ambayo mfano haujawahi kuona hapo awali.

Katika somo hili, utajifunza zaidi kuhusu aina mbili za regression: _regression ya mstari wa kawaida_ na _regression ya polynomial_, pamoja na baadhi ya hesabu zinazohusiana na mbinu hizi. Mifano hiyo itatuwezesha kutabiri bei za maboga kulingana na data tofauti za pembejeo.

[![ML kwa wanaoanza - Kuelewa Regression ya Mstari](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML kwa wanaoanza - Kuelewa Regression ya Mstari")

> 🎥 Bofya picha hapo juu kwa muhtasari mfupi wa regression ya mstari.

> Katika mtaala huu, tunadhania maarifa ya chini ya hesabu, na tunalenga kuifanya ipatikane kwa wanafunzi kutoka nyanja nyingine, kwa hivyo angalia maelezo, 🧮 maelezo ya hesabu, michoro, na zana nyingine za kujifunza kusaidia ufahamu.

### Mahitaji ya Awali

Unapaswa kuwa na ufahamu sasa wa muundo wa data ya maboga tunayochunguza. Unaweza kuipata ikiwa imepakiwa tayari na kusafishwa katika faili ya _notebook.ipynb_ ya somo hili. Katika faili hiyo, bei ya maboga inaonyeshwa kwa kila busheli katika fremu mpya ya data. Hakikisha unaweza kuendesha daftari hizi katika kernels kwenye Visual Studio Code.

### Maandalizi

Kama ukumbusho, unapakia data hii ili kuuliza maswali kuihusu. 

- Ni wakati gani bora wa kununua maboga? 
- Ni bei gani naweza kutarajia kwa sanduku la maboga madogo?
- Je, ninunue kwa vikapu vya nusu busheli au kwa sanduku la busheli 1 1/9?
Tuendelee kuchimba data hii.

Katika somo lililopita, uliunda fremu ya data ya Pandas na kuijaza na sehemu ya seti ya data ya awali, ukistandadisha bei kwa busheli. Kwa kufanya hivyo, hata hivyo, uliweza tu kukusanya takriban alama 400 za data na tu kwa miezi ya vuli.

Angalia data ambayo tumepakia tayari katika daftari linaloambatana na somo hili. Data imepakiwa tayari na mchoro wa awali wa alama za kutawanyika umechorwa kuonyesha data ya mwezi. Labda tunaweza kupata maelezo zaidi kuhusu asili ya data kwa kuisafisha zaidi.

## Mstari wa regression ya mstari

Kama ulivyojifunza katika Somo la 1, lengo la zoezi la regression ya mstari ni kuweza kuchora mstari ili:

- **Kuonyesha uhusiano wa vigezo**. Kuonyesha uhusiano kati ya vigezo
- **Kutabiri matokeo**. Kufanya utabiri sahihi wa mahali ambapo alama mpya ya data ingeangukia kuhusiana na mstari huo.

Ni kawaida kwa **Regression ya Least-Squares** kuchora aina hii ya mstari. Neno 'least-squares' linamaanisha kwamba alama zote za data zinazozunguka mstari wa regression zinasikweya na kisha kuongezwa. Kwa hali bora, jumla hiyo ya mwisho inapaswa kuwa ndogo iwezekanavyo, kwa sababu tunataka idadi ndogo ya makosa, au `least-squares`.

Tunafanya hivyo kwa sababu tunataka kuunda mstari ambao una umbali wa chini kabisa wa jumla kutoka kwa alama zetu zote za data. Pia tunasikweya maneno kabla ya kuyaongeza kwa sababu tunajali ukubwa wake badala ya mwelekeo wake.

> **🧮 Nionyeshe hesabu** 
> 
> Mstari huu, unaoitwa _mstari wa kufaa zaidi_ unaweza kuonyeshwa na [mchoro wa equation](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` ni 'kigezo cha kueleza'. `Y` ni 'kigezo tegemezi'. Mteremko wa mstari ni `b` na `a` ni y-intercept, ambayo inahusu thamani ya `Y` wakati `X = 0`. 
>
>![hesabu mteremko](../../../../2-Regression/3-Linear/images/slope.png)
>
> Kwanza, hesabu mteremko `b`. Picha ya habari na [Jen Looper](https://twitter.com/jenlooper)
>
> Kwa maneno mengine, na kurejelea swali la awali la data ya maboga: "tabiri bei ya boga kwa busheli kwa mwezi", `X` ingerejelea bei na `Y` ingerejelea mwezi wa mauzo. 
>
>![kamilisha equation](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Hesabu thamani ya Y. Ikiwa unalipa karibu $4, lazima iwe Aprili! Picha ya habari na [Jen Looper](https://twitter.com/jenlooper)
>
> Hesabu inayokokotoa mstari lazima ionyeshe mteremko wa mstari, ambao pia unategemea intercept, au mahali ambapo `Y` iko wakati `X = 0`.
>
> Unaweza kuona mbinu ya hesabu ya maadili haya kwenye tovuti ya [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Pia tembelea [kikokotoo cha Least-squares](https://www.mathsisfun.com/data/least-squares-calculator.html) ili kuona jinsi maadili ya namba yanavyoathiri mstari.

## Uwiano

Neno moja zaidi la kuelewa ni **Kipengele cha Uwiano** kati ya vigezo vilivyotolewa vya X na Y. Kwa kutumia mchoro wa kutawanyika, unaweza kuona haraka kipengele hiki. Mchoro wenye alama za data zilizotawanyika kwa mstari mzuri una uwiano wa juu, lakini mchoro wenye alama za data zilizotawanyika kila mahali kati ya X na Y una uwiano wa chini.

Mfano mzuri wa regression ya mstari utakuwa ule wenye Kipengele cha Uwiano cha juu (karibu na 1 kuliko 0) kwa kutumia mbinu ya Least-Squares Regression na mstari wa regression.

✅ Endesha daftari linaloambatana na somo hili na uangalie mchoro wa kutawanyika wa Mwezi hadi Bei. Je, data inayohusisha Mwezi na Bei ya mauzo ya maboga inaonekana kuwa na uwiano wa juu au wa chini, kulingana na tafsiri yako ya kuona ya mchoro wa kutawanyika? Je, hilo linabadilika ikiwa unatumia kipimo cha kina zaidi badala ya `Mwezi`, mfano *siku ya mwaka* (yaani, idadi ya siku tangu mwanzo wa mwaka)?

Katika msimbo hapa chini, tutadhania kuwa tumesafisha data, na kupata fremu ya data inayoitwa `new_pumpkins`, inayofanana na ifuatayo:

ID | Mwezi | SikuYaMwaka | Aina | Jiji | Kifurushi | Bei ya Chini | Bei ya Juu | Bei
---|-------|-------------|------|------|----------|-------------|------------|-----
70 | 9 | 267 | AINA YA PIE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | AINA YA PIE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | AINA YA PIE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | AINA YA PIE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | AINA YA PIE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Msimbo wa kusafisha data unapatikana katika [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Tumefanya hatua zile zile za kusafisha kama katika somo lililopita, na tumekokotoa safu ya `SikuYaMwaka` kwa kutumia usemi ufuatao: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Sasa kwa kuwa unaelewa hesabu nyuma ya regression ya mstari, hebu tuunde Mfano wa Regression ili kuona kama tunaweza kutabiri ni kifurushi gani cha maboga kitakuwa na bei bora za maboga. Mtu anayenunua maboga kwa ajili ya shamba la likizo la maboga anaweza kutaka taarifa hii ili kuboresha ununuzi wao wa vifurushi vya maboga kwa shamba.

## Kutafuta Uwiano

[![ML kwa wanaoanza - Kutafuta Uwiano: Muhimu kwa Regression ya Mstari](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML kwa wanaoanza - Kutafuta Uwiano: Muhimu kwa Regression ya Mstari")

> 🎥 Bofya picha hapo juu kwa muhtasari mfupi wa uwiano.

Kutoka somo lililopita labda umeona kuwa bei ya wastani kwa miezi tofauti inaonekana kama hii:

<img alt="Bei ya wastani kwa mwezi" src="../2-Data/images/barchart.png" width="50%"/>

Hii inapendekeza kuwa kunaweza kuwa na uwiano, na tunaweza kujaribu kufundisha mfano wa regression ya mstari kutabiri uhusiano kati ya `Mwezi` na `Bei`, au kati ya `SikuYaMwaka` na `Bei`. Hapa kuna mchoro wa kutawanyika unaoonyesha uhusiano wa pili:

<img alt="Mchoro wa kutawanyika wa Bei dhidi ya Siku ya Mwaka" src="images/scatter-dayofyear.png" width="50%" /> 

Hebu tuone kama kuna uwiano kwa kutumia kazi ya `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Inaonekana kama uwiano ni mdogo sana, -0.15 kwa `Mwezi` na -0.17 kwa `SikuYaMwezi`, lakini kunaweza kuwa na uhusiano mwingine muhimu. Inaonekana kuna makundi tofauti ya bei yanayohusiana na aina tofauti za maboga. Ili kuthibitisha dhana hii, hebu tuchore kila aina ya boga kwa rangi tofauti. Kwa kupitisha parameter ya `ax` kwa kazi ya kuchora `scatter` tunaweza kuchora alama zote kwenye mchoro mmoja:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Mchoro wa kutawanyika wa Bei dhidi ya Siku ya Mwaka" src="images/scatter-dayofyear-color.png" width="50%" /> 

Uchunguzi wetu unapendekeza kuwa aina ina athari kubwa zaidi kwa bei ya jumla kuliko tarehe halisi ya mauzo. Tunaweza kuona hili kwa mchoro wa bar:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Mchoro wa bar wa bei dhidi ya aina" src="images/price-by-variety.png" width="50%" /> 

Hebu tuzingatie kwa sasa aina moja tu ya maboga, 'aina ya pie', na tuone athari gani tarehe inayo kwa bei:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Mchoro wa kutawanyika wa Bei dhidi ya Siku ya Mwaka" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Ikiwa sasa tutakokotoa uwiano kati ya `Bei` na `SikuYaMwaka` kwa kutumia kazi ya `corr`, tutapata kitu kama `-0.27` - ambayo inamaanisha kuwa kufundisha mfano wa utabiri kuna mantiki.

> Kabla ya kufundisha mfano wa regression ya mstari, ni muhimu kuhakikisha kuwa data yetu ni safi. Regression ya mstari haifanyi kazi vizuri na thamani zilizokosekana, kwa hivyo inafaa kuondoa seli zote tupu:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Njia nyingine itakuwa kujaza thamani hizo tupu na thamani za wastani kutoka safu husika.

## Regression Rahisi ya Mstari

[![ML kwa wanaoanza - Regression ya Mstari na Polynomial kwa kutumia Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML kwa wanaoanza - Regression ya Mstari na Polynomial kwa kutumia Scikit-learn")

> 🎥 Bofya picha hapo juu kwa muhtasari mfupi wa regression ya mstari na polynomial.

Ili kufundisha mfano wetu wa Regression ya Mstari, tutatumia maktaba ya **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Tunaanza kwa kutenganisha thamani za pembejeo (vipengele) na matokeo yanayotarajiwa (lebo) katika safu tofauti za numpy:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Kumbuka kuwa tulilazimika kufanya `reshape` kwenye data ya pembejeo ili kifurushi cha Regression ya Mstari kiielewe kwa usahihi. Regression ya Mstari inatarajia safu ya 2D kama pembejeo, ambapo kila safu ya safu inalingana na vector ya vipengele vya pembejeo. Katika kesi yetu, kwa kuwa tuna pembejeo moja tu - tunahitaji safu yenye umbo la N×1, ambapo N ni ukubwa wa seti ya data.

Kisha, tunahitaji kugawanya data katika seti za mafunzo na majaribio, ili tuweze kuthibitisha mfano wetu baada ya mafunzo:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Hatimaye, kufundisha mfano halisi wa Regression ya Mstari kunachukua mistari miwili tu ya msimbo. Tunafafanua kitu cha `LinearRegression`, na kukifaa kwa data yetu kwa kutumia njia ya `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Kitu cha `LinearRegression` baada ya `fit`-ting kina coefficients zote za regression, ambazo zinaweza kupatikana kwa kutumia mali ya `.coef_`. Katika kesi yetu, kuna coefficient moja tu, ambayo inapaswa kuwa karibu na `-0.017`. Inamaanisha kuwa bei zinaonekana kushuka kidogo kwa muda, lakini si sana, karibu senti 2 kwa siku. Tunaweza pia kufikia sehemu ya makutano ya regression na mhimili wa Y kwa kutumia `lin_reg.intercept_` - itakuwa karibu `21` katika kesi yetu, ikionyesha bei mwanzoni mwa mwaka.

Ili kuona jinsi mfano wetu ulivyo sahihi, tunaweza kutabiri bei kwenye seti ya data ya majaribio, na kisha kupima jinsi utabiri wetu ulivyo karibu na maadili yanayotarajiwa. Hii inaweza kufanywa kwa kutumia kipimo cha makosa ya mraba wa wastani (MSE), ambacho ni wastani wa tofauti zote zilizopigwa mraba kati ya thamani inayotarajiwa na iliyotabiriwa.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Kosa letu linaonekana kuwa katika alama 2, ambayo ni ~17%. Sio nzuri sana. Kiashiria kingine cha ubora wa modeli ni **mgawo wa uamuzi**, ambao unaweza kupatikana kwa njia hii:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```  
Ikiwa thamani ni 0, inamaanisha kuwa modeli haizingatii data ya pembejeo, na inafanya kazi kama *tabiri mbaya zaidi ya mstari*, ambayo ni wastani wa matokeo. Thamani ya 1 inamaanisha tunaweza kutabiri kikamilifu matokeo yote yanayotarajiwa. Katika hali yetu, mgawo uko karibu na 0.06, ambayo ni ya chini sana.

Tunaweza pia kuchora data ya majaribio pamoja na mstari wa regression ili kuona vizuri jinsi regression inavyofanya kazi katika hali yetu:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```  

<img alt="Linear regression" src="images/linear-results.png" width="50%" />

## Regression ya Polynomial  

Aina nyingine ya Regression ya Mstari ni Regression ya Polynomial. Wakati mwingine kuna uhusiano wa mstari kati ya vigezo - kadri malenge yanavyokuwa makubwa kwa ujazo, ndivyo bei inavyoongezeka - lakini wakati mwingine uhusiano huu hauwezi kuchorwa kama ndege au mstari wa moja kwa moja.

✅ Hapa kuna [mifano zaidi](https://online.stat.psu.edu/stat501/lesson/9/9.8) ya data inayoweza kutumia Regression ya Polynomial  

Angalia tena uhusiano kati ya Tarehe na Bei. Je, mchoro huu wa alama unaonekana kama unapaswa kuchambuliwa kwa mstari wa moja kwa moja? Je, bei haziwezi kubadilika? Katika hali hii, unaweza kujaribu regression ya polynomial.

✅ Polynomials ni maonyesho ya kihesabu ambayo yanaweza kuwa na moja au zaidi ya vigezo na vipeo  

Regression ya Polynomial huunda mstari wa mviringo ili kutosheleza data isiyo ya mstari vizuri zaidi. Katika hali yetu, ikiwa tutajumuisha kigezo cha `DayOfYear` kilichopigwa mraba katika data ya pembejeo, tunapaswa kuwa na uwezo wa kutosheleza data yetu na mviringo wa parabola, ambao utakuwa na kiwango cha chini katika sehemu fulani ndani ya mwaka.

Scikit-learn inajumuisha [API ya pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) inayosaidia kuunganisha hatua tofauti za usindikaji wa data pamoja. **Pipeline** ni mnyororo wa **estimators**. Katika hali yetu, tutaunda pipeline ambayo kwanza inaongeza vipengele vya polynomial kwenye modeli yetu, kisha inafundisha regression:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```  

Kutumia `PolynomialFeatures(2)` inamaanisha kuwa tutajumuisha polynomials zote za kiwango cha pili kutoka data ya pembejeo. Katika hali yetu, itamaanisha tu `DayOfYear`<sup>2</sup>, lakini kwa vigezo viwili vya pembejeo X na Y, hii itaongeza X<sup>2</sup>, XY na Y<sup>2</sup>. Tunaweza pia kutumia polynomials za kiwango cha juu ikiwa tunataka.

Pipelines zinaweza kutumika kwa njia sawa na kitu cha awali cha `LinearRegression`, yaani tunaweza `fit` pipeline, kisha kutumia `predict` kupata matokeo ya utabiri. Hapa kuna mchoro unaoonyesha data ya majaribio, na mstari wa takriban:

<img alt="Polynomial regression" src="images/poly-results.png" width="50%" />

Kwa kutumia Regression ya Polynomial, tunaweza kupata MSE ya chini kidogo na mgawo wa uamuzi wa juu zaidi, lakini sio kwa kiasi kikubwa. Tunahitaji kuzingatia vipengele vingine!

> Unaweza kuona kuwa bei za chini kabisa za malenge zinapatikana karibu na Halloween. Unaweza kuelezea hili vipi?

🎃 Hongera, umekamilisha modeli inayoweza kusaidia kutabiri bei ya malenge ya pai. Unaweza kurudia utaratibu huo kwa aina zote za malenge, lakini hilo litakuwa kazi ya kuchosha. Sasa hebu tujifunze jinsi ya kuzingatia aina ya malenge katika modeli yetu!

## Vipengele vya Kategoria  

Katika ulimwengu bora, tunataka kuwa na uwezo wa kutabiri bei za aina tofauti za malenge kwa kutumia modeli moja. Hata hivyo, safu ya `Variety` ni tofauti kidogo na safu kama `Month`, kwa sababu ina thamani zisizo za nambari. Safu kama hizi zinaitwa **kategoria**.

[![ML kwa wanaoanza - Utabiri wa Vipengele vya Kategoria na Regression ya Mstari](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML kwa wanaoanza - Utabiri wa Vipengele vya Kategoria na Regression ya Mstari")

> 🎥 Bofya picha hapo juu kwa muhtasari mfupi wa kutumia vipengele vya kategoria.

Hapa unaweza kuona jinsi bei ya wastani inavyotegemea aina:

<img alt="Average price by variety" src="images/price-by-variety.png" width="50%" />

Ili kuzingatia aina, tunahitaji kwanza kuibadilisha kuwa fomu ya nambari, au **kuencode**. Kuna njia kadhaa tunazoweza kutumia:

* **Numeric encoding** rahisi itajenga jedwali la aina tofauti, kisha kubadilisha jina la aina kwa index katika jedwali hilo. Hii sio wazo bora kwa regression ya mstari, kwa sababu regression ya mstari inachukua thamani halisi ya nambari ya index, na kuiongeza kwenye matokeo, ikizidisha kwa kipengele fulani. Katika hali yetu, uhusiano kati ya nambari ya index na bei ni wazi kuwa sio wa mstari, hata kama tunahakikisha kuwa indices zimepangwa kwa njia maalum.
* **One-hot encoding** itabadilisha safu ya `Variety` kuwa safu 4 tofauti, moja kwa kila aina. Kila safu itakuwa na `1` ikiwa safu husika ni ya aina fulani, na `0` vinginevyo. Hii inamaanisha kutakuwa na coefficients nne katika regression ya mstari, moja kwa kila aina ya malenge, inayohusika na "bei ya kuanzia" (au "bei ya ziada") kwa aina hiyo maalum.

Hapa kuna jinsi tunaweza kufanya one-hot encoding kwa aina:

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

Ili kufundisha regression ya mstari kwa kutumia aina iliyofanyiwa one-hot encoding kama pembejeo, tunahitaji tu kuanzisha data ya `X` na `y` kwa usahihi:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```  

Sehemu nyingine ya msimbo ni sawa na tuliyotumia hapo juu kufundisha Regression ya Mstari. Ukijaribu, utaona kuwa kosa la wastani la mraba ni karibu sawa, lakini tunapata mgawo wa uamuzi wa juu zaidi (~77%). Ili kupata utabiri sahihi zaidi, tunaweza kuzingatia vipengele zaidi vya kategoria, pamoja na vipengele vya nambari, kama `Month` au `DayOfYear`. Ili kupata safu moja kubwa ya vipengele, tunaweza kutumia `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```  

Hapa tunazingatia pia `City` na aina ya `Package`, ambayo inatupa MSE 2.84 (10%), na uamuzi 0.94!

## Kuweka Yote Pamoja  

Ili kufanya modeli bora zaidi, tunaweza kutumia data iliyojumuishwa (kategoria zilizofanyiwa one-hot encoding + nambari) kutoka mfano hapo juu pamoja na Regression ya Polynomial. Hapa kuna msimbo kamili kwa urahisi wako:

```python
# set up training data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# setup and train the pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data
pred = pipeline.predict(X_test)

# calculate MSE and determination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```  

Hii inapaswa kutupa mgawo bora wa uamuzi wa karibu 97%, na MSE=2.23 (~8% kosa la utabiri).

| Modeli | MSE | Uamuzi |  
|-------|-----|--------|  
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |  
| `DayOfYear` Polynomial | 2.73 (17.0%) | 0.08 |  
| `Variety` Linear | 5.24 (19.7%) | 0.77 |  
| Vipengele Vyote Linear | 2.84 (10.5%) | 0.94 |  
| Vipengele Vyote Polynomial | 2.23 (8.25%) | 0.97 |  

🏆 Hongera! Umeunda modeli nne za Regression katika somo moja, na kuboresha ubora wa modeli hadi 97%. Katika sehemu ya mwisho ya Regression, utajifunza kuhusu Regression ya Logistic ili kuamua kategoria.

---
## 🚀Changamoto  

Jaribu vigezo tofauti katika daftari hili ili kuona jinsi uhusiano unavyolingana na usahihi wa modeli.

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea  

Katika somo hili tulijifunza kuhusu Regression ya Mstari. Kuna aina nyingine muhimu za Regression. Soma kuhusu mbinu za Stepwise, Ridge, Lasso na Elasticnet. Kozi nzuri ya kusoma ili kujifunza zaidi ni [Kozi ya Stanford ya Statistical Learning](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Kazi  

[Jenga Modeli](assignment.md)  

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati asilia katika lugha yake ya awali inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.