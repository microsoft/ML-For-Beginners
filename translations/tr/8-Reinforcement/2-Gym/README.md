<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-06T08:04:44+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "tr"
}
-->
## Ön Koşullar

Bu derste, farklı **ortamları** simüle etmek için **OpenAI Gym** adlı bir kütüphane kullanacağız. Bu dersin kodunu yerel olarak (örneğin, Visual Studio Code'dan) çalıştırabilirsiniz; bu durumda simülasyon yeni bir pencerede açılacaktır. Kodu çevrimiçi çalıştırırken, [burada](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7) açıklandığı gibi kodda bazı değişiklikler yapmanız gerekebilir.

## OpenAI Gym

Önceki derste, oyunun kuralları ve durum, kendimiz tanımladığımız `Board` sınıfı tarafından belirlenmişti. Burada, dengeleyen direğin fiziğini simüle edecek özel bir **simülasyon ortamı** kullanacağız. Takviye öğrenme algoritmalarını eğitmek için en popüler simülasyon ortamlarından biri, [Gym](https://gym.openai.com/) olarak adlandırılır ve [OpenAI](https://openai.com/) tarafından geliştirilmiştir. Bu gym'i kullanarak, bir cartpole simülasyonundan Atari oyunlarına kadar farklı **ortamlar** oluşturabiliriz.

> **Not**: OpenAI Gym tarafından sunulan diğer ortamları [buradan](https://gym.openai.com/envs/#classic_control) görebilirsiniz.

Öncelikle gym'i yükleyelim ve gerekli kütüphaneleri içe aktaralım (kod bloğu 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Alıştırma - bir cartpole ortamı başlatma

Cartpole dengeleme problemiyle çalışmak için ilgili ortamı başlatmamız gerekiyor. Her ortam şu özelliklere sahiptir:

- **Gözlem alanı**: Ortamdan aldığımız bilgilerin yapısını tanımlar. Cartpole probleminde, direğin pozisyonu, hızı ve diğer bazı değerleri alırız.

- **Eylem alanı**: Olası eylemleri tanımlar. Bizim durumumuzda eylem alanı ayrık olup iki eylemden oluşur - **sol** ve **sağ**. (kod bloğu 2)

1. Başlatmak için şu kodu yazın:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Ortamın nasıl çalıştığını görmek için 100 adımlık kısa bir simülasyon çalıştıralım. Her adımda, yapılacak bir eylemi sağlıyoruz - bu simülasyonda `action_space`'den rastgele bir eylem seçiyoruz.

1. Aşağıdaki kodu çalıştırın ve ne sonuç verdiğini görün.

    ✅ Bu kodu yerel bir Python kurulumunda çalıştırmanız önerilir! (kod bloğu 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Şuna benzer bir şey görmelisiniz:

    ![dengelemeyen cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Simülasyon sırasında, nasıl hareket edeceğimize karar vermek için gözlemler almamız gerekiyor. Aslında, step fonksiyonu mevcut gözlemleri, bir ödül fonksiyonunu ve simülasyona devam etmenin mantıklı olup olmadığını gösteren bir done bayrağını döndürür: (kod bloğu 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Notebook çıktısında buna benzer bir şey göreceksiniz:

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    Simülasyonun her adımında döndürülen gözlem vektörü şu değerleri içerir:
    - Kartın pozisyonu
    - Kartın hızı
    - Direğin açısı
    - Direğin dönüş hızı

1. Bu sayıların minimum ve maksimum değerlerini alın: (kod bloğu 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Ayrıca, her simülasyon adımında ödül değerinin her zaman 1 olduğunu fark edebilirsiniz. Bunun nedeni, amacımızın mümkün olduğunca uzun süre hayatta kalmak, yani direği makul bir dikey pozisyonda en uzun süre tutmak olmasıdır.

    ✅ Aslında, CartPole simülasyonu, 100 ardışık deneme boyunca ortalama 195 ödül almayı başardığımızda çözülmüş kabul edilir.

## Durum Ayrıklaştırma

Q-Öğrenme'de, her durumda ne yapılacağını tanımlayan bir Q-Tablosu oluşturmamız gerekiyor. Bunu yapabilmek için, durumun **ayrık** olması, daha doğrusu, sonlu sayıda ayrık değer içermesi gerekir. Bu nedenle, gözlemlerimizi bir şekilde **ayrıklaştırmamız**, onları sonlu bir durum kümesine eşlememiz gerekiyor.

Bunu yapmanın birkaç yolu vardır:

- **Aralıklara bölmek**. Belirli bir değerin aralığını biliyorsak, bu aralığı bir dizi **aralığa** bölebilir ve ardından değeri ait olduğu aralık numarasıyla değiştirebiliriz. Bu, numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) yöntemi kullanılarak yapılabilir. Bu durumda, durum boyutunu kesin olarak bileceğiz, çünkü dijitalleştirme için seçtiğimiz aralık sayısına bağlı olacaktır.
  
✅ Değerleri belirli bir sonlu aralığa (örneğin, -20 ile 20 arasında) getirmek için doğrusal interpolasyon kullanabilir ve ardından sayıları yuvarlayarak tam sayılara dönüştürebiliriz. Bu, durum boyutunun büyüklüğü üzerinde biraz daha az kontrol sağlar, özellikle de giriş değerlerinin kesin aralıklarını bilmiyorsak. Örneğin, bizim durumumuzda 4 değerden 2'sinin üst/alt sınırları yoktur, bu da sonsuz sayıda duruma yol açabilir.

Bizim örneğimizde ikinci yaklaşımı kullanacağız. Daha sonra fark edeceğiniz gibi, tanımsız üst/alt sınırlara rağmen, bu değerler nadiren belirli sonlu aralıkların dışına çıkar, bu nedenle aşırı değerlerle durumlar çok nadir olacaktır.

1. İşte modelimizden gözlemi alacak ve 4 tam sayı değerinden oluşan bir tuple üretecek fonksiyon: (kod bloğu 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Aralıklar kullanarak başka bir ayrıklaştırma yöntemini de keşfedelim: (kod bloğu 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
    nbins = [20,20,10,10] # number of bins for each parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Şimdi kısa bir simülasyon çalıştıralım ve bu ayrık ortam değerlerini gözlemleyelim. `discretize` ve `discretize_bins` yöntemlerini denemekten çekinmeyin ve aralarında bir fark olup olmadığını görün.

    ✅ discretize_bins, 0 tabanlı olan aralık numarasını döndürür. Bu nedenle, giriş değişkeninin 0 civarındaki değerleri için aralığın ortasındaki numarayı (10) döndürür. Discretize'de, çıktı değerlerinin aralığıyla ilgilenmedik, onların negatif olmasına izin verdik, bu nedenle durum değerleri kaydırılmadı ve 0, 0'a karşılık gelir. (kod bloğu 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Ortamın nasıl çalıştığını görmek istiyorsanız, env.render ile başlayan satırı yorumdan çıkarabilirsiniz. Aksi takdirde, bunu arka planda çalıştırabilirsiniz, bu daha hızlıdır. Q-Öğrenme sürecimiz sırasında bu "görünmez" yürütmeyi kullanacağız.

## Q-Tablosu Yapısı

Önceki dersimizde, durum 0'dan 8'e kadar olan basit bir sayı çiftiydi ve bu nedenle Q-Tablosunu 8x8x2 şekline sahip bir numpy tensörü ile temsil etmek uygundu. Aralık ayrıklaştırmasını kullanırsak, durum vektörümüzün boyutu da bilinir, bu nedenle aynı yaklaşımı kullanabilir ve durumu 20x20x10x10x2 şeklinde bir dizi ile temsil edebiliriz (burada 2, eylem alanının boyutudur ve ilk boyutlar gözlem alanındaki her parametre için seçtiğimiz aralık sayısına karşılık gelir).

Ancak, bazen gözlem alanının kesin boyutları bilinmez. `discretize` fonksiyonu durumunda, durumumuzun belirli sınırlar içinde kalacağından asla emin olamayabiliriz, çünkü bazı orijinal değerler sınırlı değildir. Bu nedenle, biraz farklı bir yaklaşım kullanacağız ve Q-Tablosunu bir sözlükle temsil edeceğiz.

1. *(state,action)* çiftini sözlük anahtarı olarak kullanın ve değer Q-Tablosu giriş değerine karşılık gelir. (kod bloğu 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Burada ayrıca, bir duruma karşılık gelen tüm olası eylemler için Q-Tablosu değerlerinin bir listesini döndüren `qvalues()` fonksiyonunu tanımlıyoruz. Giriş Q-Tablosunda mevcut değilse, varsayılan olarak 0 döndüreceğiz.

## Q-Öğrenmeye Başlayalım

Şimdi Peter'a dengeyi öğretmeye hazırız!

1. İlk olarak, bazı hiperparametreler belirleyelim: (kod bloğu 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Burada, `alpha`, Q-Tablosunun mevcut değerlerini her adımda ne ölçüde ayarlamamız gerektiğini tanımlayan **öğrenme oranıdır**. Önceki derste 1 ile başladık ve ardından `alpha` değerlerini eğitim sırasında daha düşük değerlere düşürdük. Bu örnekte basitlik adına sabit tutacağız ve daha sonra `alpha` değerlerini ayarlamayı deneyebilirsiniz.

    `gamma`, gelecekteki ödülü mevcut ödüle göre ne ölçüde önceliklendirmemiz gerektiğini gösteren **indirim faktörüdür**.

    `epsilon`, keşif/istismar faktörüdür ve keşfi mi yoksa istismarı mı tercih etmemiz gerektiğini belirler. Algoritmamızda, `epsilon` yüzdesinde bir sonraki eylemi Q-Tablosu değerlerine göre seçeceğiz ve kalan durumlarda rastgele bir eylem gerçekleştireceğiz. Bu, daha önce hiç görmediğimiz arama alanı bölgelerini keşfetmemizi sağlayacaktır.

    ✅ Denge açısından - rastgele bir eylem seçmek (keşif), yanlış yönde rastgele bir darbe gibi davranır ve direk bu "hatalardan" dengeyi nasıl kurtaracağını öğrenmek zorunda kalır.

### Algoritmayı Geliştirme

Önceki dersten algoritmamıza iki iyileştirme yapabiliriz:

- **Ortalama kümülatif ödülü hesaplama**, bir dizi simülasyon boyunca. İlerlemeyi her 5000 iterasyonda yazdıracağız ve kümülatif ödülümüzü bu süre boyunca ortalama alacağız. Bu, 195 puandan fazla alırsak - problemi çözmüş kabul edebiliriz, hatta gereken kaliteden daha yüksek bir şekilde.

- **Maksimum ortalama kümülatif sonucu hesaplama**, `Qmax`, ve bu sonuca karşılık gelen Q-Tablosunu saklayacağız. Eğitimi çalıştırdığınızda, bazen ortalama kümülatif sonucun düşmeye başladığını fark edeceksiniz ve bu durumda, Q-Tablosunda durumu daha kötü hale getiren değerlerle zaten öğrenilmiş değerleri "bozabiliriz".

1. Her simülasyondaki kümülatif ödülleri `rewards` vektöründe toplayın ve daha sonra grafik çizmek için kullanın. (kod bloğu 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Bu sonuçlardan fark edebileceğiniz şeyler:

- **Hedefimize çok yakınız**. 100+ ardışık simülasyon çalıştırması boyunca 195 kümülatif ödül alma hedefimize çok yakınız veya aslında bunu başarmış olabiliriz! Daha küçük sayılar alsak bile, hala bilmiyoruz, çünkü 5000 çalıştırma boyunca ortalama alıyoruz ve resmi kriterde yalnızca 100 çalıştırma gerekiyor.

- **Ödül düşmeye başlıyor**. Bazen ödül düşmeye başlar, bu da Q-Tablosunda zaten öğrenilmiş değerleri daha kötü hale getiren değerlerle "bozabileceğimiz" anlamına gelir.

Bu gözlem, eğitim ilerlemesini grafikle gösterdiğimizde daha net bir şekilde görülebilir.

## Eğitim İlerlemesini Grafikle Gösterme

Eğitim sırasında, her iterasyondaki kümülatif ödül değerini `rewards` vektörüne topladık. İşte bunu iterasyon numarasına karşı grafikle gösterdiğimizde nasıl göründüğü:

```python
plt.plot(rewards)
```

![ham ilerleme](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Bu grafikten bir şey söylemek mümkün değil, çünkü stokastik eğitim sürecinin doğası gereği eğitim oturumlarının uzunluğu büyük ölçüde değişiyor. Bu grafiği daha anlamlı hale getirmek için, bir dizi deney boyunca **hareketli ortalama** hesaplayabiliriz, örneğin 100. Bu, `np.convolve` kullanılarak kolayca yapılabilir: (kod bloğu 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![eğitim ilerlemesi](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Hiperparametreleri Değiştirme

Öğrenmeyi daha istikrarlı hale getirmek için, eğitim sırasında bazı hiperparametrelerimizi ayarlamak mantıklı olabilir. Özellikle:

- **Öğrenme oranı** için, `alpha`, 1'e yakın değerlerle başlayabilir ve ardından bu parametreyi düşürmeye devam edebiliriz. Zamanla, Q-Tablosunda iyi olasılık değerleri elde edeceğiz ve bu nedenle onları hafifçe ayarlamalı, tamamen yeni değerlerle üzerine yazmamalıyız.

- **Epsilon'u artırma**. `epsilon`'u yavaşça artırmak isteyebiliriz, böylece daha az keşif yapıp daha fazla istismar yapabiliriz. Muhtemelen daha düşük bir `epsilon` değeriyle başlamak ve neredeyse 1'e kadar çıkarmak mantıklı olacaktır.
> **Görev 1**: Hiperparametre değerleriyle oynayın ve daha yüksek toplam ödül elde edip edemeyeceğinizi görün. 195'in üzerine çıkabiliyor musunuz?
> **Görev 2**: Problemi resmi olarak çözmek için, 100 ardışık çalışmada 195 ortalama ödül almanız gerekiyor. Bunu eğitim sırasında ölçün ve problemi resmi olarak çözdüğünüzden emin olun!

## Sonucu eylemde görmek

Eğitilmiş modelin nasıl davrandığını görmek ilginç olurdu. Simülasyonu çalıştıralım ve eğitim sırasında olduğu gibi aynı eylem seçme stratejisini izleyelim, Q-Tablosundaki olasılık dağılımına göre örnekleme yapalım: (kod bloğu 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

Şuna benzer bir şey görmelisiniz:

![dengeleyen bir cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Meydan Okuma

> **Görev 3**: Burada, Q-Tablosunun son kopyasını kullanıyorduk, ancak bu en iyi olanı olmayabilir. Unutmayın ki en iyi performans gösteren Q-Tablosunu `Qbest` değişkenine kaydettik! `Qbest`'i `Q` üzerine kopyalayarak en iyi performans gösteren Q-Tablosuyla aynı örneği deneyin ve farkı fark edip etmediğinizi görün.

> **Görev 4**: Burada her adımda en iyi eylemi seçmiyorduk, bunun yerine ilgili olasılık dağılımıyla örnekleme yapıyorduk. Her zaman en iyi eylemi, en yüksek Q-Tablosu değerine sahip olanı seçmek daha mantıklı olur mu? Bu, en yüksek Q-Tablosu değerine karşılık gelen eylem numarasını bulmak için `np.argmax` fonksiyonu kullanılarak yapılabilir. Bu stratejiyi uygulayın ve dengelemeyi iyileştirip iyileştirmediğini görün.

## [Ders sonrası sınav](https://ff-quizzes.netlify.app/en/ml/)

## Ödev
[Bir Mountain Car Eğitin](assignment.md)

## Sonuç

Artık ajanları, oyunun istenen durumunu tanımlayan bir ödül fonksiyonu sağlayarak ve arama alanını akıllıca keşfetme fırsatı vererek iyi sonuçlar elde etmeleri için nasıl eğiteceğimizi öğrendik. Q-Öğrenme algoritmasını, ayrık ve sürekli ortamlar durumunda, ancak ayrık eylemlerle başarıyla uyguladık.

Eylem durumunun da sürekli olduğu ve gözlem alanının çok daha karmaşık olduğu, örneğin Atari oyun ekranından bir görüntü gibi durumları incelemek de önemlidir. Bu tür problemler için genellikle iyi sonuçlar elde etmek amacıyla daha güçlü makine öğrenimi tekniklerini, örneğin sinir ağlarını kullanmamız gerekir. Bu daha ileri düzey konular, yaklaşan daha gelişmiş AI kursumuzun konusudur.

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluğu sağlamak için çaba göstersek de, otomatik çeviriler hata veya yanlışlıklar içerebilir. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalardan sorumlu değiliz.