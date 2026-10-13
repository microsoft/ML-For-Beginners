# CartPole Pateni

Önceki derste çözdüğümüz problem oyuncak bir problem gibi görünebilir, gerçek yaşam senaryoları için gerçekten uygulanabilir değilmiş gibi. Bu durum böyle değil, çünkü birçok gerçek dünya problemi de bu senaryoyu paylaşır - Satranç veya Go oynama dahil. Benzerler çünkü elimizde verilen kurallara sahip bir tahta ve **kesikli bir durum** bulunur.

## [Ders öncesi quiz](https://ff-quizzes.netlify.app/en/ml/)

## Giriş

Bu derste Q-Öğrenmenin aynı ilkelerini **sürekli durum** problemlerine uygulayacağız, yani durumu bir veya daha fazla gerçek sayıyla verilen bir problemdir. Aşağıdaki problemle ilgileneceğiz:

> **Problem**: Eğer Peter kurttan kaçmak istiyorsa, daha hızlı hareket edebilmesi gerekir. Peter'ın nasıl kaymayı, özellikle dengeyi nasıl koruyabileceğini Q-Öğrenme kullanarak göreceğiz.

![Büyük kaçış!](../../../../translated_images/tr/escape.18862db9930337e3.webp)

> Peter ve arkadaşları kurtlardan kaçmak için yaratıcı oluyor! Görsel: [Jen Looper](https://twitter.com/jenlooper)

Deneyimde, dengelemeyle ilgili basitleştirilmiş bir versiyon olan **CartPole** probleminden faydalanacağız. Cartpole dünyasında, yatay hareket edebilen bir kaydırıcı var ve hedef bu kaydırıcının üzerinde dikey bir direği dengelemektir.

<img alt="bir cartpole" src="../../../../translated_images/tr/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Önkoşullar

Bu derste, farklı **ortamları** simüle etmek için **OpenAI Gym** adlı bir kütüphane kullanacağız. Bu dersin kodunu yerel olarak (örneğin Visual Studio Code'dan) çalıştırabilirsiniz; bu durumda simülasyon yeni bir pencerede açılır. Çevrimiçi kod çalıştırırken, burada açıklandığı gibi [birkaç ayar yapmanız](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7) gerekebilir.

## OpenAI Gym

Önceki derste, oyunun kuralları ve durum, kendimizin tanımladığı `Board` sınıfı tarafından sağlanıyordu. Burada ise denge direğinin fiziksel simülasyonunu sağlayan özel bir **simülasyon ortamı** kullanacağız. Takviye öğrenme algoritmalarını eğitmek için en popüler simülasyon ortamlarından biri [Gym](https://gym.openai.com/) olarak adlandırılır ve [OpenAI](https://openai.com/) tarafından sürdürülür. Bu gym'i kullanarak cartpole simülasyonundan Atari oyunlarına kadar çeşitli **ortamlar** oluşturabiliriz.

> **Not**: OpenAI Gym tarafından sunulan diğer ortamları [buradan](https://gym.openai.com/envs/#classic_control) görebilirsiniz.

Öncelikle, gym'i yükleyelim ve gerekli kütüphaneleri içe aktaralım (kod bloğu 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Alıştırma - Cartpole ortamını başlatmak

Cartpole dengeleme problemiyle çalışmak için ilgili ortamı başlatmamız gerekir. Her ortam aşağıdakilerle ilişkilidir:

- Çevreden aldığımız bilgilerin yapısını tanımlayan **Gözlem uzayı**. Cartpole probleminde, direğin konumu, hızı ve bazı diğer değerleri alırız.

- Olası eylemleri tanımlayan **Eylem uzayı**. Bizim durumumuzda eylem uzayı kesiklidir ve iki eylemden oluşur - **sol** ve **sağ**. (kod bloğu 2)

1. Başlatmak için aşağıdaki kodu yazın:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Ortamın nasıl çalıştığını görmek için 100 adımlık kısa bir simülasyon yapalım. Her adımda yapılacak eylemlerden biri sağlanır - bu simülasyonda `action_space` içinden rastgele bir eylem seçiyoruz.

1. Aşağıdaki kodu çalıştırın ve neye yol açtığını görün.

    ✅ Bu kodu yerel Python kurulumunda çalıştırmanız tercih edilir! (kod bloğu 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Benzer bir görüntü görmelisiniz:

    ![dengelenmeyen cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Simülasyon esnasında, nasıl hareket edeceğimize karar vermek için gözlemler almalıyız. Aslında step fonksiyonu mevcut gözlemleri, ödül fonksiyonunu ve simülasyonun devam ettirilip ettirilmemesi gerektiğini gösteren done bayrağını döndürür: (kod bloğu 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Defter çıktısında aşağıdaki gibi bir şey göreceksiniz:

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
    - Arabacığın pozisyonu
    - Arabacığın hız vektörü
    - Direğin açısı
    - Direğin dönme hızı

1. Bu sayıların minimum ve maksimum değerlerini alın: (kod bloğu 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Ayrıca her simülasyon adımında ödülün daima 1 olduğunu görebilirsiniz. Bunun sebebi hedefimizin mümkün olduğunca uzun yaşamak, yani direği makul bir dik pozisyonda olabildiğince uzun tutmak olmasıdır.

    ✅ Aslında, CartPole simülasyonu, 100 ardışık denemde ortalama 195 ödüle ulaşılırsa çözüldü olarak kabul edilir.

## Durumun kesik hale getirilmesi

Q-Öğrenmede, her durumda ne yapılacağını tanımlayan Q-Tablosu oluşturmalıyız. Bunu yapabilmek için, durumun **kesikli** olması gerekir, yani sonlu sayıda kesikli değeri içermelidir. Bu nedenle gözlemlerimizi finİt bir dizi duruma eşleyerek bir şekilde **kesikli hale getirmemiz** gerekir.

Bunun birkaç yolu vardır:

- **Bölgelere ayırmak**. Belirli bir değerin aralığını biliyorsak, bu aralığı birkaç **bölgeye** ayırabilir ve ardından değeri ait olduğu bölge numarası ile değiştirebiliriz. Bu numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) metodu ile yapılabilir. Bu durumda durum boyutunu tam olarak biliriz çünkü seçtiğimiz dijitalizasyon bölgelerinin sayısına bağlıdır.
  
✅ Değerleri sonlu bir aralığa (örneğin -20 ile 20 arası) getirmek için lineer enterpolasyon kullanabiliriz, ardından sayıları yuvarlayarak tam sayılara dönüştürebiliriz. Bu, özellikle giriş değerlerinin tam aralıklarını bilmiyorsak durum boyutunu tam kontrol etmede biraz daha az esneklik sağlar. Örneğin, bizim durumumuzda 4 değerden 2'sinin üst/alt sınırları yoktur, bu da sonsuz sayıda durum olmasına yol açabilir.

Örneğimizde, ikinci yaklaşımı kullanacağız. Daha sonra fark edebileceğiniz gibi, tanımsız üst/alt sınırlarına rağmen, bu değerler nadiren belirli sonlu aralıkların dışına çıkar, bu nedenle aşırı değerli durumlar çok nadir olacaktır.

1. Modelimizden aldığı gözlemi alıp 4 tam sayıdan oluşan bir tuple döndürecek fonksiyon: (kod bloğu 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Bölgelere ayırma yöntemini kullanarak başka bir kesik hale getirme yöntemi de keşfedelim: (kod bloğu 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # her parametre için değer aralıkları
    nbins = [20,20,10,10] # her parametre için kutu sayısı
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Kısa bir simülasyon yapalım ve bu kesikli ortam değerlerine bakalım. Her iki fonksiyonu da deneyip arada fark olup olmadığını gözlemleyebilirsiniz.

    ✅ discretize_bins, 0 tabanlı bölge numarasını döndürür. Bu nedenle giriş değişkeni 0 civarında olanlar için aralığın ortasından bir sayı (10) verir. discretize'de ise çıktı değerlerinin aralığı ile ilgilenmedik, negatif olabilir, bu yüzden durum değerleri kaydırılmaz ve 0, 0'a karşılık gelir. (kod bloğu 8)

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

    ✅ Ortamın nasıl çalıştığını görmek isterseniz env.render ile başlayan satırı açabilirsiniz. Aksi halde arka planda çalıştırabilirsiniz, bu daha hızlıdır. Q-Öğrenme sürecimiz boyunca bu "görünmez" çalıştırmayı kullanacağız.

## Q-Tablo yapısı

Önceki dersimizde durum, 0 ile 8 arasında basit bir sayı çiftiydi, bu yüzden Q-Tablosunu 8x8x2 şekline sahip numpy tensörü olarak temsil etmek uygundu. Bölgelere ayırma kullanırsak durum vektörünün boyutu da bilinir, aynı yaklaşımı kullanabiliriz ve durumu 20x20x10x10x2 şekline sahip bir dizi olarak gösterebiliriz (burada 2, eylem uzayının boyutu ve diğer boyutlar gözlem uzayındaki parametreler için seçilen bölge sayısına karşılık gelir).

Ancak bazen gözlem uzayının kesin boyutları bilinmez. `discretize` fonksiyonunda durumumuz belirli limitler içinde kalmayabilir, çünkü bazı orijinal değerler sınırlı değildir. Bu nedenle, Q-Tablosunu sözlük olarak temsil ederek biraz farklı bir yaklaşım kullanacağız.

1. * (durum, eylem) * çiftini sözlük anahtarı olarak kullanın, değer ise Q-Tablo girişi olacaktır. (kod bloğu 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Burada ayrıca verilen durum için tüm olası eylemleri karşılayan Q-Tablo değerlerinin listesini döndüren `qvalues()` fonksiyonunu tanımlıyoruz. Eğer Q-Tabloda giriş yoksa varsayılan olarak 0 döndüreceğiz.

## Q-Öğrenme başlıyor

Şimdi Peter'a dengeyi öğretmeye hazırız!

1. Öncelikle bazı hiperparametreleri ayarlayalım: (kod bloğu 10)

    ```python
    # hiperparametreler
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Burada `alpha`, her adımda Q-Tablodaki mevcut değerlerin ne ölçüde ayarlanacağını belirleyen **öğrenme oranıdır**. Önceki derste 1 ile başlıyorduk ve eğitim sırasında `alpha` değerini düşürüyorduk. Bu örnekte basitlik için sabit tutacağız, ancak daha sonra `alpha` değerleriyle denemeler yapabilirsiniz.

    `gamma`, gelecekteki ödülü mevcut ödüle tercih etme derecesini gösteren **indirim faktörüdür**.

    `epsilon`, keşif/sömürü faktörüdür ve keşfi sömürüye tercih edip etmeyeceğimizi belirler. Algoritmamızda, `epsilon` yüzdesi kadar durumda Q-Tablosuna göre sonraki eylemi seçeceğiz, kalan durumda ise rastgele eylem gerçekleştireceğiz. Böylece daha önce keşfetmediğimiz arama alanlarını keşfetme şansımız olur.

    ✅ Dengeleme açısından, rastgele eylem seçmek (keşif), yanlış yöne rastgele bir hamle yapmak gibi olur ve direk bu "hatalardan" nasıl dengeyi yeniden sağlayacağını öğrenir.

### Algoritmayı geliştirin

Önceki dersten algoritmamıza iki iyileştirme yapabiliriz:

- Belirli bir sayıda simülasyondaki **ortalama kümülatif ödülü hesaplayın**. Her 5000 iterasyonda ilerlemeyi yazdıracağız ve o süre boyunca kümülatif ödüllerimizi ortalayacağız. Yani 195 puandan fazla alırsak problemi çözülmüş sayabiliriz, hatta gerekenin üzerinde kaliteyle.
  
- **Maksimum ortalama kümülatif sonucu**, `Qmax` hesaplayacağız ve bu sonuca karşılık gelen Q-Tablosunu saklayacağız. Eğitimi çalıştırdığınızda bazen ortalama kümülatif sonucun azaldığını gözlemlersiniz ve eğitim sırasında gözlemlenen en iyi modele karşılık gelen Q-Tablosu değerlerini korumak isteriz.

1. Grafik çizmek için her simülasyondaki kümülatif ödülleri `rewards` vektöründe toplayın. (kod bloğu 11)

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
        # == simülasyonu yap ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # sömürü - Q-Tablo olasılıklarına göre eylemi seç
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # keşif - rastgele eylemi seç
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Sonuçları periyodik olarak yazdır ve ortalama ödülü hesapla ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Bu sonuçlardan fark edebileceğiniz:

- **Hedefimize yakınız**. 100'den fazla ardışık simülasyonda 195 kümülatif ödül alma hedefine çok yakınız veya belki de tam olarak başardık! Daha düşük sayılar alsak bile henüz bilmiyoruz, çünkü ortalamayı 5000 koşu üzerinden alıyoruz ve resmi kriterde sadece 100 koşu isteniyor.
  
- **Ödül düşmeye başlıyor**. Bazen ödül azalmaya başlıyor, bu da Q-Tablosundaki daha önce öğrenilmiş değerleri durumu kötüleştirenlerle "bozabileceğimiz" anlamına geliyor.

Bu gözlem, eğitim ilerlemesini grafikle gösterdiğimizde daha net görünür.

## Eğitim İlerlemesini Grafikle Görüntüleme

Eğitim sırasında, her iterasyondaki kümülatif ödül değerlerini `rewards` vektörüne kaydettik. Burada iterasyon numarasına karşı grafiği:

```python
plt.plot(rewards)
```

![ham ilerleme](../../../../translated_images/tr/train_progress_raw.2adfdf2daea09c59.webp)

Bu grafikle bir şey söylemek mümkün değil, çünkü stohastik eğitim sürecinin doğası gereği eğitim süresi çok değişkendir. Bu grafiğin daha anlamlı olması için deneylerin üzerinde **koşan ortalama** hesaplayabiliriz, örneğin 100 deney üzerinden. Bu, `np.convolve` ile kolayca yapılabilir: (kod bloğu 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![eğitim ilerlemesi](../../../../translated_images/tr/train_progress_runav.c71694a8fa9ab359.webp)

## Hiperparametreleri değiştirmek

Öğrenmeyi daha kararlı hale getirmek için hiperparametrelerimizden bazılarını eğitim sırasında değiştirmek mantıklıdır. Özellikle:

- **Öğrenme oranı** `alpha` için başlangıçta 1'e yakın değerler ile başlayabilir ve sonra bu parametreyi azaltabiliriz. Zamanla Q-Tablosunda iyi olasılık değerleri elde edeceğiz, dolayısıyla onları sadece biraz ayarlamalı, tamamen yenileriyle değiştirmemeliyiz.

- **Epsilon'u artırmak**. `epsilon`'u yavaşça artırmak isteyebiliriz, böylece keşif daha az, sömürü daha fazla olur. Muhtemelen düşük `epsilon` ile başlayıp neredeyse 1'e kadar çıkarız.

> **Görev 1**: Hiperparametre değerleriyle oynayın ve daha yüksek kümülatif ödül elde edip edemediğinizi görün. 195'in üzerinde alıyor musunuz?


> **Görev 2**: Problemi resmi olarak çözmek için, 100 ardışık denemede ortalama 195 ödül almanız gerekiyor. Bunu eğitim sırasında ölçün ve problemin resmi olarak çözüldüğünden emin olun!

## Sonucu eylem halinde görmek

Eğitilmiş modelin nasıl davrandığını gerçekten görmek ilginç olurdu. Simülasyonu çalıştıralım ve eğitim sırasında kullandığımız aynı eylem seçme stratejisini izleyelim, Q-Tablosundaki olasılık dağılımına göre örneklem yapalım: (kod bloğu 13)

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

Şunun gibi bir şey görmelisiniz:

![bir dengeleyen cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Meydan Okuma

> **Görev 3**: Burada en son kopya Q-Tablosunu kullanıyorduk, bu en iyi olanı olmayabilir. En iyi performans gösteren Q-Tablosunu `Qbest` değişkenine kaydettiğimizi unutmayın! Aynı örneği en iyi performans gösteren Q-Tablosu ile deneyin, `Qbest`i `Q` üzerine kopyalayarak farkı görüp görmediğinizi kontrol edin.

> **Görev 4**: Burada her adımda en iyi eylemi seçmiyorduk, ancak karşılık gelen olasılık dağılımıyla örneklem yapıyorduk. Her zaman en yüksek Q-Tablo değerine sahip olan en iyi eylemi seçmek daha mantıklı olur mu? Bu, `np.argmax` fonksiyonunu kullanarak en yüksek Q-Tablo değerine karşılık gelen eylem numarasını bulmakla yapılabilir. Bu stratejiyi uygulayın ve dengelemenin iyileşip iyileşmediğine bakın.

## [Ders sonrası quiz](https://ff-quizzes.netlify.app/en/ml/)

## Ödev
[Bir Dağ Arabası Eğitin](assignment.md)

## Sonuç

Artık, oyunun istenen durumunu tanımlayan bir ödül fonksiyonu sağlayarak ve arama alanını akıllıca keşfetmelerine fırsat vererek, ajanları iyi sonuçlar elde edecek şekilde nasıl eğiteceğimizi öğrendik. Q-Öğrenme algoritmasını hem ayrık hem de sürekli ortamlarda, ancak ayrık eylemlerle başarıyla uyguladık.

Eylem durumunun da sürekli olduğu ve gözlem alanının Atari oyun ekranından gelen görüntü gibi çok daha karmaşık olduğu durumları incelemek de önemlidir. Bu tür problemlerde genellikle iyi sonuçlar elde etmek için sinir ağları gibi daha güçlü makine öğrenme tekniklerini kullanmamız gerekir. Bu daha ileri konular, yaklaşmakta olan daha ileri seviye Yapay Zeka dersimizin konusudur.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Feragatname**:
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayınız. Orijinal belge, kendi dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımı sonucu ortaya çıkabilecek yanlış anlamalardan veya yanlış yorumlamalardan sorumlu değiliz.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->