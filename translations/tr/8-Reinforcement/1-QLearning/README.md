<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-06T08:03:43+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "tr"
}
-->
# Pekiştirmeli Öğrenme ve Q-Öğrenme'ye Giriş

![Makine öğreniminde pekiştirme özetini içeren bir sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote: [Tomomi Imura](https://www.twitter.com/girlie_mac)

Pekiştirmeli öğrenme üç önemli kavram içerir: ajan, bazı durumlar ve her durum için bir dizi eylem. Belirli bir durumda bir eylem gerçekleştirerek, ajana bir ödül verilir. Yine Super Mario bilgisayar oyununu hayal edin. Siz Mario'sunuz, bir oyun seviyesindesiniz ve uçurumun kenarında duruyorsunuz. Üstünüzde bir altın var. Mario olarak, bir oyun seviyesinde, belirli bir pozisyonda olmanız... bu sizin durumunuzdur. Sağa bir adım atmak (bir eylem) sizi uçurumdan aşağıya götürür ve bu size düşük bir sayısal puan verir. Ancak zıplama tuşuna basmak size bir puan kazandırır ve hayatta kalırsınız. Bu olumlu bir sonuçtur ve size olumlu bir sayısal puan kazandırmalıdır.

Pekiştirmeli öğrenme ve bir simülatör (oyun) kullanarak, hayatta kalmayı ve mümkün olduğunca çok puan toplamayı hedefleyerek oyunu nasıl oynayacağınızı öğrenebilirsiniz.

[![Pekiştirmeli Öğrenmeye Giriş](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Yukarıdaki görsele tıklayarak Dmitry'nin Pekiştirmeli Öğrenme hakkındaki konuşmasını dinleyin

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

## Ön Koşullar ve Kurulum

Bu derste, Python'da bazı kodlarla denemeler yapacağız. Bu dersteki Jupyter Notebook kodunu bilgisayarınızda veya bulutta çalıştırabilmelisiniz.

[Ders not defterini](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) açabilir ve bu dersi adım adım takip edebilirsiniz.

> **Not:** Bu kodu buluttan açıyorsanız, not defteri kodunda kullanılan [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) dosyasını da almanız gerekir. Bu dosyayı not defteriyle aynı dizine ekleyin.

## Giriş

Bu derste, Rus besteci [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) tarafından yazılmış bir müzikal masaldan esinlenilen **[Peter ve Kurt](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)** dünyasını keşfedeceğiz. Peter'ın çevresini keşfetmesine, lezzetli elmalar toplamasına ve kurtla karşılaşmaktan kaçınmasına olanak tanımak için **Pekiştirmeli Öğrenme** kullanacağız.

**Pekiştirmeli Öğrenme** (RL), bir **ajan**ın bir **çevrede** optimal davranışını öğrenmesine olanak tanıyan bir öğrenme tekniğidir. Bu çevredeki bir ajanın bir **hedefi** olmalıdır ve bu hedef bir **ödül fonksiyonu** ile tanımlanır.

## Çevre

Basitlik açısından, Peter'ın dünyasını `genişlik` x `yükseklik` boyutlarında bir kare tahta olarak düşünelim:

![Peter'ın Çevresi](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Bu tahtadaki her hücre şu şekilde olabilir:

* **zemin**, Peter ve diğer canlıların üzerinde yürüyebileceği yer.
* **su**, üzerinde yürüyemeyeceğiniz yer.
* bir **ağaç** veya **çimen**, dinlenebileceğiniz bir yer.
* bir **elma**, Peter'ın kendini beslemek için bulmaktan memnun olacağı bir şey.
* bir **kurt**, tehlikeli ve kaçınılması gereken bir şey.

Bu çevreyle çalışmak için kod içeren ayrı bir Python modülü olan [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) bulunmaktadır. Bu kod, kavramlarımızı anlamak için önemli olmadığından, modülü içe aktaracağız ve örnek tahtayı oluşturmak için kullanacağız (kod bloğu 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Bu kod, yukarıdaki resme benzer bir çevre görüntüsü yazdırmalıdır.

## Eylemler ve Politika

Örneğimizde, Peter'ın hedefi bir elma bulmak, kurt ve diğer engellerden kaçınmak olacaktır. Bunu yapmak için, elmayı bulana kadar etrafta dolaşabilir.

Bu nedenle, herhangi bir pozisyonda yukarı, aşağı, sola ve sağa hareket etmek gibi eylemlerden birini seçebilir.

Bu eylemleri bir sözlük olarak tanımlayacağız ve bunları ilgili koordinat değişiklikleriyle eşleyeceğiz. Örneğin, sağa hareket etmek (`R`) `(1,0)` çiftine karşılık gelir. (kod bloğu 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Bu senaryonun stratejisi ve hedefi şu şekilde özetlenebilir:

- **Strateji**, ajanımızın (Peter) stratejisi, **politika** olarak adlandırılan bir fonksiyonla tanımlanır. Politika, herhangi bir durumda eylemi döndüren bir fonksiyondur. Bizim durumumuzda, problemin durumu, oyuncunun mevcut pozisyonu dahil olmak üzere tahtayla temsil edilir.

- **Hedef**, pekiştirmeli öğrenmenin nihai hedefi, problemi verimli bir şekilde çözmemizi sağlayacak iyi bir politika öğrenmektir. Ancak, temel olarak, **rastgele yürüyüş** adı verilen en basit politikayı ele alalım.

## Rastgele Yürüyüş

Öncelikle rastgele yürüyüş stratejisini uygulayarak problemimizi çözelim. Rastgele yürüyüşte, elmaya ulaşana kadar izin verilen eylemlerden birini rastgele seçeriz (kod bloğu 3).

1. Aşağıdaki kodla rastgele yürüyüşü uygulayın:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    `walk` çağrısı, bir çalıştırmadan diğerine değişebilen ilgili yolun uzunluğunu döndürmelidir.

1. Yürüyüş deneyini birkaç kez (örneğin, 100 kez) çalıştırın ve elde edilen istatistikleri yazdırın (kod bloğu 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    Ortalama yol uzunluğunun yaklaşık 30-40 adım olduğunu unutmayın, bu oldukça fazladır, çünkü en yakın elmaya olan ortalama mesafe yaklaşık 5-6 adımdır.

    Ayrıca Peter'ın rastgele yürüyüş sırasında hareketinin nasıl göründüğünü görebilirsiniz:

    ![Peter'ın Rastgele Yürüyüşü](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Ödül Fonksiyonu

Politikamızı daha akıllı hale getirmek için hangi hareketlerin diğerlerinden "daha iyi" olduğunu anlamamız gerekir. Bunu yapmak için hedefimizi tanımlamamız gerekir.

Hedef, her durum için bir puan değeri döndüren bir **ödül fonksiyonu** ile tanımlanabilir. Sayı ne kadar yüksekse, ödül fonksiyonu o kadar iyidir. (kod bloğu 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

Ödül fonksiyonlarıyla ilgili ilginç bir şey, çoğu durumda *oyunun sonunda yalnızca önemli bir ödül verilmesidir*. Bu, algoritmamızın olumlu bir ödüle yol açan "iyi" adımları bir şekilde hatırlaması ve önemlerini artırması gerektiği anlamına gelir. Benzer şekilde, kötü sonuçlara yol açan tüm hareketler caydırılmalıdır.

## Q-Öğrenme

Burada tartışacağımız algoritma **Q-Öğrenme** olarak adlandırılır. Bu algoritmada politika, bir **Q-Tablosu** adı verilen bir fonksiyon (veya veri yapısı) ile tanımlanır. Bu tablo, belirli bir durumdaki her bir eylemin "iyiliğini" kaydeder.

Q-Tablosu olarak adlandırılır çünkü genellikle bir tablo veya çok boyutlu bir dizi olarak temsil etmek uygundur. Tahtamızın boyutları `genişlik` x `yükseklik` olduğundan, Q-Tablosunu `genişlik` x `yükseklik` x `len(actions)` şeklinde bir numpy dizisi kullanarak temsil edebiliriz: (kod bloğu 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Q-Tablosunun tüm değerlerini eşit bir değerle, bizim durumumuzda - 0.25 ile başlatıyoruz. Bu, her durumda tüm hareketlerin eşit derecede iyi olduğu "rastgele yürüyüş" politikasına karşılık gelir. Q-Tablosunu tahtada görselleştirmek için `plot` fonksiyonuna geçirebiliriz: `m.plot(Q)`.

![Peter'ın Çevresi](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Her hücrenin ortasında, hareket yönünü gösteren bir "ok" vardır. Tüm yönler eşit olduğundan, bir nokta gösterilir.

Şimdi simülasyonu çalıştırmamız, çevremizi keşfetmemiz ve Q-Tablosu değerlerinin daha iyi bir dağılımını öğrenmemiz gerekiyor, bu da elmaya giden yolu çok daha hızlı bulmamızı sağlayacaktır.

## Q-Öğrenmenin Özeti: Bellman Denklemi

Hareket etmeye başladığımızda, her eylemin karşılık gelen bir ödülü olacaktır, yani teorik olarak en yüksek anlık ödüle göre bir sonraki eylemi seçebiliriz. Ancak, çoğu durumda, hareket hedefimize ulaşmamızı sağlamayacak ve bu nedenle hangi yönün daha iyi olduğunu hemen karar veremeyiz.

> Önemli olan anlık sonuç değil, simülasyonun sonunda elde edeceğimiz nihai sonuçtur.

Bu gecikmiş ödülü hesaba katmak için, problemimizi özyinelemeli olarak düşünmemize olanak tanıyan **[dinamik programlama](https://en.wikipedia.org/wiki/Dynamic_programming)** ilkelerini kullanmamız gerekir.

Şimdi *s* durumundayız ve bir sonraki *s'* durumuna geçmek istiyoruz. Bunu yaparak, ödül fonksiyonu tarafından tanımlanan *r(s,a)* anlık ödülünü ve gelecekteki bir ödülü alacağız. Eğer Q-Tablomuzun her eylemin "çekiciliğini" doğru bir şekilde yansıttığını varsayarsak, *s'* durumunda *Q(s',a')*'nin maksimum değerine karşılık gelen bir eylem *a* seçeceğiz. Böylece, *s* durumunda alabileceğimiz en iyi olası gelecekteki ödül şu şekilde tanımlanacaktır: `max`

## Politikayı Kontrol Etme

Q-Tablosu, her durumdaki her bir eylemin "çekiciliğini" listelediği için, dünyamızda verimli bir gezinmeyi tanımlamak için kullanılması oldukça kolaydır. En basit durumda, en yüksek Q-Tablo değerine karşılık gelen eylemi seçebiliriz: (kod bloğu 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Yukarıdaki kodu birkaç kez denerseniz, bazen "takıldığını" fark edebilirsiniz ve bunu durdurmak için not defterindeki STOP düğmesine basmanız gerekir. Bu, iki durumun optimal Q-Değeri açısından birbirine "işaret ettiği" durumlarda meydana gelir; bu durumda ajan, bu durumlar arasında sonsuz bir şekilde hareket eder.

## 🚀Meydan Okuma

> **Görev 1:** `walk` fonksiyonunu, yolun maksimum uzunluğunu belirli bir adım sayısıyla (örneğin, 100) sınırlayacak şekilde değiştirin ve yukarıdaki kodun bu değeri zaman zaman döndürdüğünü gözlemleyin.

> **Görev 2:** `walk` fonksiyonunu, daha önce bulunduğu yerlere geri dönmemesini sağlayacak şekilde değiştirin. Bu, `walk`'un döngüye girmesini engelleyecektir, ancak ajan yine de kaçamayacağı bir konumda "sıkışıp" kalabilir.

## Gezinme

Daha iyi bir gezinme politikası, eğitim sırasında kullandığımız ve sömürü ile keşfi birleştiren politika olacaktır. Bu politikada, her bir eylemi belirli bir olasılıkla, Q-Tablosundaki değerlere orantılı olarak seçiyoruz. Bu strateji, ajanın daha önce keşfettiği bir konuma geri dönmesine neden olabilir, ancak aşağıdaki koddan görebileceğiniz gibi, istenen konuma çok kısa bir ortalama yol ile sonuçlanır (unutmayın ki `print_statistics` simülasyonu 100 kez çalıştırır): (kod bloğu 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Bu kodu çalıştırdıktan sonra, önceki ortalama yol uzunluğundan çok daha küçük bir değer elde etmelisiniz, genellikle 3-6 aralığında.

## Öğrenme Sürecini İnceleme

Bahsettiğimiz gibi, öğrenme süreci, problem alanının yapısı hakkında kazanılan bilginin keşfi ve kullanımı arasında bir dengedir. Öğrenme sonuçlarının (ajanın hedefe kısa bir yol bulmasına yardımcı olma yeteneği) iyileştiğini gördük, ancak öğrenme süreci sırasında ortalama yol uzunluğunun nasıl davrandığını gözlemlemek de ilginçtir:

## Öğrenilenler şu şekilde özetlenebilir:

- **Ortalama yol uzunluğu artar**. Burada gördüğümüz şey, başlangıçta ortalama yol uzunluğunun arttığıdır. Bu muhtemelen, çevre hakkında hiçbir şey bilmediğimizde kötü durumlarda, su veya kurt gibi, sıkışma olasılığımızın yüksek olmasından kaynaklanmaktadır. Daha fazla bilgi edindikçe ve bu bilgiyi kullanmaya başladıkça, çevreyi daha uzun süre keşfedebiliriz, ancak elmaların nerede olduğunu hala çok iyi bilmiyoruz.

- **Yol uzunluğu, daha fazla öğrendikçe azalır**. Yeterince öğrendikten sonra, ajanın hedefe ulaşması daha kolay hale gelir ve yol uzunluğu azalmaya başlar. Ancak, hala keşfe açık olduğumuz için, genellikle en iyi yoldan saparız ve yeni seçenekleri keşfederek yolu optimalden daha uzun hale getiririz.

- **Uzunluk aniden artar**. Grafikte ayrıca uzunluğun bir noktada aniden arttığını gözlemliyoruz. Bu, sürecin stokastik doğasını ve Q-Tablo katsayılarını yeni değerlerle üzerine yazmak suretiyle "bozabileceğimizi" gösterir. Bu, ideal olarak öğrenme oranını azaltarak (örneğin, eğitimin sonlarına doğru Q-Tablo değerlerini yalnızca küçük bir değerle ayarlayarak) en aza indirgenmelidir.

Genel olarak, öğrenme sürecinin başarısı ve kalitesi, öğrenme oranı, öğrenme oranı azalması ve indirim faktörü gibi parametrelere önemli ölçüde bağlıdır. Bunlar genellikle **hiperparametreler** olarak adlandırılır, çünkü eğitim sırasında optimize ettiğimiz **parametrelerden** (örneğin, Q-Tablo katsayıları) farklıdırlar. En iyi hiperparametre değerlerini bulma sürecine **hiperparametre optimizasyonu** denir ve bu ayrı bir konuyu hak eder.

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Ödev 
[Daha Gerçekçi Bir Dünya](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.