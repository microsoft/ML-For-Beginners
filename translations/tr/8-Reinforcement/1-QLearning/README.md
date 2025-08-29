## Politikayı kontrol etme

Q-Tablosu her durumdaki her eylemin "çekiciliğini" listeler, bu nedenle dünyamızda verimli navigasyonu tanımlamak oldukça kolaydır. En basit durumda, en yüksek Q-Tablosu değerine karşılık gelen eylemi seçebiliriz: (kod bloğu 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Yukarıdaki kodu birkaç kez denerseniz, bazen "takıldığını" ve kesmek için not defterindeki DURDUR düğmesine basmanız gerektiğini fark edebilirsiniz. Bu, iki durumun optimal Q-Değeri açısından birbirine "işaret ettiği" durumlar olabileceğinden, bu durumda ajan bu durumlar arasında sonsuz bir şekilde hareket etmeye başlar.

## 🚀Meydan Okuma

> **Görev 1:** `walk` function to limit the maximum length of path by a certain number of steps (say, 100), and watch the code above return this value from time to time.

> **Task 2:** Modify the `walk` function so that it does not go back to the places where it has already been previously. This will prevent `walk` from looping, however, the agent can still end up being "trapped" in a location from which it is unable to escape.

## Navigation

A better navigation policy would be the one that we used during training, which combines exploitation and exploration. In this policy, we will select each action with a certain probability, proportional to the values in the Q-Table. This strategy may still result in the agent returning back to a position it has already explored, but, as you can see from the code below, it results in a very short average path to the desired location (remember that `print_statistics` simülasyonu 100 kez çalıştıracak şekilde değiştirin: (kod bloğu 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Bu kodu çalıştırdıktan sonra, önceki ortalama yol uzunluğundan çok daha küçük bir ortalama yol uzunluğu elde etmelisiniz, 3-6 aralığında.

## Öğrenme sürecini araştırma

Belirttiğimiz gibi, öğrenme süreci, problem alanının yapısı hakkında elde edilen bilgilerin keşfi ve keşfi arasında bir dengedir. Öğrenme sonuçlarının (bir ajanın hedefe kısa bir yol bulma yeteneği) iyileştiğini gördük, ancak öğrenme sürecinde ortalama yol uzunluğunun nasıl davrandığını gözlemlemek de ilginçtir:

Öğrenilenler şu şekilde özetlenebilir:

- **Ortalama yol uzunluğu artar**. Burada gördüğümüz şey, başlangıçta ortalama yol uzunluğunun arttığıdır. Bu, çevre hakkında hiçbir şey bilmediğimizde, kötü durumlara, suya veya kurda yakalanma olasılığımızın yüksek olması nedeniyle olabilir. Daha fazla bilgi edindikçe ve bu bilgiyi kullanmaya başladıkça, çevreyi daha uzun süre keşfedebiliriz, ancak elma nerede olduğunu hala çok iyi bilmiyoruz.

- **Öğrendikçe yol uzunluğu azalır**. Yeterince öğrendiğimizde, ajanın hedefe ulaşması daha kolay hale gelir ve yol uzunluğu azalmaya başlar. Ancak, keşfe hala açığız, bu yüzden genellikle en iyi yoldan saparız ve yeni seçenekleri keşfederiz, bu da yolu optimalden daha uzun hale getirir.

- **Uzunluk ani bir şekilde artar**. Bu grafikte ayrıca, bir noktada uzunluğun ani bir şekilde arttığını gözlemliyoruz. Bu, sürecin stokastik doğasını ve bir noktada Q-Tablosu katsayılarını yeni değerlerle üzerine yazarak "bozabileceğimizi" gösterir. Bu, ideal olarak öğrenme oranını azaltarak en aza indirilmelidir (örneğin, eğitimin sonuna doğru, Q-Tablosu değerlerini sadece küçük bir değerle ayarlayarak).

Genel olarak, öğrenme sürecinin başarısı ve kalitesinin, öğrenme oranı, öğrenme oranı düşüşü ve indirim faktörü gibi parametrelere önemli ölçüde bağlı olduğunu hatırlamak önemlidir. Bunlar genellikle **hiperparametreler** olarak adlandırılır, çünkü eğitim sırasında optimize ettiğimiz **parametrelerden** (örneğin, Q-Tablosu katsayıları) farklıdır. En iyi hiperparametre değerlerini bulma sürecine **hiperparametre optimizasyonu** denir ve ayrı bir konuyu hak eder.

## [Ders sonrası quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## Ödev
[Daha Gerçekçi Bir Dünya](assignment.md)

**Feragatname**:
Bu belge, makine tabanlı yapay zeka çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Orijinal belge, kendi dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.