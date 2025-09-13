<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T20:22:05+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "ms"
}
-->
## Prasyarat

Dalam pelajaran ini, kita akan menggunakan perpustakaan bernama **OpenAI Gym** untuk mensimulasikan pelbagai **persekitaran**. Anda boleh menjalankan kod pelajaran ini secara tempatan (contohnya dari Visual Studio Code), di mana simulasi akan dibuka dalam tetingkap baru. Apabila menjalankan kod secara dalam talian, anda mungkin perlu membuat beberapa penyesuaian pada kod, seperti yang diterangkan [di sini](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Dalam pelajaran sebelumnya, peraturan permainan dan keadaan diberikan oleh kelas `Board` yang kita tentukan sendiri. Di sini kita akan menggunakan **persekitaran simulasi** khas, yang akan mensimulasikan fizik di sebalik tiang yang seimbang. Salah satu persekitaran simulasi yang paling popular untuk melatih algoritma pembelajaran pengukuhan dipanggil [Gym](https://gym.openai.com/), yang diselenggarakan oleh [OpenAI](https://openai.com/). Dengan menggunakan gym ini, kita boleh mencipta pelbagai **persekitaran** daripada simulasi cartpole hingga permainan Atari.

> **Nota**: Anda boleh melihat persekitaran lain yang tersedia dari OpenAI Gym [di sini](https://gym.openai.com/envs/#classic_control).

Pertama, mari pasang gym dan import perpustakaan yang diperlukan (blok kod 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Latihan - inisialisasi persekitaran cartpole

Untuk bekerja dengan masalah keseimbangan cartpole, kita perlu menginisialisasi persekitaran yang sesuai. Setiap persekitaran dikaitkan dengan:

- **Ruang pemerhatian** yang menentukan struktur maklumat yang kita terima daripada persekitaran. Untuk masalah cartpole, kita menerima kedudukan tiang, kelajuan, dan beberapa nilai lain.

- **Ruang tindakan** yang menentukan tindakan yang mungkin. Dalam kes kita, ruang tindakan adalah diskret, dan terdiri daripada dua tindakan - **kiri** dan **kanan**. (blok kod 2)

1. Untuk menginisialisasi, taip kod berikut:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Untuk melihat bagaimana persekitaran berfungsi, mari jalankan simulasi pendek selama 100 langkah. Pada setiap langkah, kita memberikan salah satu tindakan untuk diambil - dalam simulasi ini kita hanya memilih tindakan secara rawak daripada `action_space`.

1. Jalankan kod di bawah dan lihat hasilnya.

    ✅ Ingat bahawa adalah lebih baik untuk menjalankan kod ini pada pemasangan Python tempatan! (blok kod 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Anda sepatutnya melihat sesuatu yang serupa dengan imej ini:

    ![cartpole tidak seimbang](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Semasa simulasi, kita perlu mendapatkan pemerhatian untuk menentukan cara bertindak. Sebenarnya, fungsi langkah mengembalikan pemerhatian semasa, fungsi ganjaran, dan bendera selesai yang menunjukkan sama ada masuk akal untuk meneruskan simulasi atau tidak: (blok kod 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Anda akan melihat sesuatu seperti ini dalam output notebook:

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

    Vektor pemerhatian yang dikembalikan pada setiap langkah simulasi mengandungi nilai berikut:
    - Kedudukan kereta
    - Kelajuan kereta
    - Sudut tiang
    - Kadar putaran tiang

1. Dapatkan nilai minimum dan maksimum bagi nombor-nombor tersebut: (blok kod 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Anda juga mungkin perasan bahawa nilai ganjaran pada setiap langkah simulasi sentiasa 1. Ini kerana matlamat kita adalah untuk bertahan selama mungkin, iaitu mengekalkan tiang pada kedudukan yang agak menegak untuk tempoh masa yang paling lama.

    ✅ Sebenarnya, simulasi CartPole dianggap selesai jika kita berjaya mendapatkan ganjaran purata sebanyak 195 dalam 100 percubaan berturut-turut.

## Diskretisasi keadaan

Dalam Q-Learning, kita perlu membina Q-Table yang menentukan apa yang perlu dilakukan pada setiap keadaan. Untuk dapat melakukan ini, kita memerlukan keadaan untuk menjadi **diskret**, lebih tepat lagi, ia harus mengandungi bilangan nilai diskret yang terhad. Oleh itu, kita perlu **mendiskretkan** pemerhatian kita, memetakan mereka kepada satu set keadaan yang terhad.

Terdapat beberapa cara kita boleh melakukan ini:

- **Bahagikan kepada bin**. Jika kita tahu selang bagi nilai tertentu, kita boleh membahagikan selang ini kepada beberapa **bin**, dan kemudian menggantikan nilai dengan nombor bin yang ia tergolong. Ini boleh dilakukan menggunakan kaedah numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Dalam kes ini, kita akan tahu dengan tepat saiz keadaan, kerana ia akan bergantung pada bilangan bin yang kita pilih untuk digitalisasi.

✅ Kita boleh menggunakan interpolasi linear untuk membawa nilai kepada beberapa selang terhad (contohnya, dari -20 hingga 20), dan kemudian menukar nombor kepada integer dengan membundarkan mereka. Ini memberikan kita kawalan yang kurang terhadap saiz keadaan, terutamanya jika kita tidak tahu julat tepat nilai input. Sebagai contoh, dalam kes kita, 2 daripada 4 nilai tidak mempunyai had atas/bawah pada nilai mereka, yang mungkin menghasilkan bilangan keadaan yang tidak terhingga.

Dalam contoh kita, kita akan menggunakan pendekatan kedua. Seperti yang anda mungkin perasan kemudian, walaupun had atas/bawah tidak ditentukan, nilai-nilai tersebut jarang mengambil nilai di luar selang terhad tertentu, jadi keadaan dengan nilai ekstrem akan sangat jarang berlaku.

1. Berikut adalah fungsi yang akan mengambil pemerhatian daripada model kita dan menghasilkan tuple 4 nilai integer: (blok kod 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Mari kita juga terokai kaedah diskretisasi lain menggunakan bin: (blok kod 7)

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

1. Sekarang mari jalankan simulasi pendek dan perhatikan nilai persekitaran diskret tersebut. Jangan ragu untuk mencuba kedua-dua `discretize` dan `discretize_bins` dan lihat jika terdapat perbezaan.

    ✅ discretize_bins mengembalikan nombor bin, yang berasaskan 0. Oleh itu, untuk nilai pemboleh ubah input sekitar 0, ia mengembalikan nombor dari tengah-tengah selang (10). Dalam discretize, kita tidak peduli tentang julat nilai output, membenarkan mereka menjadi negatif, jadi nilai keadaan tidak beralih, dan 0 sepadan dengan 0. (blok kod 8)

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

    ✅ Nyahkomen baris yang bermula dengan env.render jika anda ingin melihat bagaimana persekitaran dilaksanakan. Jika tidak, anda boleh melaksanakannya di latar belakang, yang lebih cepat. Kita akan menggunakan pelaksanaan "tidak kelihatan" ini semasa proses Q-Learning kita.

## Struktur Q-Table

Dalam pelajaran sebelumnya, keadaan adalah pasangan nombor mudah dari 0 hingga 8, dan oleh itu ia mudah untuk mewakili Q-Table dengan tensor numpy dengan bentuk 8x8x2. Jika kita menggunakan diskretisasi bin, saiz vektor keadaan kita juga diketahui, jadi kita boleh menggunakan pendekatan yang sama dan mewakili keadaan dengan array bentuk 20x20x10x10x2 (di sini 2 adalah dimensi ruang tindakan, dan dimensi pertama sepadan dengan bilangan bin yang kita pilih untuk digunakan bagi setiap parameter dalam ruang pemerhatian).

Walau bagaimanapun, kadangkala dimensi tepat ruang pemerhatian tidak diketahui. Dalam kes fungsi `discretize`, kita mungkin tidak pernah pasti bahawa keadaan kita kekal dalam had tertentu, kerana beberapa nilai asal tidak terikat. Oleh itu, kita akan menggunakan pendekatan yang sedikit berbeza dan mewakili Q-Table dengan kamus.

1. Gunakan pasangan *(state,action)* sebagai kunci kamus, dan nilai akan sepadan dengan nilai entri Q-Table. (blok kod 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Di sini kita juga mentakrifkan fungsi `qvalues()`, yang mengembalikan senarai nilai Q-Table untuk keadaan tertentu yang sepadan dengan semua tindakan yang mungkin. Jika entri tidak hadir dalam Q-Table, kita akan mengembalikan 0 sebagai lalai.

## Mari mulakan Q-Learning

Sekarang kita bersedia untuk mengajar Peter untuk menyeimbangkan!

1. Pertama, mari tetapkan beberapa hiperparameter: (blok kod 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Di sini, `alpha` adalah **kadar pembelajaran** yang menentukan sejauh mana kita harus menyesuaikan nilai semasa Q-Table pada setiap langkah. Dalam pelajaran sebelumnya kita bermula dengan 1, dan kemudian menurunkan `alpha` kepada nilai yang lebih rendah semasa latihan. Dalam contoh ini kita akan mengekalkannya tetap untuk kesederhanaan, dan anda boleh bereksperimen dengan menyesuaikan nilai `alpha` kemudian.

    `gamma` adalah **faktor diskaun** yang menunjukkan sejauh mana kita harus mengutamakan ganjaran masa depan berbanding ganjaran semasa.

    `epsilon` adalah **faktor penerokaan/eksploitasi** yang menentukan sama ada kita harus memilih penerokaan berbanding eksploitasi atau sebaliknya. Dalam algoritma kita, kita akan dalam peratusan `epsilon` kes memilih tindakan seterusnya mengikut nilai Q-Table, dan dalam baki kes kita akan melaksanakan tindakan secara rawak. Ini akan membolehkan kita meneroka kawasan ruang carian yang belum pernah kita lihat sebelum ini.

    ✅ Dalam konteks keseimbangan - memilih tindakan secara rawak (penerokaan) akan bertindak sebagai pukulan rawak ke arah yang salah, dan tiang perlu belajar bagaimana untuk memulihkan keseimbangan daripada "kesilapan" tersebut.

### Meningkatkan algoritma

Kita juga boleh membuat dua penambahbaikan pada algoritma kita daripada pelajaran sebelumnya:

- **Kira ganjaran kumulatif purata**, sepanjang beberapa simulasi. Kita akan mencetak kemajuan setiap 5000 iterasi, dan kita akan mengambil purata ganjaran kumulatif kita sepanjang tempoh masa tersebut. Ini bermakna jika kita mendapat lebih daripada 195 mata - kita boleh menganggap masalah itu selesai, dengan kualiti yang lebih tinggi daripada yang diperlukan.

- **Kira hasil kumulatif purata maksimum**, `Qmax`, dan kita akan menyimpan Q-Table yang sepadan dengan hasil tersebut. Apabila anda menjalankan latihan, anda akan perasan bahawa kadangkala hasil kumulatif purata mula menurun, dan kita mahu menyimpan nilai Q-Table yang sepadan dengan model terbaik yang diperhatikan semasa latihan.

1. Kumpulkan semua ganjaran kumulatif pada setiap simulasi dalam vektor `rewards` untuk plot kemudian. (blok kod 11)

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

Apa yang anda mungkin perasan daripada hasil tersebut:

- **Hampir mencapai matlamat**. Kita sangat hampir mencapai matlamat mendapatkan 195 ganjaran kumulatif dalam lebih daripada 100 percubaan berturut-turut simulasi, atau kita mungkin telah mencapainya! Walaupun kita mendapat nombor yang lebih kecil, kita masih tidak tahu, kerana kita mengambil purata lebih daripada 5000 percubaan, dan hanya 100 percubaan diperlukan dalam kriteria formal.

- **Ganjaran mula menurun**. Kadangkala ganjaran mula menurun, yang bermaksud kita boleh "merosakkan" nilai yang telah dipelajari dalam Q-Table dengan nilai yang menjadikan keadaan lebih buruk.

Pemerhatian ini lebih jelas kelihatan jika kita memplotkan kemajuan latihan.

## Memplotkan Kemajuan Latihan

Semasa latihan, kita telah mengumpulkan nilai ganjaran kumulatif pada setiap iterasi ke dalam vektor `rewards`. Berikut adalah bagaimana ia kelihatan apabila kita memplotkannya terhadap nombor iterasi:

```python
plt.plot(rewards)
```

![kemajuan mentah](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Daripada graf ini, tidak mungkin untuk memberitahu apa-apa, kerana disebabkan sifat proses latihan stokastik, panjang sesi latihan berbeza dengan ketara. Untuk membuat graf ini lebih bermakna, kita boleh mengira **purata berjalan** sepanjang siri eksperimen, katakan 100. Ini boleh dilakukan dengan mudah menggunakan `np.convolve`: (blok kod 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![kemajuan latihan](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Mengubah hiperparameter

Untuk membuat pembelajaran lebih stabil, masuk akal untuk menyesuaikan beberapa hiperparameter kita semasa latihan. Khususnya:

- **Untuk kadar pembelajaran**, `alpha`, kita boleh bermula dengan nilai yang hampir dengan 1, dan kemudian terus menurunkan parameter. Dengan masa, kita akan mendapat nilai kebarangkalian yang baik dalam Q-Table, dan oleh itu kita harus menyesuaikannya sedikit, dan tidak menulis semula sepenuhnya dengan nilai baru.

- **Tingkatkan epsilon**. Kita mungkin mahu meningkatkan `epsilon` secara perlahan, untuk meneroka kurang dan mengeksploitasi lebih banyak. Mungkin masuk akal untuk bermula dengan nilai `epsilon` yang lebih rendah, dan meningkatkannya hampir kepada 1.
> **Tugas 1**: Cuba ubah nilai hiperparameter dan lihat jika anda boleh mencapai ganjaran kumulatif yang lebih tinggi. Adakah anda mendapat lebih daripada 195?
> **Tugas 2**: Untuk menyelesaikan masalah ini secara formal, anda perlu mencapai purata ganjaran sebanyak 195 dalam 100 larian berturut-turut. Ukur semasa latihan dan pastikan anda telah menyelesaikan masalah ini secara formal!

## Melihat hasil dalam tindakan

Ia akan menarik untuk melihat bagaimana model yang telah dilatih berfungsi. Mari jalankan simulasi dan ikuti strategi pemilihan tindakan yang sama seperti semasa latihan, dengan pensampelan mengikut taburan kebarangkalian dalam Q-Table: (blok kod 13)

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

Anda sepatutnya melihat sesuatu seperti ini:

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Cabaran

> **Tugas 3**: Di sini, kita menggunakan salinan akhir Q-Table, yang mungkin bukan yang terbaik. Ingat bahawa kita telah menyimpan Q-Table yang berprestasi terbaik ke dalam pemboleh ubah `Qbest`! Cuba contoh yang sama dengan Q-Table yang berprestasi terbaik dengan menyalin `Qbest` ke `Q` dan lihat jika anda perasan perbezaannya.

> **Tugas 4**: Di sini kita tidak memilih tindakan terbaik pada setiap langkah, tetapi sebaliknya melakukan pensampelan dengan taburan kebarangkalian yang sepadan. Adakah lebih masuk akal untuk sentiasa memilih tindakan terbaik, dengan nilai Q-Table tertinggi? Ini boleh dilakukan dengan menggunakan fungsi `np.argmax` untuk mencari nombor tindakan yang sepadan dengan nilai Q-Table tertinggi. Laksanakan strategi ini dan lihat jika ia meningkatkan keseimbangan.

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tugasan
[Latih Mountain Car](assignment.md)

## Kesimpulan

Kita kini telah belajar bagaimana melatih agen untuk mencapai hasil yang baik hanya dengan menyediakan fungsi ganjaran yang menentukan keadaan permainan yang diinginkan, dan dengan memberi mereka peluang untuk meneroka ruang carian secara bijak. Kita telah berjaya menggunakan algoritma Q-Learning dalam kes persekitaran diskret dan berterusan, tetapi dengan tindakan diskret.

Adalah penting untuk juga mengkaji situasi di mana keadaan tindakan juga berterusan, dan apabila ruang pemerhatian jauh lebih kompleks, seperti imej dari skrin permainan Atari. Dalam masalah tersebut, kita sering perlu menggunakan teknik pembelajaran mesin yang lebih berkuasa, seperti rangkaian neural, untuk mencapai hasil yang baik. Topik yang lebih maju ini adalah subjek kursus AI lanjutan kita yang akan datang.

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.