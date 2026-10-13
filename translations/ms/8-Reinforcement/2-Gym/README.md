# Luncur CartPole

Masalah yang telah kita selesaikan dalam pelajaran sebelumnya mungkin kelihatan seperti masalah mainan, tidak benar-benar sesuai untuk situasi kehidupan sebenar. Ini tidak benar, kerana banyak masalah dunia sebenar juga berkongsi senario ini - termasuk bermain Catur atau Go. Mereka serupa, kerana kita juga mempunyai papan dengan peraturan yang ditetapkan dan **keadaan diskret**.

## [Kuiz pra-ceramah](https://ff-quizzes.netlify.app/en/ml/)

## Pengenalan

Dalam pelajaran ini kita akan menerapkan prinsip yang sama dari Q-Learning kepada masalah dengan **keadaan berterusan**, iaitu keadaan yang diberikan oleh satu atau lebih nombor nyata. Kita akan menangani masalah berikut:

> **Masalah**: Jika Peter ingin melarikan diri dari serigala, dia perlu dapat bergerak lebih cepat. Kita akan lihat bagaimana Peter boleh belajar meluncur, khususnya, untuk mengekalkan keseimbangan, menggunakan Q-Learning.

![The great escape!](../../../../translated_images/ms/escape.18862db9930337e3.webp)

> Peter dan kawan-kawannya menjadi kreatif untuk melarikan diri dari serigala! Gambar oleh [Jen Looper](https://twitter.com/jenlooper)

Kita akan menggunakan versi ringkas keseimbangan yang dikenali sebagai masalah **CartPole**. Dalam dunia cartpole, kita mempunyai peluncur mendatar yang boleh bergerak ke kiri atau kanan, dan tujuannya adalah untuk menyeimbangkan tiang menegak di atas peluncur.

<img alt="a cartpole" src="../../../../translated_images/ms/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Prasyarat

Dalam pelajaran ini, kita akan menggunakan perpustakaan yang dipanggil **OpenAI Gym** untuk mensimulasikan pelbagai **persekitaran**. Anda boleh menjalankan kod pelajaran ini secara tempatan (contoh dari Visual Studio Code), dalam kes ini simulasi akan dibuka dalam tetingkap baru. Apabila menjalankan kod secara dalam talian, anda mungkin perlu membuat beberapa pengubahsuaian pada kod, seperti yang diterangkan [di sini](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Dalam pelajaran sebelumnya, peraturan permainan dan keadaan diberikan oleh kelas `Board` yang kita definisikan sendiri. Di sini kita akan menggunakan **persekitaran simulasi** khas, yang akan mensimulasikan fizik di sebalik tiang keseimbangan. Salah satu persekitaran simulasi yang paling popular untuk melatih algoritma pembelajaran penguatan dikenali sebagai [Gym](https://gym.openai.com/), yang dikendalikan oleh [OpenAI](https://openai.com/). Dengan menggunakan gym ini kita boleh mencipta pelbagai **persekitaran** dari simulasi cartpole hingga permainan Atari.

> **Nota**: Anda boleh melihat persekitaran lain yang tersedia di OpenAI Gym [di sini](https://gym.openai.com/envs/#classic_control). 

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

Untuk bekerja dengan masalah keseimbangan cartpole, kita perlu menginisialisasi persekitaran yang bersesuaian. Setiap persekitaran dikaitkan dengan:

- **Ruang pemerhatian** yang mentakrifkan struktur maklumat yang kita terima dari persekitaran. Untuk masalah cartpole, kita menerima posisi tiang, halaju dan beberapa nilai lain.

- **Ruang tindakan** yang mentakrifkan tindakan yang mungkin. Dalam kes kita, ruang tindakan adalah diskret, dan terdiri daripada dua tindakan - **kiri** dan **kanan**. (blok kod 2)

1. Untuk menginisialisasi, taip kod berikut:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Untuk melihat bagaimana persekitaran berfungsi, mari jalankan simulasi pendek selama 100 langkah. Pada setiap langkah, kita menyediakan salah satu tindakan yang akan diambil - dalam simulasi ini kita hanya memilih tindakan secara rawak dari `action_space`.

1. Jalankan kod di bawah dan lihat apa hasilnya.

    ✅ Ingat bahawa lebih digalakkan untuk menjalankan kod ini pada pemasangan Python tempatan! (blok kod 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Anda sepatutnya melihat sesuatu yang serupa dengan imej ini:

    ![cartpole tidak seimbang](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Semasa simulasi, kita perlu mendapatkan pemerhatian untuk menentukan bagaimana bertindak. Sebenarnya, fungsi step mengembalikan pemerhatian semasa, fungsi ganjaran, dan bendera done yang menunjukkan sama ada simulasi perlu diteruskan atau tidak: (blok kod 4)

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
    - Posisi gerabak
    - Halaju gerabak
    - Sudut tiang
    - Kadar putaran tiang

1. Dapatkan nilai minimum dan maksimum bagi nombor-nombor tersebut: (blok kod 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Anda juga mungkin perasan bahawa nilai ganjaran pada setiap langkah simulasi sentiasa 1. Ini kerana tujuan kita adalah untuk bertahan selama mungkin, iaitu mengekalkan tiang pada posisi hampir menegak untuk tempoh masa terpanjang.

    ✅ Sebenarnya, simulasi CartPole dianggap selesai jika kita berjaya mendapat purata ganjaran sebanyak 195 selama 100 ujian berturut-turut.

## Diskretisasi keadaan

Dalam Q-Learning, kita perlu membina Q-Table yang menetapkan apa yang perlu dilakukan pada setiap keadaan. Untuk melakukan ini, kita perlukan keadaan yang **diskret**, dengan lebih tepat, ia harus mengandungi bilangan nilai diskret yang terhingga. Oleh itu, kita perlu **diskretkan** pemerhatian kita, memetakannya ke set terhingga keadaan.

Terdapat beberapa cara yang kita boleh lakukan ini:

- **Bahagikan kepada bin**. Jika kita tahu selang nilai tertentu, kita boleh bahagikan selang ini kepada beberapa **bin**, dan kemudian gantikan nilai tersebut dengan nombor bin yang dimilikinya. Ini boleh dilakukan menggunakan kaedah numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Dalam kes ini, kita akan tahu dengan tepat saiz keadaan, kerana ia bergantung pada bilangan bin yang kita pilih untuk digitalisasi.
  
✅ Kita boleh menggunakan interpolasi linear untuk membawa nilai ke dalam selang terhingga (kata, dari -20 hingga 20), kemudian menukar nombor kepada integer dengan pembulatan. Ini memberi kita kurang kawalan terhadap saiz keadaan, terutamanya jika kita tidak tahu jarak input sebenar. Sebagai contoh, dalam kes kita, 2 daripada 4 nilai tidak mempunyai batas atas/bawah yang ditetapkan, yang boleh menyebabkan bilangan keadaan tak terhingga.

Dalam contoh kita, kita akan memilih pendekatan kedua. Seperti yang anda mungkin perasan kemudian, walaupun batas atas/bawah tidak ditentukan, nilai-nilai ini jarang mengambil nilai di luar selang terhingga tertentu, jadi keadaan dengan nilai ekstrim sangat jarang.

1. Berikut adalah fungsi yang akan mengambil pemerhatian dari model kita dan menghasilkan tuple empat nilai integer: (blok kod 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Mari juga terokai satu lagi kaedah diskretisasi menggunakan bin: (blok kod 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # selang nilai untuk setiap parameter
    nbins = [20,20,10,10] # bilangan bin untuk setiap parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Sekarang jalankan simulasi pendek dan perhatikan nilai persekitaran diskret tersebut. Sila cuba kedua-dua `discretize` dan `discretize_bins` dan lihat jika ada perbezaan.

    ✅ discretize_bins mengembalikan nombor bin, yang bermula dari 0. Jadi bagi nilai pemboleh ubah input sekitar 0 ia mengembalikan nombor dari tengah interval (10). Dalam discretize, kita tidak mengambil kira julat nilai keluaran, membenarkan nilai negatif, jadi nilai keadaan tidak berganjak, dan 0 merujuk kepada 0. (blok kod 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(diskretkan_bin(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Buka komen pada baris yang bermula dengan env.render jika anda mahu melihat bagaimana persekitaran berjalan. Jika tidak, anda boleh menjalankannya di latar belakang, yang lebih pantas. Kita akan menggunakan pelaksanaan "tidak nampak" ini sepanjang proses Q-Learning kita.

## Struktur Q-Table

Dalam pelajaran sebelumnya, keadaan adalah sepasang nombor dari 0 hingga 8, dan oleh itu mudah untuk mewakili Q-Table dengan tensor numpy berbentuk 8x8x2. Jika kita menggunakan diskretisasi bin, saiz vektor keadaan juga diketahui, jadi kita boleh menggunakan pendekatan yang sama dan wakili keadaan dengan array berbentuk 20x20x10x10x2 (di sini 2 adalah dimensi ruang tindakan, dan dimensi pertama merujuk kepada bilangan bin yang kita pilih untuk setiap parameter dalam ruang pemerhatian).

Walau bagaimanapun, kadangkala dimensi tepat ruang pemerhatian tidak diketahui. Dalam kes fungsi `discretize`, kita mungkin tidak pasti sama ada keadaan kita kekal dalam had tertentu, kerana beberapa nilai asal tidak terbatas. Oleh itu, kita akan menggunakan pendekatan sedikit berbeza dan mewakili Q-Table dengan kamus. 

1. Gunakan pasangan *(state,action)* sebagai kekunci kamus, dan nilai akan merujuk kepada entri Q-Table. (blok kod 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Di sini kita juga mentakrifkan fungsi `qvalues()`, yang mengembalikan senarai nilai Q-Table untuk keadaan tertentu yang meliputi semua tindakan yang mungkin. Jika entri tidak ada dalam Q-Table, kita akan mengembalikan 0 sebagai nilai lalai.

## Mari mulakan Q-Learning

Sekarang kita bersedia untuk mengajar Peter cara mengekalkan keseimbangan!

1. Pertama, mari tetapkan beberapa hiperparameter: (blok kod 10)

    ```python
    # hiperparameter
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Di sini, `alpha` adalah **kadar pembelajaran** yang mentakrifkan sejauh mana kita harus melaraskan nilai Q-Table pada setiap langkah. Dalam pelajaran sebelumnya kita bermula dengan 1, kemudian mengurangkan `alpha` kepada nilai lebih rendah semasa latihan. Dalam contoh ini kita akan mengekalkannya sebagai tetap untuk kesederhanaan, dan anda boleh bereksperimen dengan melaraskan nilai `alpha` kemudian.

    `gamma` adalah **faktor diskaun** yang menunjukkan sejauh mana kita harus mengutamakan ganjaran masa depan berbanding ganjaran semasa.

    `epsilon` adalah **faktor eksplorasi/eksploitasi** yang menentukan sama ada kita harus mengutamakan eksplorasi berbanding eksploitasi atau sebaliknya. Dalam algoritma kita, dalam peratusan `epsilon` kes kita akan memilih tindakan seterusnya menurut nilai Q-Table, dan dalam baki kes kita akan melakukan tindakan rawak. Ini membolehkan kita meneroka kawasan ruang carian yang belum pernah dilihat sebelum ini.

    ✅ Dalam konteks keseimbangan - memilih tindakan rawak (eksplorasi) bertindak seperti tumbukan rawak ke arah yang salah, dan tiang perlu belajar bagaimana memulihkan keseimbangan dari "kesilapan" tersebut.

### Memperbaiki algoritma

Kita juga boleh membuat dua penambahbaikan pada algoritma kita dari pelajaran sebelumnya:

- **Kira ganjaran kumulatif purata**, sepanjang beberapa simulasi. Kita akan mencetak kemajuan setiap 5000 iterasi, dan kita akan puratakan ganjaran kumulatif kita sepanjang tempoh masa tersebut. Ini bermakna jika kita mendapat lebih daripada 195 mata - kita boleh anggap masalah selesai, dengan kualiti lebih tinggi daripada yang diperlukan.
  
- **Kira hasil kumulatif purata maksimum**, `Qmax`, dan kita akan simpan Q-Table yang sepadan dengan hasil itu. Apabila anda menjalankan latihan, anda akan perasan kadang-kadang hasil kumulatif purata mula menurun, dan kita mahu mengekalkan nilai Q-Table yang sepadan dengan model terbaik yang diperhatikan semasa latihan.

1. Kumpul semua ganjaran kumulatif setiap simulasi dalam vektor `rewards` untuk plot kemudian. (blok kod  11)

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
        # == lakukan simulasi ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # eksploitasi - pilih tindakan menurut kebarangkalian Jadual-Q
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # eksplorasi - pilih tindakan secara rawak
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Cetak hasil secara berkala dan kira purata ganjaran ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Apa yang mungkin anda perhatikan dari keputusan itu:

- **Hampir mencapai matlamat**. Kita sangat hampir dengan mencapai matlamat mendapat 195 ganjaran kumulatif selama lebih 100 larian berturut-turut simulasi, atau kita mungkin sudah mencapainya! Walaupun kita mendapat nombor lebih rendah, kita masih tidak pasti kerana kita mempuratakan atas 5000 larian, dan hanya 100 larian diperlukan dalam kriteria rasmi.
  
- **Ganjaran mula menurun**. Kadang-kadang ganjaran mula menurun, yang bermaksud kita boleh "menghancurkan" nilai yang telah dipelajari dalam Q-Table dengan nilai yang membuat keadaan lebih buruk.

Pemerhatian ini lebih jelas jika kita plot kemajuan latihan.

## Plot Kemajuan Latihan

Semasa latihan, kita telah mengumpul nilai ganjaran kumulatif pada setiap iterasi ke dalam vektor `rewards`. Inilah bagaimana ia kelihatan apabila kita plot terhadap nombor iterasi:

```python
plt.plot(rewards)
```

![raw  progress](../../../../translated_images/ms/train_progress_raw.2adfdf2daea09c59.webp)

Daripada graf ini, tidak mungkin untuk membuat kesimpulan, kerana sifat proses latihan stokastik menyebabkan panjang sesi latihan sangat berbeza-beza. Untuk membuatkan graf ini lebih bermakna, kita boleh kira **purata bergerak** sepanjang beberapa eksperimen, katakan 100. Ini boleh dilakukan dengan mudah menggunakan `np.convolve`: (blok kod 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![training progress](../../../../translated_images/ms/train_progress_runav.c71694a8fa9ab359.webp)

## Menukar hiperparameter

Untuk menjadikan pembelajaran lebih stabil, masuk akal untuk melaraskan beberapa hiperparameter semasa latihan. Khususnya:

- **Untuk kadar pembelajaran**, `alpha`, kita boleh mulakan dengan nilai hampir 1, kemudian kurangkan parameter ini. Dengan masa, kita akan mendapat nilai kebarangkalian yang baik dalam Q-Table, jadi kita perlu melaraskannya sedikit, dan bukan menimpa sepenuhnya dengan nilai baru.

- **Tingkatkan epsilon**. Kita mungkin mahu tingkatkan `epsilon` perlahan-lahan, supaya kurang meneroka dan lebih mengeksploitasi. Mungkin masuk akal bermula dengan nilai `epsilon` rendah, dan naik ke hampir 1.

> **Tugasan 1**: Cuba bermain dengan nilai hiperparameter dan lihat jika anda boleh mencapai ganjaran kumulatif yang lebih tinggi. Adakah anda mendapat lebih daripada 195?


> **Tugasan 2**: Untuk menyelesaikan masalah secara rasmi, anda perlu mendapatkan purata ganjaran sebanyak 195 sepanjang 100 larian berturut-turut. Ukur itu semasa latihan dan pastikan anda telah menyelesaikan masalah secara rasmi!

## Melihat hasil dalam tindakan

Ia akan menjadi menarik untuk benar-benar melihat bagaimana model yang dilatih berkelakuan. Mari jalankan simulasi dan ikut strategi pemilihan tindakan yang sama seperti semasa latihan, sampel mengikut taburan kebarangkalian dalam Q-Table: (blok kod 13)

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

Anda harus melihat sesuatu seperti ini:

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Cabaran

> **Tugasan 3**: Di sini, kami menggunakan salinan terakhir Q-Table, yang mungkin bukan yang terbaik. Ingat bahawa kami telah menyimpan Q-Table dengan prestasi terbaik ke dalam pembolehubah `Qbest`! Cuba contoh yang sama dengan Q-Table prestasi terbaik dengan menyalin `Qbest` ke `Q` dan lihat jika anda perasan perbezaannya.

> **Tugasan 4**: Di sini kami tidak memilih tindakan terbaik pada setiap langkah, tetapi sebaliknya menyampel dengan taburan kebarangkalian yang sepadan. Adakah lebih masuk akal untuk sentiasa memilih tindakan terbaik, dengan nilai Q-Table tertinggi? Ini boleh dilakukan dengan menggunakan fungsi `np.argmax` untuk mencari nombor tindakan yang sepadan dengan nilai Q-Table tertinggi. Laksanakan strategi ini dan lihat jika ia meningkatkan keseimbangan.

## [Kuiz pasca kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tugasan
[Latih Kereta Gunung](assignment.md)

## Kesimpulan

Kami kini telah belajar bagaimana untuk melatih agen untuk mencapai keputusan yang baik hanya dengan memberikan mereka fungsi ganjaran yang mentakrifkan keadaan permainan yang dikehendaki, dan dengan memberi mereka peluang untuk menjelajah ruang carian secara bijak. Kami telah berjaya menggunakan algoritma Q-Learning dalam kes persekitaran diskret dan berterusan, tetapi dengan tindakan diskret.

Ia penting juga untuk mengkaji situasi di mana keadaan tindakan juga berterusan, dan apabila ruang pemerhatian jauh lebih kompleks, seperti imej daripada skrin permainan Atari. Dalam masalah tersebut, kami sering perlu menggunakan teknik pembelajaran mesin yang lebih berkuasa, seperti rangkaian neural, untuk mencapai keputusan yang baik. Topik yang lebih maju ini adalah subjek kursus AI lanjutan kami yang akan datang.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk ketepatan, sila ambil maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang sahih. Untuk maklumat penting, terjemahan oleh manusia profesional adalah disyorkan. Kami tidak bertanggungjawab terhadap sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->