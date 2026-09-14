# CartPole Skating

Masalah yang telah kita selesaikan dalam pelajaran sebelumnya mungkin tampak seperti masalah mainan, tidak benar-benar berlaku untuk skenario kehidupan nyata. Ini bukanlah kasusnya, karena banyak masalah dunia nyata juga berbagi skenario ini - termasuk bermain Catur atau Go. Mereka serupa, karena kita juga memiliki papan dengan aturan tertentu dan **state diskrit**.

## [Kuis Pra-lecture](https://ff-quizzes.netlify.app/en/ml/)

## Pendahuluan

Dalam pelajaran ini kita akan menerapkan prinsip yang sama dari Q-Learning untuk sebuah masalah dengan **state kontinu**, yaitu state yang diberikan oleh satu atau lebih angka riil. Kita akan menangani masalah berikut:

> **Masalah**: Jika Peter ingin melarikan diri dari serigala, ia harus bisa bergerak lebih cepat. Kita akan melihat bagaimana Peter dapat belajar bermain skating, khususnya, menjaga keseimbangan, menggunakan Q-Learning.

![The great escape!](../../../../translated_images/id/escape.18862db9930337e3.webp)

> Peter dan teman-temannya menjadi kreatif untuk melarikan diri dari serigala! Gambar oleh [Jen Looper](https://twitter.com/jenlooper)

Kita akan menggunakan versi sederhana dari keseimbangan yang dikenal sebagai masalah **CartPole**. Dalam dunia cartpole, kita memiliki sebuah slider horizontal yang dapat bergerak ke kiri atau kanan, dan tujuannya adalah menyeimbangkan tiang vertikal di atas slider.

<img alt="a cartpole" src="../../../../translated_images/id/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Prasyarat

Dalam pelajaran ini, kita akan menggunakan sebuah pustaka bernama **OpenAI Gym** untuk mensimulasikan berbagai **lingkungan**. Kamu dapat menjalankan kode pelajaran ini secara lokal (misalnya dari Visual Studio Code), dalam hal ini simulasi akan terbuka di jendela baru. Saat menjalankan kode secara online, kamu mungkin perlu melakukan beberapa penyesuaian pada kode, seperti yang dijelaskan [di sini](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Dalam pelajaran sebelumnya, aturan permainan dan state diberikan oleh kelas `Board` yang kita definisikan sendiri. Di sini kita akan menggunakan **lingkungan simulasi** khusus, yang akan mensimulasikan fisika di balik tiang yang seimbang. Salah satu lingkungan simulasi paling populer untuk melatih algoritma reinforcement learning disebut [Gym](https://gym.openai.com/), yang dikelola oleh [OpenAI](https://openai.com/). Dengan menggunakan gym ini kita dapat membuat berbagai **lingkungan** mulai dari simulasi cartpole hingga permainan Atari.

> **Catatan**: Kamu dapat melihat lingkungan lain yang tersedia dari OpenAI Gym [di sini](https://gym.openai.com/envs/#classic_control). 

Pertama, mari kita instal gym dan impor pustaka yang dibutuhkan (blok kode 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Latihan - inisialisasi lingkungan cartpole

Untuk bekerja dengan masalah keseimbangan cartpole, kita perlu menginisialisasi lingkungan yang sesuai. Setiap lingkungan terkait dengan:

- **Ruang observasi** yang mendefinisikan struktur informasi yang kita terima dari lingkungan. Untuk masalah cartpole, kita menerima posisi tiang, kecepatan dan beberapa nilai lainnya.

- **Ruang aksi** yang mendefinisikan kemungkinan tindakan. Dalam kasus kita ruang aksi bersifat diskrit, dan terdiri dari dua tindakan - **kiri** dan **kanan**. (blok kode 2)

1. Untuk menginisialisasi, ketik kode berikut:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Untuk melihat bagaimana lingkungan bekerja, mari jalankan simulasi singkat selama 100 langkah. Pada setiap langkah, kita memberikan salah satu tindakan yang akan diambil - dalam simulasi ini kita hanya memilih tindakan secara acak dari `action_space`. 

1. Jalankan kode di bawah dan lihat hasilnya.

    ✅ Ingat, sebaiknya jalankan kode ini pada instalasi Python lokal! (blok kode 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Kamu akan melihat sesuatu yang mirip dengan gambar ini:

    ![non-balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Selama simulasi, kita perlu mendapatkan observasi untuk memutuskan bagaimana bertindak. Faktanya, fungsi step mengembalikan observasi saat ini, fungsi reward, dan flag done yang menunjukkan apakah simulasi harus diteruskan atau tidak: (blok kode 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Kamu akan melihat sesuatu seperti ini di output notebook:

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

    Vektor observasi yang dikembalikan pada setiap langkah simulasi berisi nilai-nilai berikut:
    - Posisi cart
    - Kecepatan cart
    - Sudut tiang
    - Laju rotasi tiang

1. Dapatkan nilai minimum dan maksimum dari angka-angka tersebut: (blok kode 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Kamu mungkin juga menyadari bahwa nilai reward pada setiap langkah simulasi selalu 1. Ini karena tujuan kita adalah bertahan selama mungkin, yaitu menjaga tiang dalam posisi yang relatif vertikal selama periode waktu terpanjang.

    ✅ Faktanya, simulasi CartPole dianggap berhasil jika kita berhasil mendapatkan rata-rata reward 195 selama 100 percobaan berturut-turut.

## Diskretisasi State

Dalam Q-Learning, kita perlu membangun Q-Table yang mendefinisikan apa yang harus dilakukan pada setiap state. Agar bisa melakukan ini, kita perlu agar state menjadi **diskrit**, lebih tepatnya, harus berisi jumlah nilai diskrit yang terbatas. Maka dari itu, kita perlu **mendiskretisasi** observasi kita, memetakan mereka ke sebuah himpunan nilai diskrit yang terbatas.

Ada beberapa cara untuk melakukan ini:

- **Membagi ke dalam bins**. Jika kita mengetahui interval dari nilai tertentu, kita dapat membagi interval ini menjadi sejumlah **bins**, dan kemudian mengganti nilai dengan nomor bin yang sesuai. Ini dapat dilakukan menggunakan metode numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Dalam hal ini, kita akan tahu dengan tepat ukuran state karena tergantung pada jumlah bins yang kita pilih untuk digitalisasi.
  
✅ Kita dapat menggunakan interpolasi linier untuk membawa nilai ke interval terbatas (misalnya, dari -20 sampai 20), kemudian mengubah angka menjadi bilangan bulat dengan pembulatan. Ini memberi kita kendali yang sedikit kurang pada ukuran state, terutama jika kita tidak mengetahui rentang pasti nilai input. Misalnya, dalam kasus kita 2 dari 4 nilai tidak memiliki batas atas/bawah yang pasti, yang dapat menghasilkan jumlah state yang tak hingga.

Dalam contoh kita, kita akan menggunakan pendekatan kedua. Seperti yang akan kamu lihat nanti, meskipun batas atas/bawah tidak terdefinisi, nilai-nilai tersebut jarang mengambil nilai di luar interval tertentu yang terbatas, sehingga state dengan nilai ekstrim akan sangat jarang.

1. Berikut fungsi yang akan mengambil observasi dari model kita dan menghasilkan tuple dari 4 nilai bilangan bulat: (blok kode 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Mari kita juga jelajahi metode diskretisasi lain menggunakan bins: (blok kode 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # interval nilai untuk setiap parameter
    nbins = [20,20,10,10] # jumlah bin untuk setiap parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Sekarang mari jalankan simulasi singkat dan perhatikan nilai lingkungan diskrit tersebut. Silakan coba kedua `discretize` dan `discretize_bins` dan lihat apakah ada perbedaan.

    ✅ discretize_bins mengembalikan nomor bin yang berbasis 0. Jadi untuk nilai variabel input sekitar 0, ini mengembalikan nomor dari tengah interval (10). Dalam discretize, kita tidak memperhatikan rentang nilai output, mengizinkan nilai negatif, sehingga nilai state tidak bergeser, dan 0 sesuai dengan 0. (blok kode 8)

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

    ✅ Batalkan komentar baris yang diawali dengan env.render jika kamu ingin melihat bagaimana lingkungan mengeksekusi. Jika tidak, kamu dapat mengeksekusinya di latar belakang, yang lebih cepat. Kita akan menggunakan eksekusi "tidak terlihat" ini selama proses Q-Learning.

## Struktur Q-Table

Dalam pelajaran sebelumnya, state adalah pasangan angka sederhana dari 0 sampai 8, sehingga mudah merepresentasikan Q-Table dengan tensor numpy berbentuk 8x8x2. Jika kita menggunakan diskretisasi bins, ukuran vektor state juga diketahui, sehingga kita dapat menggunakan pendekatan yang sama dan merepresentasikan state dengan array berbentuk 20x20x10x10x2 (di sini 2 adalah dimensi ruang aksi, dan dimensi pertama sesuai dengan jumlah bins yang kita pilih untuk masing-masing parameter dalam ruang observasi).

Namun, terkadang dimensi tepat dari ruang observasi tidak diketahui. Dalam kasus fungsi `discretize`, kita mungkin tidak pernah yakin state kita tetap dalam batas tertentu, karena beberapa nilai asli tidak dibatasi. Oleh karena itu, kita akan menggunakan pendekatan yang sedikit berbeda dan merepresentasikan Q-Table dengan dictionary.

1. Gunakan pasangan *(state, action)* sebagai kunci dictionary, dan nilainya akan sesuai dengan entri nilai Q-Table. (blok kode 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Di sini kita juga mendefinisikan fungsi `qvalues()`, yang mengembalikan daftar nilai Q-Table untuk state tertentu yang sesuai dengan semua aksi yang mungkin. Jika entri tidak ada di Q-Table, kita akan mengembalikan 0 sebagai default.

## Mari mulai Q-Learning

Sekarang kita siap mengajarkan Peter untuk menjaga keseimbangan!

1. Pertama, mari kita tetapkan beberapa hyperparameter: (blok kode 10)

    ```python
    # hiperparameter
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Di sini, `alpha` adalah **laju pembelajaran** yang menentukan sejauh mana kita harus menyesuaikan nilai Q-Table saat ini di setiap langkah. Dalam pelajaran sebelumnya kita mulai dengan 1, kemudian menurunkan `alpha` ke nilai lebih rendah selama pelatihan. Dalam contoh ini kita akan mempertahankannya konstan untuk kesederhanaan, dan kamu bisa bereksperimen dengan mengubah nilai `alpha` nanti.

    `gamma` adalah **faktor diskonto** yang menunjukkan sejauh mana kita harus memprioritaskan reward masa depan dibandingkan reward saat ini.

    `epsilon` adalah **faktor eksplorasi/eksploitasi** yang menentukan apakah kita harus lebih memilih eksplorasi daripada eksploitasi atau sebaliknya. Dalam algoritma kita, `epsilon` persen dari kasus akan memilih aksi berikutnya sesuai nilai Q-Table, dan sisanya akan menjalankan aksi acak. Ini memungkinkan kita menjelajahi area ruang pencarian yang belum pernah kita lihat sebelumnya.

    ✅ Dalam hal menjaga keseimbangan - memilih tindakan acak (eksplorasi) akan bertindak seperti pukulan acak ke arah yang salah, dan tiang harus belajar cara mengembalikan keseimbangan dari "kesalahan" tersebut.

### Memperbaiki Algoritma

Kita juga dapat membuat dua perbaikan pada algoritma dari pelajaran sebelumnya:

- **Menghitung rata-rata reward kumulatif**, selama sejumlah simulasi. Kita akan mencetak kemajuan setiap 5000 iterasi, dan kita akan mengambil rata-rata reward kumulatif selama periode waktu tersebut. Ini berarti jika kita mendapatkan lebih dari 195 poin - kita dapat menganggap masalah berhasil diselesaikan, dengan kualitas lebih tinggi dari yang diperlukan.
  
- **Menghitung maksimum rata-rata hasil kumulatif**, `Qmax`, dan kita akan menyimpan Q-Table yang sesuai dengan hasil tersebut. Saat kamu menjalankan pelatihan, kamu akan melihat terkadang hasil rata-ratanya mulai turun, dan kita ingin menyimpan nilai Q-Table yang sesuai dengan model terbaik yang diamati selama pelatihan.

1. Kumpulkan semua reward kumulatif pada setiap simulasi dalam vektor `rewards` untuk plotting lebih lanjut. (blok kode  11)

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
                # eksploitasi - memilih aksi sesuai dengan probabilitas Q-Table
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # eksplorasi - memilih aksi secara acak
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Secara berkala cetak hasil dan hitung rata-rata hadiah ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Apa yang mungkin kamu perhatikan dari hasil tersebut:

- **Dekat dengan tujuan kita**. Kita sangat dekat dengan mencapai tujuan mendapatkan 195 reward kumulatif selama 100+ percobaan berturut-turut, atau mungkin kita sudah mencapainya! Bahkan jika nilainya lebih kecil, kita belum tahu pasti, karena kita mengambil rata-rata selama 5000 percobaan, dan hanya 100 percobaan yang diperlukan dalam kriteria formal.
  
- **Reward mulai turun**. Terkadang reward mulai turun, yang berarti kita dapat "menghancurkan" nilai yang telah dipelajari dalam Q-Table dengan nilai yang membuat situasi menjadi lebih buruk.

Pengamatan ini lebih jelas jika kita plot kemajuan pelatihan.

## Memplot Kemajuan Pelatihan

Selama pelatihan, kita mengumpulkan nilai reward kumulatif pada setiap iterasi di vektor `rewards`. Berikut adalah bagaimana tampilannya saat kita plot terhadap nomor iterasi:

```python
plt.plot(rewards)
```

![raw  progress](../../../../translated_images/id/train_progress_raw.2adfdf2daea09c59.webp)

Dari grafik ini, tidak mungkin menyimpulkan apa pun, karena sifat proses pelatihan stokastik, durasi sesi pelatihan sangat bervariasi. Untuk lebih memahami grafik ini, kita dapat menghitung **rata-rata berjalan** selama serangkaian eksperimen, misalnya 100. Ini dapat dilakukan dengan mudah menggunakan `np.convolve`: (blok kode 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![training progress](../../../../translated_images/id/train_progress_runav.c71694a8fa9ab359.webp)

## Mengubah Hyperparameter

Untuk membuat pembelajaran lebih stabil, masuk akal mengubah beberapa hyperparameter selama pelatihan. Secara khusus:

- **Untuk laju pembelajaran**, `alpha`, kita dapat mulai dengan nilai mendekati 1, lalu terus menurunkan parameter. Seiring waktu, kita akan mendapatkan nilai probabilitas bagus dalam Q-Table, sehingga harus menyesuaikannya sedikit, dan tidak menimpa sepenuhnya dengan nilai baru.

- **Meningkatkan epsilon**. Kita mungkin ingin meningkatkan `epsilon` secara perlahan, untuk mengurangi eksplorasi dan meningkatkan eksploitasi. Mungkin masuk akal mulai dengan nilai epsilon rendah, lalu naikkan mendekati 1.

> **Tugas 1**: Bermain dengan nilai hyperparameter dan lihat apakah kamu bisa mencapai reward kumulatif lebih tinggi. Apakah kamu mendapat lebih dari 195?


> **Tugas 2**: Untuk menyelesaikan masalah secara formal, Anda perlu mendapatkan rata-rata hadiah 195 selama 100 kali percobaan berturut-turut. Ukur itu selama pelatihan dan pastikan bahwa Anda benar-benar telah menyelesaikan masalah!

## Melihat hasil dalam aksi

Akan menarik untuk benar-benar melihat bagaimana perilaku model yang sudah dilatih. Mari kita jalankan simulasi dan ikuti strategi pemilihan aksi yang sama seperti saat pelatihan, dengan sampling sesuai distribusi probabilitas di Q-Table: (blok kode 13)

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

## 🚀Tantangan

> **Tugas 3**: Di sini, kami menggunakan salinan terakhir dari Q-Table, yang mungkin bukan yang terbaik. Ingat bahwa kami telah menyimpan Q-Table dengan performa terbaik ke dalam variabel `Qbest`! Cobalah contoh yang sama dengan Q-Table berperforma terbaik dengan menyalin `Qbest` ke `Q` dan lihat apakah Anda memperhatikan perbedaannya.

> **Tugas 4**: Di sini kami tidak memilih aksi terbaik pada setiap langkah, tapi sampling sesuai distribusi probabilitas yang sesuai. Apakah lebih masuk akal untuk selalu memilih aksi terbaik, dengan nilai Q-Table tertinggi? Ini bisa dilakukan dengan menggunakan fungsi `np.argmax` untuk mencari nomor aksi yang sesuai dengan nilai Q-Table tertinggi. Implementasikan strategi ini dan lihat apakah itu meningkatkan keseimbangan.

## [Kuis pasca kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tugas
[Latih sebuah Mountain Car](assignment.md)

## Kesimpulan

Sekarang kita telah belajar bagaimana melatih agen untuk mencapai hasil yang baik hanya dengan memberikan fungsi hadiah yang mendefinisikan keadaan permainan yang diinginkan, dan dengan memberi mereka kesempatan untuk menjelajahi ruang pencarian secara cerdas. Kita telah berhasil menerapkan algoritma Q-Learning dalam kasus lingkungan diskrit dan kontinu, tetapi dengan aksi diskrit.

Penting juga mempelajari situasi dimana ruang aksi juga kontinu, dan ketika ruang observasi jauh lebih kompleks, seperti gambar dari layar permainan Atari. Dalam masalah tersebut kita sering perlu menggunakan teknik pembelajaran mesin yang lebih kuat, seperti jaringan saraf, untuk mencapai hasil yang baik. Topik yang lebih lanjut tersebut adalah subjek dari kursus AI lanjutan kami yang akan datang.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan layanan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berupaya untuk mencapai akurasi, harap diketahui bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang sah. Untuk informasi penting, disarankan menggunakan terjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->