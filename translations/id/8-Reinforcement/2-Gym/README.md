<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T20:21:26+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "id"
}
-->
## Prasyarat

Dalam pelajaran ini, kita akan menggunakan pustaka bernama **OpenAI Gym** untuk mensimulasikan berbagai **lingkungan**. Anda dapat menjalankan kode pelajaran ini secara lokal (misalnya dari Visual Studio Code), di mana simulasi akan terbuka di jendela baru. Saat menjalankan kode secara online, Anda mungkin perlu melakukan beberapa penyesuaian pada kode, seperti yang dijelaskan [di sini](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Dalam pelajaran sebelumnya, aturan permainan dan keadaan diberikan oleh kelas `Board` yang kita definisikan sendiri. Di sini kita akan menggunakan **lingkungan simulasi** khusus, yang akan mensimulasikan fisika di balik tiang yang seimbang. Salah satu lingkungan simulasi paling populer untuk melatih algoritma pembelajaran penguatan disebut [Gym](https://gym.openai.com/), yang dikelola oleh [OpenAI](https://openai.com/). Dengan menggunakan gym ini, kita dapat membuat berbagai **lingkungan** mulai dari simulasi cartpole hingga permainan Atari.

> **Catatan**: Anda dapat melihat lingkungan lain yang tersedia dari OpenAI Gym [di sini](https://gym.openai.com/envs/#classic_control).

Pertama, mari kita instal gym dan impor pustaka yang diperlukan (kode blok 1):

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

- **Observation space** yang mendefinisikan struktur informasi yang kita terima dari lingkungan. Untuk masalah cartpole, kita menerima posisi tiang, kecepatan, dan beberapa nilai lainnya.

- **Action space** yang mendefinisikan tindakan yang mungkin dilakukan. Dalam kasus kita, action space bersifat diskrit, dan terdiri dari dua tindakan - **kiri** dan **kanan**. (kode blok 2)

1. Untuk menginisialisasi, ketik kode berikut:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Untuk melihat bagaimana lingkungan bekerja, mari kita jalankan simulasi singkat selama 100 langkah. Pada setiap langkah, kita memberikan salah satu tindakan yang akan dilakukan - dalam simulasi ini kita hanya memilih tindakan secara acak dari `action_space`.

1. Jalankan kode di bawah ini dan lihat hasilnya.

    ✅ Ingatlah bahwa lebih disarankan untuk menjalankan kode ini pada instalasi Python lokal! (kode blok 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Anda seharusnya melihat sesuatu yang mirip dengan gambar ini:

    ![cartpole tanpa keseimbangan](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Selama simulasi, kita perlu mendapatkan observasi untuk memutuskan tindakan apa yang harus dilakukan. Faktanya, fungsi langkah mengembalikan observasi saat ini, fungsi reward, dan flag selesai yang menunjukkan apakah simulasi masih perlu dilanjutkan atau tidak: (kode blok 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Anda akan melihat sesuatu seperti ini di output notebook:

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
    - Posisi kereta
    - Kecepatan kereta
    - Sudut tiang
    - Laju rotasi tiang

1. Dapatkan nilai minimum dan maksimum dari angka-angka tersebut: (kode blok 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Anda mungkin juga memperhatikan bahwa nilai reward pada setiap langkah simulasi selalu 1. Hal ini karena tujuan kita adalah bertahan selama mungkin, yaitu menjaga tiang tetap dalam posisi vertikal selama mungkin.

    ✅ Faktanya, simulasi CartPole dianggap berhasil jika kita berhasil mendapatkan rata-rata reward sebesar 195 selama 100 percobaan berturut-turut.

## Diskretisasi State

Dalam Q-Learning, kita perlu membangun Q-Table yang mendefinisikan tindakan apa yang harus dilakukan pada setiap state. Untuk dapat melakukan ini, kita memerlukan state yang **diskrit**, lebih tepatnya, state tersebut harus berisi sejumlah nilai diskrit yang terbatas. Oleh karena itu, kita perlu **mendiskretkan** observasi kita, memetakannya ke dalam kumpulan state yang terbatas.

Ada beberapa cara untuk melakukan ini:

- **Membagi menjadi beberapa bin**. Jika kita mengetahui interval dari suatu nilai tertentu, kita dapat membagi interval ini menjadi sejumlah **bin**, dan kemudian mengganti nilai dengan nomor bin tempat nilai tersebut berada. Hal ini dapat dilakukan menggunakan metode numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Dalam kasus ini, kita akan mengetahui ukuran state secara tepat, karena ukuran tersebut akan bergantung pada jumlah bin yang kita pilih untuk digitalisasi.

✅ Kita dapat menggunakan interpolasi linier untuk membawa nilai ke beberapa interval terbatas (misalnya, dari -20 hingga 20), dan kemudian mengonversi angka menjadi bilangan bulat dengan membulatkannya. Ini memberikan kita sedikit kontrol pada ukuran state, terutama jika kita tidak mengetahui rentang nilai input secara pasti. Misalnya, dalam kasus kita, 2 dari 4 nilai tidak memiliki batas atas/bawah pada nilainya, yang dapat menghasilkan jumlah state yang tak terbatas.

Dalam contoh kita, kita akan menggunakan pendekatan kedua. Seperti yang mungkin Anda perhatikan nanti, meskipun batas atas/bawah tidak terdefinisi, nilai-nilai tersebut jarang mengambil nilai di luar interval tertentu, sehingga state dengan nilai ekstrem akan sangat jarang.

1. Berikut adalah fungsi yang akan mengambil observasi dari model kita dan menghasilkan tuple dari 4 nilai bilangan bulat: (kode blok 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Mari kita juga eksplorasi metode diskretisasi lain menggunakan bin: (kode blok 7)

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

1. Sekarang mari kita jalankan simulasi singkat dan amati nilai-nilai lingkungan diskrit tersebut. Silakan coba `discretize` dan `discretize_bins` dan lihat apakah ada perbedaan.

    ✅ discretize_bins mengembalikan nomor bin, yang berbasis 0. Jadi untuk nilai variabel input di sekitar 0, ia mengembalikan nomor dari tengah interval (10). Dalam discretize, kita tidak peduli dengan rentang nilai output, memungkinkan mereka menjadi negatif, sehingga nilai state tidak bergeser, dan 0 sesuai dengan 0. (kode blok 8)

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

    ✅ Hapus komentar pada baris yang dimulai dengan env.render jika Anda ingin melihat bagaimana lingkungan dieksekusi. Jika tidak, Anda dapat menjalankannya di latar belakang, yang lebih cepat. Kami akan menggunakan eksekusi "tak terlihat" ini selama proses Q-Learning kami.

## Struktur Q-Table

Dalam pelajaran sebelumnya, state adalah pasangan angka sederhana dari 0 hingga 8, sehingga nyaman untuk merepresentasikan Q-Table dengan tensor numpy dengan bentuk 8x8x2. Jika kita menggunakan diskretisasi bin, ukuran vektor state kita juga diketahui, sehingga kita dapat menggunakan pendekatan yang sama dan merepresentasikan state dengan array berbentuk 20x20x10x10x2 (di sini 2 adalah dimensi action space, dan dimensi pertama sesuai dengan jumlah bin yang kita pilih untuk setiap parameter dalam observation space).

Namun, terkadang dimensi observation space tidak diketahui secara pasti. Dalam kasus fungsi `discretize`, kita mungkin tidak pernah yakin bahwa state kita tetap berada dalam batas tertentu, karena beberapa nilai asli tidak memiliki batas. Oleh karena itu, kita akan menggunakan pendekatan yang sedikit berbeda dan merepresentasikan Q-Table dengan sebuah dictionary.

1. Gunakan pasangan *(state,action)* sebagai kunci dictionary, dan nilainya akan sesuai dengan nilai entri Q-Table. (kode blok 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Di sini kita juga mendefinisikan fungsi `qvalues()`, yang mengembalikan daftar nilai Q-Table untuk state tertentu yang sesuai dengan semua tindakan yang mungkin. Jika entri tidak ada dalam Q-Table, kita akan mengembalikan 0 sebagai default.

## Mari Mulai Q-Learning

Sekarang kita siap mengajari Peter untuk menjaga keseimbangan!

1. Pertama, mari kita tetapkan beberapa hyperparameter: (kode blok 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Di sini, `alpha` adalah **learning rate** yang menentukan sejauh mana kita harus menyesuaikan nilai Q-Table saat ini pada setiap langkah. Dalam pelajaran sebelumnya, kita memulai dengan 1, dan kemudian menurunkan `alpha` ke nilai yang lebih rendah selama pelatihan. Dalam contoh ini, kita akan mempertahankannya tetap konstan demi kesederhanaan, dan Anda dapat bereksperimen dengan menyesuaikan nilai `alpha` nanti.

    `gamma` adalah **discount factor** yang menunjukkan sejauh mana kita harus memprioritaskan reward di masa depan dibandingkan reward saat ini.

    `epsilon` adalah **exploration/exploitation factor** yang menentukan apakah kita harus lebih memilih eksplorasi daripada eksploitasi atau sebaliknya. Dalam algoritma kita, kita akan memilih tindakan berikutnya sesuai dengan nilai Q-Table dalam persentase `epsilon` dari kasus, dan dalam jumlah kasus yang tersisa kita akan melakukan tindakan acak. Ini memungkinkan kita untuk menjelajahi area ruang pencarian yang belum pernah kita lihat sebelumnya.

    ✅ Dalam hal keseimbangan - memilih tindakan acak (eksplorasi) akan bertindak sebagai dorongan acak ke arah yang salah, dan tiang harus belajar bagaimana memulihkan keseimbangan dari "kesalahan" tersebut.

### Tingkatkan Algoritma

Kita juga dapat membuat dua peningkatan pada algoritma kita dari pelajaran sebelumnya:

- **Hitung rata-rata reward kumulatif**, selama sejumlah simulasi. Kita akan mencetak kemajuan setiap 5000 iterasi, dan kita akan merata-ratakan reward kumulatif kita selama periode waktu tersebut. Artinya, jika kita mendapatkan lebih dari 195 poin - kita dapat menganggap masalah telah terpecahkan, bahkan dengan kualitas yang lebih tinggi dari yang diperlukan.

- **Hitung hasil kumulatif rata-rata maksimum**, `Qmax`, dan kita akan menyimpan Q-Table yang sesuai dengan hasil tersebut. Saat Anda menjalankan pelatihan, Anda akan melihat bahwa terkadang hasil kumulatif rata-rata mulai menurun, dan kita ingin menyimpan nilai Q-Table yang sesuai dengan model terbaik yang diamati selama pelatihan.

1. Kumpulkan semua reward kumulatif pada setiap simulasi di vektor `rewards` untuk plotting lebih lanjut. (kode blok 11)

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

Apa yang mungkin Anda perhatikan dari hasil tersebut:

- **Dekat dengan tujuan kita**. Kita sangat dekat dengan mencapai tujuan mendapatkan reward kumulatif sebesar 195 selama 100+ percobaan simulasi berturut-turut, atau kita mungkin telah benar-benar mencapainya! Bahkan jika kita mendapatkan angka yang lebih kecil, kita masih tidak tahu, karena kita merata-ratakan selama 5000 kali, dan hanya 100 kali yang diperlukan dalam kriteria formal.

- **Reward mulai menurun**. Terkadang reward mulai menurun, yang berarti kita dapat "merusak" nilai yang sudah dipelajari dalam Q-Table dengan nilai yang membuat situasi menjadi lebih buruk.

Pengamatan ini lebih jelas terlihat jika kita memplot kemajuan pelatihan.

## Memplot Kemajuan Pelatihan

Selama pelatihan, kita telah mengumpulkan nilai reward kumulatif pada setiap iterasi ke dalam vektor `rewards`. Berikut adalah tampilannya saat kita plot terhadap nomor iterasi:

```python
plt.plot(rewards)
```

![kemajuan mentah](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Dari grafik ini, tidak mungkin untuk menyimpulkan apa pun, karena sifat proses pelatihan stokastik membuat panjang sesi pelatihan sangat bervariasi. Untuk membuat grafik ini lebih masuk akal, kita dapat menghitung **rata-rata berjalan** selama serangkaian eksperimen, misalnya 100. Hal ini dapat dilakukan dengan mudah menggunakan `np.convolve`: (kode blok 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![kemajuan pelatihan](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Memvariasikan Hyperparameter

Untuk membuat pembelajaran lebih stabil, masuk akal untuk menyesuaikan beberapa hyperparameter kita selama pelatihan. Secara khusus:

- **Untuk learning rate**, `alpha`, kita dapat memulai dengan nilai yang mendekati 1, dan kemudian terus menurunkan parameter tersebut. Seiring waktu, kita akan mendapatkan nilai probabilitas yang baik dalam Q-Table, sehingga kita harus menyesuaikannya sedikit, dan tidak menimpa sepenuhnya dengan nilai baru.

- **Tingkatkan epsilon**. Kita mungkin ingin meningkatkan `epsilon` secara perlahan, agar lebih sedikit eksplorasi dan lebih banyak eksploitasi. Mungkin masuk akal untuk memulai dengan nilai `epsilon` yang lebih rendah, dan meningkatkannya hingga hampir 1.
> **Tugas 1**: Coba ubah nilai hyperparameter dan lihat apakah Anda bisa mencapai total reward yang lebih tinggi. Apakah Anda mendapatkan di atas 195?
> **Tugas 2**: Untuk secara formal menyelesaikan masalah ini, Anda perlu mencapai rata-rata reward sebesar 195 dalam 100 kali percobaan berturut-turut. Ukur itu selama pelatihan dan pastikan bahwa Anda telah secara formal menyelesaikan masalah ini!

## Melihat hasilnya secara langsung

Akan menarik untuk melihat bagaimana model yang telah dilatih berperilaku. Mari kita jalankan simulasi dan gunakan strategi pemilihan aksi yang sama seperti saat pelatihan, yaitu sampling berdasarkan distribusi probabilitas di Q-Table: (blok kode 13)

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

Anda seharusnya melihat sesuatu seperti ini:

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Tantangan

> **Tugas 3**: Di sini, kita menggunakan salinan akhir dari Q-Table, yang mungkin bukan yang terbaik. Ingat bahwa kita telah menyimpan Q-Table dengan performa terbaik ke dalam variabel `Qbest`! Coba contoh yang sama dengan Q-Table terbaik dengan menyalin `Qbest` ke `Q` dan lihat apakah Anda melihat perbedaannya.

> **Tugas 4**: Di sini kita tidak memilih aksi terbaik di setiap langkah, melainkan sampling berdasarkan distribusi probabilitas yang sesuai. Apakah lebih masuk akal untuk selalu memilih aksi terbaik, dengan nilai Q-Table tertinggi? Ini dapat dilakukan dengan menggunakan fungsi `np.argmax` untuk menemukan nomor aksi yang sesuai dengan nilai Q-Table tertinggi. Implementasikan strategi ini dan lihat apakah itu meningkatkan keseimbangan.

## [Kuis setelah kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tugas
[Latih Mountain Car](assignment.md)

## Kesimpulan

Kita sekarang telah belajar bagaimana melatih agen untuk mencapai hasil yang baik hanya dengan memberikan mereka fungsi reward yang mendefinisikan keadaan yang diinginkan dalam permainan, dan dengan memberikan mereka kesempatan untuk secara cerdas menjelajahi ruang pencarian. Kita telah berhasil menerapkan algoritma Q-Learning dalam kasus lingkungan diskret dan kontinu, tetapi dengan aksi diskret.

Penting juga untuk mempelajari situasi di mana keadaan aksi juga kontinu, dan ketika ruang observasi jauh lebih kompleks, seperti gambar dari layar permainan Atari. Dalam masalah-masalah tersebut, kita sering kali perlu menggunakan teknik pembelajaran mesin yang lebih kuat, seperti jaringan saraf, untuk mencapai hasil yang baik. Topik-topik yang lebih maju ini akan menjadi subjek dari kursus AI lanjutan kami yang akan datang.

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.