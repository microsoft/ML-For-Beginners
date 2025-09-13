<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T20:14:51+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "id"
}
-->
# Pengantar Pembelajaran Penguatan dan Q-Learning

![Ringkasan pembelajaran penguatan dalam pembelajaran mesin dalam bentuk sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

Pembelajaran penguatan melibatkan tiga konsep penting: agen, beberapa keadaan, dan satu set tindakan untuk setiap keadaan. Dengan melakukan suatu tindakan dalam keadaan tertentu, agen akan mendapatkan hadiah. Bayangkan permainan komputer Super Mario. Anda adalah Mario, berada di level permainan, berdiri di tepi jurang. Di atas Anda ada sebuah koin. Anda sebagai Mario, berada di level permainan, di posisi tertentu ... itulah keadaan Anda. Melangkah satu langkah ke kanan (tindakan) akan membuat Anda jatuh ke jurang, dan itu akan memberikan skor numerik rendah. Namun, menekan tombol lompat akan membuat Anda mendapatkan poin dan tetap hidup. Itu adalah hasil positif dan seharusnya memberikan skor numerik positif.

Dengan menggunakan pembelajaran penguatan dan simulator (permainan), Anda dapat belajar cara memainkan permainan untuk memaksimalkan hadiah, yaitu tetap hidup dan mendapatkan sebanyak mungkin poin.

[![Pengantar Pembelajaran Penguatan](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klik gambar di atas untuk mendengar Dmitry membahas Pembelajaran Penguatan

## [Kuis sebelum pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Prasyarat dan Pengaturan

Dalam pelajaran ini, kita akan bereksperimen dengan beberapa kode dalam Python. Anda harus dapat menjalankan kode Jupyter Notebook dari pelajaran ini, baik di komputer Anda atau di cloud.

Anda dapat membuka [notebook pelajaran](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) dan mengikuti pelajaran ini untuk membangun.

> **Catatan:** Jika Anda membuka kode ini dari cloud, Anda juga perlu mengambil file [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), yang digunakan dalam kode notebook. Tambahkan file tersebut ke direktori yang sama dengan notebook.

## Pengantar

Dalam pelajaran ini, kita akan menjelajahi dunia **[Peter dan Serigala](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, terinspirasi oleh dongeng musikal karya komposer Rusia, [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Kita akan menggunakan **Pembelajaran Penguatan** untuk membiarkan Peter menjelajahi lingkungannya, mengumpulkan apel yang lezat, dan menghindari bertemu dengan serigala.

**Pembelajaran Penguatan** (RL) adalah teknik pembelajaran yang memungkinkan kita mempelajari perilaku optimal dari seorang **agen** dalam suatu **lingkungan** dengan menjalankan banyak eksperimen. Agen dalam lingkungan ini harus memiliki **tujuan**, yang didefinisikan oleh **fungsi hadiah**.

## Lingkungan

Untuk kesederhanaan, mari kita anggap dunia Peter sebagai papan persegi berukuran `lebar` x `tinggi`, seperti ini:

![Lingkungan Peter](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Setiap sel di papan ini dapat berupa:

* **tanah**, tempat Peter dan makhluk lain dapat berjalan.
* **air**, tempat Anda jelas tidak bisa berjalan.
* **pohon** atau **rumput**, tempat Anda bisa beristirahat.
* **apel**, yang mewakili sesuatu yang Peter senang temukan untuk makanannya.
* **serigala**, yang berbahaya dan harus dihindari.

Ada modul Python terpisah, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), yang berisi kode untuk bekerja dengan lingkungan ini. Karena kode ini tidak penting untuk memahami konsep kita, kita akan mengimpor modul tersebut dan menggunakannya untuk membuat papan contoh (kode blok 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Kode ini harus mencetak gambar lingkungan yang mirip dengan yang di atas.

## Tindakan dan kebijakan

Dalam contoh kita, tujuan Peter adalah menemukan apel, sambil menghindari serigala dan rintangan lainnya. Untuk melakukan ini, dia pada dasarnya dapat berjalan-jalan sampai menemukan apel.

Oleh karena itu, pada posisi mana pun, dia dapat memilih salah satu dari tindakan berikut: atas, bawah, kiri, dan kanan.

Kita akan mendefinisikan tindakan-tindakan tersebut sebagai kamus, dan memetakan mereka ke pasangan perubahan koordinat yang sesuai. Misalnya, bergerak ke kanan (`R`) akan sesuai dengan pasangan `(1,0)`. (kode blok 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Secara keseluruhan, strategi dan tujuan dari skenario ini adalah sebagai berikut:

- **Strategi**, dari agen kita (Peter) didefinisikan oleh apa yang disebut **kebijakan**. Kebijakan adalah fungsi yang mengembalikan tindakan pada keadaan tertentu. Dalam kasus kita, keadaan masalah diwakili oleh papan, termasuk posisi pemain saat ini.

- **Tujuan**, dari pembelajaran penguatan adalah akhirnya mempelajari kebijakan yang baik yang memungkinkan kita menyelesaikan masalah secara efisien. Namun, sebagai dasar, mari kita pertimbangkan kebijakan paling sederhana yang disebut **jalan acak**.

## Jalan acak

Mari kita selesaikan masalah kita terlebih dahulu dengan menerapkan strategi jalan acak. Dengan jalan acak, kita akan secara acak memilih tindakan berikutnya dari tindakan yang diizinkan, sampai kita mencapai apel (kode blok 3).

1. Terapkan jalan acak dengan kode di bawah ini:

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

    Panggilan ke `walk` harus mengembalikan panjang jalur yang sesuai, yang dapat bervariasi dari satu percobaan ke percobaan lainnya.

1. Jalankan eksperimen jalan beberapa kali (misalnya, 100 kali), dan cetak statistik yang dihasilkan (kode blok 4):

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

    Perhatikan bahwa rata-rata panjang jalur adalah sekitar 30-40 langkah, yang cukup banyak, mengingat jarak rata-rata ke apel terdekat adalah sekitar 5-6 langkah.

    Anda juga dapat melihat bagaimana gerakan Peter terlihat selama jalan acak:

    ![Jalan Acak Peter](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Fungsi hadiah

Untuk membuat kebijakan kita lebih cerdas, kita perlu memahami langkah mana yang "lebih baik" daripada yang lain. Untuk melakukan ini, kita perlu mendefinisikan tujuan kita.

Tujuan dapat didefinisikan dalam bentuk **fungsi hadiah**, yang akan mengembalikan nilai skor untuk setiap keadaan. Semakin tinggi angkanya, semakin baik fungsi hadiahnya. (kode blok 5)

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

Hal menarik tentang fungsi hadiah adalah bahwa dalam banyak kasus, *kita hanya diberikan hadiah yang substansial di akhir permainan*. Ini berarti algoritma kita harus mengingat langkah-langkah "baik" yang mengarah pada hadiah positif di akhir, dan meningkatkan kepentingannya. Demikian pula, semua langkah yang mengarah pada hasil buruk harus dihindari.

## Q-Learning

Algoritma yang akan kita bahas di sini disebut **Q-Learning**. Dalam algoritma ini, kebijakan didefinisikan oleh fungsi (atau struktur data) yang disebut **Q-Table**. Q-Table mencatat "kebaikan" dari setiap tindakan dalam keadaan tertentu.

Disebut Q-Table karena sering kali lebih nyaman untuk merepresentasikannya sebagai tabel, atau array multi-dimensi. Karena papan kita memiliki dimensi `lebar` x `tinggi`, kita dapat merepresentasikan Q-Table menggunakan array numpy dengan bentuk `lebar` x `tinggi` x `len(actions)`: (kode blok 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Perhatikan bahwa kita menginisialisasi semua nilai Q-Table dengan nilai yang sama, dalam kasus kita - 0.25. Ini sesuai dengan kebijakan "jalan acak", karena semua langkah dalam setiap keadaan sama baiknya. Kita dapat meneruskan Q-Table ke fungsi `plot` untuk memvisualisasikan tabel di papan: `m.plot(Q)`.

![Lingkungan Peter](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Di tengah setiap sel terdapat "panah" yang menunjukkan arah gerakan yang disukai. Karena semua arah sama, sebuah titik ditampilkan.

Sekarang kita perlu menjalankan simulasi, menjelajahi lingkungan kita, dan mempelajari distribusi nilai Q-Table yang lebih baik, yang akan memungkinkan kita menemukan jalur ke apel jauh lebih cepat.

## Inti dari Q-Learning: Persamaan Bellman

Setelah kita mulai bergerak, setiap tindakan akan memiliki hadiah yang sesuai, yaitu kita secara teoritis dapat memilih tindakan berikutnya berdasarkan hadiah langsung tertinggi. Namun, dalam sebagian besar keadaan, langkah tersebut tidak akan mencapai tujuan kita untuk mencapai apel, sehingga kita tidak dapat langsung memutuskan arah mana yang lebih baik.

> Ingatlah bahwa bukan hasil langsung yang penting, melainkan hasil akhir, yang akan kita peroleh di akhir simulasi.

Untuk memperhitungkan hadiah yang tertunda ini, kita perlu menggunakan prinsip **[pemrograman dinamis](https://en.wikipedia.org/wiki/Dynamic_programming)**, yang memungkinkan kita memikirkan masalah kita secara rekursif.

Misalkan kita sekarang berada di keadaan *s*, dan kita ingin bergerak ke keadaan berikutnya *s'*. Dengan melakukan itu, kita akan menerima hadiah langsung *r(s,a)*, yang didefinisikan oleh fungsi hadiah, ditambah beberapa hadiah di masa depan. Jika kita menganggap bahwa Q-Table kita secara akurat mencerminkan "daya tarik" dari setiap tindakan, maka pada keadaan *s'* kita akan memilih tindakan *a* yang sesuai dengan nilai maksimum *Q(s',a')*. Dengan demikian, hadiah masa depan terbaik yang bisa kita dapatkan pada keadaan *s* akan didefinisikan sebagai `max`

Pembelajaran dapat dirangkum sebagai berikut:

- **Rata-rata panjang jalur meningkat**. Pada awalnya, rata-rata panjang jalur meningkat. Hal ini kemungkinan disebabkan oleh fakta bahwa ketika kita tidak tahu apa-apa tentang lingkungan, kita cenderung terjebak di keadaan buruk, seperti air atau serigala. Saat kita belajar lebih banyak dan mulai menggunakan pengetahuan ini, kita dapat menjelajahi lingkungan lebih lama, tetapi kita masih belum tahu dengan baik di mana apel berada.

- **Panjang jalur menurun seiring dengan pembelajaran**. Setelah kita belajar cukup banyak, menjadi lebih mudah bagi agen untuk mencapai tujuan, dan panjang jalur mulai menurun. Namun, kita masih terbuka untuk eksplorasi, sehingga kita sering menyimpang dari jalur terbaik dan mencoba opsi baru, yang membuat jalur lebih panjang dari yang optimal.

- **Panjang meningkat secara tiba-tiba**. Apa yang juga kita amati pada grafik ini adalah bahwa pada suatu titik, panjang jalur meningkat secara tiba-tiba. Hal ini menunjukkan sifat stokastik dari proses tersebut, dan bahwa kita dapat pada suatu saat "merusak" koefisien Q-Table dengan menimpanya dengan nilai-nilai baru. Idealnya, hal ini harus diminimalkan dengan menurunkan tingkat pembelajaran (misalnya, menjelang akhir pelatihan, kita hanya menyesuaikan nilai Q-Table dengan nilai kecil).

Secara keseluruhan, penting untuk diingat bahwa keberhasilan dan kualitas proses pembelajaran sangat bergantung pada parameter seperti tingkat pembelajaran, penurunan tingkat pembelajaran, dan faktor diskon. Parameter-parameter ini sering disebut sebagai **hiperparameter**, untuk membedakannya dari **parameter**, yang kita optimalkan selama pelatihan (misalnya, koefisien Q-Table). Proses menemukan nilai hiperparameter terbaik disebut **optimasi hiperparameter**, dan topik ini layak untuk dibahas secara terpisah.

## [Kuis setelah kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tugas 
[Dunia yang Lebih Realistis](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diperhatikan bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan manusia profesional. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.