<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T20:16:08+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "ms"
}
-->
# Pengenalan kepada Pembelajaran Pengukuhan dan Q-Learning

![Ringkasan pembelajaran pengukuhan dalam pembelajaran mesin dalam bentuk sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

Pembelajaran pengukuhan melibatkan tiga konsep penting: agen, beberapa keadaan, dan satu set tindakan bagi setiap keadaan. Dengan melaksanakan tindakan dalam keadaan tertentu, agen akan diberikan ganjaran. Bayangkan permainan komputer Super Mario. Anda adalah Mario, berada dalam tahap permainan, berdiri di tepi tebing. Di atas anda terdapat syiling. Anda sebagai Mario, dalam tahap permainan, di kedudukan tertentu ... itulah keadaan anda. Melangkah satu langkah ke kanan (tindakan) akan membawa anda ke tebing, dan itu akan memberikan skor numerik yang rendah. Namun, menekan butang lompat akan membolehkan anda mendapat mata dan terus hidup. Itu adalah hasil yang positif dan sepatutnya memberikan skor numerik yang positif.

Dengan menggunakan pembelajaran pengukuhan dan simulator (permainan), anda boleh belajar cara bermain permainan untuk memaksimumkan ganjaran iaitu terus hidup dan mengumpul sebanyak mungkin mata.

[![Pengenalan kepada Pembelajaran Pengukuhan](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klik imej di atas untuk mendengar Dmitry membincangkan Pembelajaran Pengukuhan

## [Kuiz pra-kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Prasyarat dan Persediaan

Dalam pelajaran ini, kita akan bereksperimen dengan beberapa kod dalam Python. Anda sepatutnya boleh menjalankan kod Jupyter Notebook daripada pelajaran ini, sama ada di komputer anda atau di awan.

Anda boleh membuka [notebook pelajaran](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) dan mengikuti pelajaran ini untuk membina.

> **Nota:** Jika anda membuka kod ini dari awan, anda juga perlu mendapatkan fail [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), yang digunakan dalam kod notebook. Tambahkan fail ini ke direktori yang sama dengan notebook.

## Pengenalan

Dalam pelajaran ini, kita akan meneroka dunia **[Peter dan Serigala](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, yang diilhamkan oleh kisah dongeng muzik oleh komposer Rusia, [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Kita akan menggunakan **Pembelajaran Pengukuhan** untuk membolehkan Peter meneroka persekitarannya, mengumpul epal yang lazat dan mengelakkan bertemu dengan serigala.

**Pembelajaran Pengukuhan** (RL) adalah teknik pembelajaran yang membolehkan kita mempelajari tingkah laku optimum bagi **agen** dalam **persekitaran** tertentu dengan menjalankan banyak eksperimen. Agen dalam persekitaran ini sepatutnya mempunyai **matlamat**, yang ditakrifkan oleh **fungsi ganjaran**.

## Persekitaran

Untuk kesederhanaan, mari kita anggap dunia Peter sebagai papan segi empat dengan saiz `width` x `height`, seperti ini:

![Persekitaran Peter](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Setiap sel dalam papan ini boleh menjadi:

* **tanah**, di mana Peter dan makhluk lain boleh berjalan.
* **air**, di mana anda jelas tidak boleh berjalan.
* **pokok** atau **rumput**, tempat di mana anda boleh berehat.
* **epal**, yang mewakili sesuatu yang Peter akan gembira untuk menemui untuk makan.
* **serigala**, yang berbahaya dan harus dielakkan.

Terdapat modul Python berasingan, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), yang mengandungi kod untuk bekerja dengan persekitaran ini. Oleh kerana kod ini tidak penting untuk memahami konsep kita, kita akan mengimport modul dan menggunakannya untuk mencipta papan contoh (blok kod 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Kod ini sepatutnya mencetak gambar persekitaran yang serupa dengan yang di atas.

## Tindakan dan polisi

Dalam contoh kita, matlamat Peter adalah untuk mencari epal, sambil mengelakkan serigala dan halangan lain. Untuk melakukan ini, dia boleh berjalan-jalan sehingga dia menemui epal.

Oleh itu, pada mana-mana kedudukan, dia boleh memilih antara salah satu tindakan berikut: atas, bawah, kiri dan kanan.

Kita akan mentakrifkan tindakan tersebut sebagai kamus, dan memetakan mereka kepada pasangan perubahan koordinat yang sepadan. Sebagai contoh, bergerak ke kanan (`R`) akan sepadan dengan pasangan `(1,0)`. (blok kod 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Secara ringkas, strategi dan matlamat senario ini adalah seperti berikut:

- **Strategi**, agen kita (Peter) ditakrifkan oleh apa yang dipanggil **polisi**. Polisi adalah fungsi yang mengembalikan tindakan pada mana-mana keadaan tertentu. Dalam kes kita, keadaan masalah diwakili oleh papan, termasuk kedudukan semasa pemain.

- **Matlamat**, pembelajaran pengukuhan adalah untuk akhirnya mempelajari polisi yang baik yang akan membolehkan kita menyelesaikan masalah dengan cekap. Walau bagaimanapun, sebagai asas, mari kita pertimbangkan polisi paling mudah yang dipanggil **jalan rawak**.

## Jalan rawak

Mari kita selesaikan masalah kita terlebih dahulu dengan melaksanakan strategi jalan rawak. Dengan jalan rawak, kita akan memilih tindakan seterusnya secara rawak daripada tindakan yang dibenarkan, sehingga kita mencapai epal (blok kod 3).

1. Laksanakan jalan rawak dengan kod di bawah:

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

    Panggilan kepada `walk` sepatutnya mengembalikan panjang laluan yang sepadan, yang boleh berbeza dari satu larian ke larian lain. 

1. Jalankan eksperimen jalan beberapa kali (katakan, 100), dan cetak statistik yang dihasilkan (blok kod 4):

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

    Perhatikan bahawa panjang purata laluan adalah sekitar 30-40 langkah, yang agak banyak, memandangkan jarak purata ke epal terdekat adalah sekitar 5-6 langkah.

    Anda juga boleh melihat bagaimana pergerakan Peter kelihatan semasa jalan rawak:

    ![Jalan Rawak Peter](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Fungsi ganjaran

Untuk menjadikan polisi kita lebih pintar, kita perlu memahami langkah mana yang "lebih baik" daripada yang lain. Untuk melakukan ini, kita perlu mentakrifkan matlamat kita.

Matlamat boleh ditakrifkan dalam bentuk **fungsi ganjaran**, yang akan mengembalikan beberapa nilai skor untuk setiap keadaan. Semakin tinggi nombor, semakin baik fungsi ganjaran. (blok kod 5)

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

Perkara menarik tentang fungsi ganjaran ialah dalam kebanyakan kes, *kita hanya diberikan ganjaran yang besar pada akhir permainan*. Ini bermakna algoritma kita sepatutnya mengingati langkah "baik" yang membawa kepada ganjaran positif pada akhirnya, dan meningkatkan kepentingannya. Begitu juga, semua langkah yang membawa kepada hasil buruk harus dielakkan.

## Q-Learning

Algoritma yang akan kita bincangkan di sini dipanggil **Q-Learning**. Dalam algoritma ini, polisi ditakrifkan oleh fungsi (atau struktur data) yang dipanggil **Q-Table**. Ia merekodkan "kebaikan" setiap tindakan dalam keadaan tertentu.

Ia dipanggil Q-Table kerana ia sering mudah untuk mewakilinya sebagai jadual, atau array multi-dimensi. Oleh kerana papan kita mempunyai dimensi `width` x `height`, kita boleh mewakili Q-Table menggunakan array numpy dengan bentuk `width` x `height` x `len(actions)`: (blok kod 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Perhatikan bahawa kita memulakan semua nilai Q-Table dengan nilai yang sama, dalam kes kita - 0.25. Ini sepadan dengan polisi "jalan rawak", kerana semua langkah dalam setiap keadaan adalah sama baik. Kita boleh menghantar Q-Table kepada fungsi `plot` untuk memvisualisasikan jadual pada papan: `m.plot(Q)`.

![Persekitaran Peter](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Di tengah-tengah setiap sel terdapat "anak panah" yang menunjukkan arah pergerakan yang disukai. Oleh kerana semua arah adalah sama, titik ditunjukkan.

Sekarang kita perlu menjalankan simulasi, meneroka persekitaran kita, dan mempelajari pengagihan nilai Q-Table yang lebih baik, yang akan membolehkan kita mencari laluan ke epal dengan lebih cepat.

## Intipati Q-Learning: Persamaan Bellman

Sebaik sahaja kita mula bergerak, setiap tindakan akan mempunyai ganjaran yang sepadan, iaitu kita secara teorinya boleh memilih tindakan seterusnya berdasarkan ganjaran segera yang tertinggi. Walau bagaimanapun, dalam kebanyakan keadaan, langkah tersebut tidak akan mencapai matlamat kita untuk mencapai epal, dan oleh itu kita tidak boleh segera memutuskan arah mana yang lebih baik.

> Ingat bahawa bukan hasil segera yang penting, tetapi hasil akhir, yang akan kita peroleh pada akhir simulasi.

Untuk mengambil kira ganjaran yang tertunda ini, kita perlu menggunakan prinsip **[pengaturcaraan dinamik](https://en.wikipedia.org/wiki/Dynamic_programming)**, yang membolehkan kita memikirkan masalah kita secara rekursif.

Katakan kita kini berada di keadaan *s*, dan kita ingin bergerak ke keadaan seterusnya *s'*. Dengan berbuat demikian, kita akan menerima ganjaran segera *r(s,a)*, yang ditakrifkan oleh fungsi ganjaran, ditambah beberapa ganjaran masa depan. Jika kita mengandaikan bahawa Q-Table kita dengan betul mencerminkan "daya tarikan" setiap tindakan, maka di keadaan *s'* kita akan memilih tindakan *a* yang sepadan dengan nilai maksimum *Q(s',a')*. Oleh itu, ganjaran masa depan terbaik yang boleh kita peroleh di keadaan *s* akan ditakrifkan sebagai `max`

## Memeriksa polisi

Oleh kerana Q-Table menyenaraikan "daya tarikan" setiap tindakan di setiap keadaan, ia agak mudah untuk menggunakannya bagi menentukan navigasi yang efisien dalam dunia kita. Dalam kes yang paling mudah, kita boleh memilih tindakan yang sepadan dengan nilai Q-Table tertinggi: (blok kod 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Jika anda mencuba kod di atas beberapa kali, anda mungkin perasan bahawa kadangkala ia "tergantung", dan anda perlu menekan butang STOP dalam notebook untuk menghentikannya. Ini berlaku kerana mungkin terdapat situasi di mana dua keadaan "menunjuk" antara satu sama lain dari segi nilai Q yang optimum, menyebabkan agen bergerak antara keadaan tersebut tanpa henti.

## 🚀Cabaran

> **Tugas 1:** Ubah fungsi `walk` untuk menghadkan panjang maksimum laluan kepada bilangan langkah tertentu (contohnya, 100), dan lihat kod di atas mengembalikan nilai ini dari semasa ke semasa.

> **Tugas 2:** Ubah fungsi `walk` supaya ia tidak kembali ke tempat yang telah dilalui sebelumnya. Ini akan menghalang `walk` daripada berulang, namun, agen masih boleh terperangkap di lokasi yang tidak dapat dilepaskan.

## Navigasi

Polisi navigasi yang lebih baik adalah yang kita gunakan semasa latihan, yang menggabungkan eksploitasi dan eksplorasi. Dalam polisi ini, kita akan memilih setiap tindakan dengan kebarangkalian tertentu, berkadar dengan nilai dalam Q-Table. Strategi ini mungkin masih menyebabkan agen kembali ke posisi yang telah diterokai, tetapi, seperti yang anda lihat dari kod di bawah, ia menghasilkan laluan purata yang sangat pendek ke lokasi yang diinginkan (ingat bahawa `print_statistics` menjalankan simulasi sebanyak 100 kali): (blok kod 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Selepas menjalankan kod ini, anda sepatutnya mendapat panjang laluan purata yang jauh lebih kecil daripada sebelumnya, dalam lingkungan 3-6.

## Menyelidik proses pembelajaran

Seperti yang telah disebutkan, proses pembelajaran adalah keseimbangan antara eksplorasi dan eksploitasi pengetahuan yang diperoleh tentang struktur ruang masalah. Kita telah melihat bahawa hasil pembelajaran (keupayaan untuk membantu agen mencari laluan pendek ke matlamat) telah bertambah baik, tetapi ia juga menarik untuk memerhatikan bagaimana panjang laluan purata berubah semasa proses pembelajaran:

Pembelajaran boleh diringkaskan seperti berikut:

- **Panjang laluan purata meningkat**. Apa yang kita lihat di sini adalah pada mulanya, panjang laluan purata meningkat. Ini mungkin disebabkan oleh fakta bahawa apabila kita tidak tahu apa-apa tentang persekitaran, kita cenderung terperangkap dalam keadaan buruk, seperti air atau serigala. Apabila kita belajar lebih banyak dan mula menggunakan pengetahuan ini, kita boleh meneroka persekitaran lebih lama, tetapi kita masih tidak tahu di mana lokasi epal dengan baik.

- **Panjang laluan berkurang, apabila kita belajar lebih banyak**. Setelah kita belajar cukup, menjadi lebih mudah bagi agen untuk mencapai matlamat, dan panjang laluan mula berkurang. Walau bagaimanapun, kita masih terbuka kepada eksplorasi, jadi kita sering menyimpang dari laluan terbaik dan meneroka pilihan baru, menjadikan laluan lebih panjang daripada yang optimum.

- **Panjang meningkat secara mendadak**. Apa yang kita juga perhatikan pada graf ini adalah pada satu ketika, panjang meningkat secara mendadak. Ini menunjukkan sifat stokastik proses tersebut, dan bahawa kita boleh pada satu ketika "merosakkan" koefisien Q-Table dengan menulis semula mereka dengan nilai baru. Ini sebaiknya diminimumkan dengan mengurangkan kadar pembelajaran (contohnya, menjelang akhir latihan, kita hanya menyesuaikan nilai Q-Table dengan nilai kecil).

Secara keseluruhan, adalah penting untuk diingat bahawa kejayaan dan kualiti proses pembelajaran sangat bergantung pada parameter seperti kadar pembelajaran, pengurangan kadar pembelajaran, dan faktor diskaun. Parameter-parameter ini sering dipanggil **hiperparameter**, untuk membezakannya daripada **parameter**, yang kita optimalkan semasa latihan (contohnya, koefisien Q-Table). Proses mencari nilai hiperparameter terbaik dipanggil **pengoptimuman hiperparameter**, dan ia layak menjadi topik tersendiri.

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tugasan 
[Dunia yang Lebih Realistik](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.