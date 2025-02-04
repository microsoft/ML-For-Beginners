## Memeriksa polisi

Oleh kerana Jadual-Q menyenaraikan "tarikan" setiap tindakan pada setiap keadaan, adalah mudah untuk menggunakannya bagi menentukan navigasi yang cekap dalam dunia kita. Dalam kes yang paling mudah, kita boleh memilih tindakan yang sepadan dengan nilai Jadual-Q tertinggi: (blok kod 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Jika anda mencuba kod di atas beberapa kali, anda mungkin perasan bahawa kadang-kadang ia "tergantung", dan anda perlu menekan butang HENTI dalam buku nota untuk menghentikannya. Ini berlaku kerana mungkin terdapat situasi apabila dua keadaan "menunjuk" antara satu sama lain dari segi nilai Q yang optimum, di mana agen akhirnya bergerak antara keadaan tersebut tanpa henti.

## 🚀Cabaran

> **Tugas 1:** Ubah suai `walk` function to limit the maximum length of path by a certain number of steps (say, 100), and watch the code above return this value from time to time.

> **Task 2:** Modify the `walk` function so that it does not go back to the places where it has already been previously. This will prevent `walk` from looping, however, the agent can still end up being "trapped" in a location from which it is unable to escape.

## Navigation

A better navigation policy would be the one that we used during training, which combines exploitation and exploration. In this policy, we will select each action with a certain probability, proportional to the values in the Q-Table. This strategy may still result in the agent returning back to a position it has already explored, but, as you can see from the code below, it results in a very short average path to the desired location (remember that `print_statistics` menjalankan simulasi 100 kali): (blok kod 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Selepas menjalankan kod ini, anda sepatutnya mendapat panjang laluan purata yang lebih kecil daripada sebelumnya, dalam lingkungan 3-6.

## Menyiasat proses pembelajaran

Seperti yang telah kita sebutkan, proses pembelajaran adalah keseimbangan antara penerokaan dan penerokaan pengetahuan yang diperoleh tentang struktur ruang masalah. Kita telah melihat bahawa hasil pembelajaran (keupayaan untuk membantu agen mencari laluan pendek ke matlamat) telah bertambah baik, tetapi juga menarik untuk melihat bagaimana panjang laluan purata berkelakuan semasa proses pembelajaran:

## Ringkasan pembelajaran

- **Panjang laluan purata meningkat**. Apa yang kita lihat di sini adalah bahawa pada mulanya, panjang laluan purata meningkat. Ini mungkin disebabkan oleh fakta bahawa apabila kita tidak tahu apa-apa tentang persekitaran, kita cenderung terperangkap dalam keadaan buruk, air atau serigala. Apabila kita belajar lebih banyak dan mula menggunakan pengetahuan ini, kita boleh meneroka persekitaran lebih lama, tetapi kita masih tidak tahu di mana epal berada dengan baik.

- **Panjang laluan berkurangan, apabila kita belajar lebih banyak**. Setelah kita belajar cukup, ia menjadi lebih mudah untuk agen mencapai matlamat, dan panjang laluan mula berkurangan. Walau bagaimanapun, kita masih terbuka kepada penerokaan, jadi kita sering menyimpang dari laluan terbaik, dan meneroka pilihan baru, menjadikan laluan lebih panjang daripada yang optimum.

- **Panjang meningkat secara mendadak**. Apa yang kita juga perhatikan pada graf ini ialah pada satu ketika, panjang meningkat secara mendadak. Ini menunjukkan sifat stokastik proses, dan bahawa kita pada satu ketika boleh "merosakkan" pekali Jadual-Q dengan menulis semula mereka dengan nilai baru. Ini sepatutnya diminimumkan dengan mengurangkan kadar pembelajaran (contohnya, menjelang akhir latihan, kita hanya menyesuaikan nilai Jadual-Q dengan nilai kecil).

Secara keseluruhan, adalah penting untuk diingat bahawa kejayaan dan kualiti proses pembelajaran sangat bergantung kepada parameter, seperti kadar pembelajaran, pengurangan kadar pembelajaran, dan faktor diskaun. Ini sering dipanggil **hiperparameter**, untuk membezakannya daripada **parameter**, yang kita optimakan semasa latihan (contohnya, pekali Jadual-Q). Proses mencari nilai hiperparameter terbaik dipanggil **pengoptimuman hiperparameter**, dan ia layak mendapat topik yang berasingan.

## [Kuiz selepas kuliah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## Tugasan 
[Dunia yang Lebih Realistik](assignment.md)

**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI berasaskan mesin. Walaupun kami berusaha untuk ketepatan, sila maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.