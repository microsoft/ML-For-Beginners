# Latih Kereta Gunung

[OpenAI Gym](http://gym.openai.com) telah direka sedemikian rupa sehingga semua persekitaran menyediakan API yang sama - iaitu kaedah yang sama `reset`, `step` dan `render`, dan abstraksi yang sama dari **ruang tindakan** dan **ruang pemerhatian**. Oleh itu, seharusnya mungkin untuk menyesuaikan algoritma pembelajaran pengukuhan yang sama ke persekitaran yang berbeza dengan perubahan kod yang minimum.

## Persekitaran Kereta Gunung

[Persekitaran Kereta Gunung](https://gym.openai.com/envs/MountainCar-v0/) mengandungi sebuah kereta yang terperangkap di dalam lembah:
Matlamatnya adalah untuk keluar dari lembah dan menangkap bendera, dengan melakukan salah satu tindakan berikut pada setiap langkah:

| Nilai | Makna |
|---|---|
| 0 | Mempercepat ke kiri |
| 1 | Tidak mempercepat |
| 2 | Mempercepat ke kanan |

Trik utama masalah ini, bagaimanapun, adalah bahawa enjin kereta tidak cukup kuat untuk mendaki gunung dalam satu kali laluan. Oleh itu, satu-satunya cara untuk berjaya adalah dengan memandu ke depan dan ke belakang untuk membina momentum.

Ruang pemerhatian terdiri daripada hanya dua nilai:

| Nombor | Pemerhatian  | Min | Maks |
|-----|--------------|-----|-----|
|  0  | Kedudukan Kereta | -1.2| 0.6 |
|  1  | Kelajuan Kereta | -0.07 | 0.07 |

Sistem ganjaran untuk kereta gunung agak rumit:

 * Ganjaran 0 diberikan jika agen mencapai bendera (kedudukan = 0.5) di atas gunung.
 * Ganjaran -1 diberikan jika kedudukan agen kurang dari 0.5.

Episod akan tamat jika kedudukan kereta lebih dari 0.5, atau panjang episod lebih dari 200.
## Arahan

Sesuaikan algoritma pembelajaran pengukuhan kami untuk menyelesaikan masalah kereta gunung. Mulakan dengan kod [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) yang ada, gantikan persekitaran baru, ubah fungsi diskritisasi keadaan, dan cuba buat algoritma yang ada untuk dilatih dengan perubahan kod yang minimum. Optimumkan hasil dengan menyesuaikan hiperparameter.

> **Note**: Penyesuaian hiperparameter mungkin diperlukan untuk membuat algoritma konvergen. 
## Rubrik

| Kriteria | Cemerlang | Memadai | Perlu Peningkatan |
| -------- | --------- | -------- | ----------------- |
|          | Algoritma Q-Learning berjaya disesuaikan dari contoh CartPole, dengan perubahan kod yang minimum, yang mampu menyelesaikan masalah menangkap bendera di bawah 200 langkah. | Algoritma Q-Learning baru telah diadopsi dari Internet, tetapi didokumentasikan dengan baik; atau algoritma yang ada diadopsi, tetapi tidak mencapai hasil yang diinginkan | Pelajar tidak dapat mengadopsi algoritma dengan berjaya, tetapi telah membuat langkah-langkah besar menuju penyelesaian (melaksanakan diskritisasi keadaan, struktur data Q-Table, dll.) |

**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI berasaskan mesin. Walaupun kami berusaha untuk ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat kritikal, terjemahan manusia profesional disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.