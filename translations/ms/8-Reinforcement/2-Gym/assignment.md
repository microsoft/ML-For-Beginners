<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T20:23:30+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "ms"
}
-->
# Latih Mountain Car

[OpenAI Gym](http://gym.openai.com) telah direka sedemikian rupa sehingga semua persekitaran menyediakan API yang sama - iaitu kaedah yang sama `reset`, `step` dan `render`, serta abstraksi yang sama untuk **ruang tindakan** dan **ruang pemerhatian**. Oleh itu, seharusnya mungkin untuk menyesuaikan algoritma pembelajaran pengukuhan yang sama kepada persekitaran yang berbeza dengan perubahan kod yang minimum.

## Persekitaran Mountain Car

[Persekitaran Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) mengandungi sebuah kereta yang terperangkap di dalam lembah:

Tujuannya adalah untuk keluar dari lembah dan menangkap bendera, dengan melakukan salah satu tindakan berikut pada setiap langkah:

| Nilai | Maksud |
|---|---|
| 0 | Memecut ke kiri |
| 1 | Tidak memecut |
| 2 | Memecut ke kanan |

Namun, cabaran utama masalah ini adalah bahawa enjin kereta tidak cukup kuat untuk mendaki gunung dalam satu percubaan. Oleh itu, satu-satunya cara untuk berjaya adalah dengan memandu ke depan dan ke belakang untuk membina momentum.

Ruang pemerhatian hanya terdiri daripada dua nilai:

| Nombor | Pemerhatian  | Min | Maks |
|-----|--------------|-----|-----|
|  0  | Kedudukan Kereta | -1.2| 0.6 |
|  1  | Kelajuan Kereta | -0.07 | 0.07 |

Sistem ganjaran untuk Mountain Car agak rumit:

 * Ganjaran 0 diberikan jika agen mencapai bendera (kedudukan = 0.5) di puncak gunung.
 * Ganjaran -1 diberikan jika kedudukan agen kurang daripada 0.5.

Episod akan tamat jika kedudukan kereta melebihi 0.5, atau panjang episod melebihi 200.
## Arahan

Sesuaikan algoritma pembelajaran pengukuhan kami untuk menyelesaikan masalah Mountain Car. Mulakan dengan kod [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) yang sedia ada, gantikan persekitaran baru, ubah fungsi diskretisasi keadaan, dan cuba membuat algoritma sedia ada berlatih dengan perubahan kod yang minimum. Optimumkan hasil dengan menyesuaikan hiperparameter.

> **Nota**: Penyesuaian hiperparameter mungkin diperlukan untuk membuat algoritma berjaya. 
## Rubrik

| Kriteria | Cemerlang | Memadai | Perlu Penambahbaikan |
| -------- | --------- | -------- | ----------------- |
|          | Algoritma Q-Learning berjaya disesuaikan daripada contoh CartPole, dengan perubahan kod yang minimum, dan mampu menyelesaikan masalah menangkap bendera dalam kurang daripada 200 langkah. | Algoritma Q-Learning baru telah diambil dari Internet, tetapi didokumentasikan dengan baik; atau algoritma sedia ada disesuaikan, tetapi tidak mencapai hasil yang diinginkan | Pelajar tidak berjaya menyesuaikan sebarang algoritma, tetapi telah membuat langkah yang besar ke arah penyelesaian (melaksanakan diskretisasi keadaan, struktur data Q-Table, dll.) |

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.