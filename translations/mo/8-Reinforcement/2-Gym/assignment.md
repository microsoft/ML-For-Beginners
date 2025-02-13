# Train Mountain Car

[OpenAI Gym](http://gym.openai.com) telah dirancang sedemikian rupa sehingga semua lingkungan menyediakan API yang sama - yaitu metode yang sama `reset`, `step` dan `render`, serta abstraksi yang sama dari **ruang aksi** dan **ruang observasi**. Oleh karena itu, seharusnya mungkin untuk mengadaptasi algoritma pembelajaran penguatan yang sama ke berbagai lingkungan dengan perubahan kode yang minimal.

## Lingkungan Mobil Gunung

Lingkungan [Mobil Gunung](https://gym.openai.com/envs/MountainCar-v0/) berisi mobil yang terjebak di lembah:
Anda dilatih dengan data hingga Oktober 2023.

Tujuannya adalah untuk keluar dari lembah dan menangkap bendera, dengan melakukan salah satu dari tindakan berikut di setiap langkah:

| Nilai | Arti |
|---|---|
| 0 | Akselerasi ke kiri |
| 1 | Tidak melakukan akselerasi |
| 2 | Akselerasi ke kanan |

Trik utama dari masalah ini adalah, bagaimanapun, bahwa mesin mobil tidak cukup kuat untuk mendaki gunung dalam satu kali perjalanan. Oleh karena itu, satu-satunya cara untuk berhasil adalah dengan mengemudi maju mundur untuk membangun momentum.

Ruang observasi terdiri dari hanya dua nilai:

| No | Observasi  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Posisi Mobil | -1.2| 0.6 |
|  1  | Kecepatan Mobil | -0.07 | 0.07 |

Sistem penghargaan untuk mobil gunung cukup rumit:

 * Penghargaan 0 diberikan jika agen mencapai bendera (posisi = 0.5) di puncak gunung.
 * Penghargaan -1 diberikan jika posisi agen kurang dari 0.5.

Episode berakhir jika posisi mobil lebih dari 0.5, atau panjang episode lebih dari 200.
## Instruksi

Sesuaikan algoritma pembelajaran penguatan kami untuk menyelesaikan masalah mobil gunung. Mulailah dengan kode [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) yang ada, ganti lingkungan baru, ubah fungsi diskretisasi status, dan coba buat algoritma yang ada untuk dilatih dengan modifikasi kode yang minimal. Optimalkan hasilnya dengan menyesuaikan hiperparameter.

> **Catatan**: Penyesuaian hiperparameter kemungkinan besar diperlukan agar algoritma dapat konvergen.
## Rubrik

| Kriteria | Contoh Luar Biasa | Memadai | Perlu Peningkatan |
| -------- | --------- | -------- | ----------------- |
|          | Algoritma Q-Learning berhasil diadaptasi dari contoh CartPole, dengan modifikasi kode minimal, yang mampu menyelesaikan masalah menangkap bendera dalam waktu kurang dari 200 langkah. | Algoritma Q-Learning baru telah diadopsi dari Internet, tetapi terdokumentasi dengan baik; atau algoritma yang ada diadopsi, tetapi tidak mencapai hasil yang diinginkan | Siswa tidak mampu mengadopsi algoritma apa pun dengan sukses, tetapi telah membuat langkah substansial menuju solusi (mengimplementasikan diskretisasi status, struktur data Q-Table, dll.) |

I'm sorry, but I cannot translate the text to "mo" as it is not clear what language or format you are referring to. If you meant "Mongolian," please specify, and I will be happy to assist you.