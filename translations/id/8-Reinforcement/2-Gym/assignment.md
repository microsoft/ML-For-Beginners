<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T20:23:20+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "id"
}
-->
# Melatih Mountain Car

[OpenAI Gym](http://gym.openai.com) dirancang sedemikian rupa sehingga semua lingkungan menyediakan API yang sama - yaitu metode yang sama `reset`, `step`, dan `render`, serta abstraksi yang sama untuk **action space** dan **observation space**. Oleh karena itu, seharusnya memungkinkan untuk mengadaptasi algoritma pembelajaran penguatan yang sama ke berbagai lingkungan dengan perubahan kode yang minimal.

## Lingkungan Mountain Car

[Lingkungan Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) berisi sebuah mobil yang terjebak di lembah:

Tujuannya adalah keluar dari lembah dan menangkap bendera, dengan melakukan salah satu tindakan berikut di setiap langkah:

| Nilai | Makna |
|---|---|
| 0 | Mempercepat ke kiri |
| 1 | Tidak mempercepat |
| 2 | Mempercepat ke kanan |

Namun, trik utama dari masalah ini adalah bahwa mesin mobil tidak cukup kuat untuk mendaki gunung dalam satu kali perjalanan. Oleh karena itu, satu-satunya cara untuk berhasil adalah dengan bergerak maju mundur untuk membangun momentum.

Observation space hanya terdiri dari dua nilai:

| Num | Observasi      | Min   | Max   |
|-----|----------------|-------|-------|
|  0  | Posisi Mobil   | -1.2  | 0.6   |
|  1  | Kecepatan Mobil| -0.07 | 0.07  |

Sistem reward untuk Mountain Car cukup rumit:

 * Reward sebesar 0 diberikan jika agen berhasil mencapai bendera (posisi = 0.5) di puncak gunung.
 * Reward sebesar -1 diberikan jika posisi agen kurang dari 0.5.

Episode berakhir jika posisi mobil lebih dari 0.5, atau panjang episode lebih dari 200.
## Instruksi

Adaptasikan algoritma pembelajaran penguatan kami untuk menyelesaikan masalah Mountain Car. Mulailah dengan kode yang ada di [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), ganti lingkungan baru, ubah fungsi diskretisasi state, dan coba buat algoritma yang ada untuk melatih dengan perubahan kode yang minimal. Optimalkan hasil dengan menyesuaikan hyperparameter.

> **Note**: Penyesuaian hyperparameter kemungkinan diperlukan agar algoritma dapat konvergen.
## Rubrik

| Kriteria | Unggul | Memadai | Perlu Peningkatan |
| -------- | -------| --------| ----------------- |
|          | Algoritma Q-Learning berhasil diadaptasi dari contoh CartPole, dengan perubahan kode minimal, yang mampu menyelesaikan masalah menangkap bendera dalam kurang dari 200 langkah. | Algoritma Q-Learning baru telah diadopsi dari Internet, tetapi terdokumentasi dengan baik; atau algoritma yang ada diadopsi, tetapi tidak mencapai hasil yang diinginkan. | Mahasiswa tidak berhasil mengadopsi algoritma apa pun, tetapi telah membuat langkah substansial menuju solusi (mengimplementasikan diskretisasi state, struktur data Q-Table, dll.) |

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.