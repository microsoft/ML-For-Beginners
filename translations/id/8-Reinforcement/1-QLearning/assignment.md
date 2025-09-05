<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T20:18:01+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "id"
}
-->
# Dunia yang Lebih Realistis

Dalam situasi kita, Peter dapat bergerak hampir tanpa merasa lelah atau lapar. Dalam dunia yang lebih realistis, dia harus duduk dan beristirahat dari waktu ke waktu, serta memberi makan dirinya sendiri. Mari kita buat dunia kita lebih realistis dengan menerapkan aturan berikut:

1. Dengan berpindah dari satu tempat ke tempat lain, Peter kehilangan **energi** dan mendapatkan **kelelahan**.
2. Peter dapat mendapatkan lebih banyak energi dengan memakan apel.
3. Peter dapat menghilangkan kelelahan dengan beristirahat di bawah pohon atau di atas rumput (yaitu berjalan ke lokasi papan dengan pohon atau rumput - lapangan hijau).
4. Peter perlu menemukan dan membunuh serigala.
5. Untuk membunuh serigala, Peter harus memiliki tingkat energi dan kelelahan tertentu, jika tidak, dia akan kalah dalam pertempuran.

## Instruksi

Gunakan notebook [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) asli sebagai titik awal untuk solusi Anda.

Modifikasi fungsi reward di atas sesuai dengan aturan permainan, jalankan algoritma pembelajaran penguatan untuk mempelajari strategi terbaik dalam memenangkan permainan, dan bandingkan hasil dari jalan acak dengan algoritma Anda dalam hal jumlah permainan yang dimenangkan dan kalah.

> **Note**: Dalam dunia baru Anda, keadaan menjadi lebih kompleks, dan selain posisi manusia juga mencakup tingkat kelelahan dan energi. Anda dapat memilih untuk merepresentasikan keadaan sebagai tuple (Board,energy,fatigue), atau mendefinisikan sebuah kelas untuk keadaan (Anda juga dapat menurunkannya dari `Board`), atau bahkan memodifikasi kelas `Board` asli di [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Dalam solusi Anda, harap pertahankan kode yang bertanggung jawab untuk strategi jalan acak, dan bandingkan hasil algoritma Anda dengan jalan acak di akhir.

> **Note**: Anda mungkin perlu menyesuaikan hiperparameter agar berhasil, terutama jumlah epoch. Karena keberhasilan permainan (melawan serigala) adalah peristiwa yang jarang terjadi, Anda dapat mengharapkan waktu pelatihan yang jauh lebih lama.

## Rubrik

| Kriteria | Unggul                                                                                                                                                                                                 | Memadai                                                                                                                                                                                | Perlu Peningkatan                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Notebook disajikan dengan definisi aturan dunia baru, algoritma Q-Learning, dan beberapa penjelasan tekstual. Q-Learning mampu secara signifikan meningkatkan hasil dibandingkan dengan jalan acak.    | Notebook disajikan, Q-Learning diimplementasikan dan meningkatkan hasil dibandingkan dengan jalan acak, tetapi tidak secara signifikan; atau notebook kurang terdokumentasi dan kode tidak terstruktur dengan baik. | Beberapa upaya untuk mendefinisikan ulang aturan dunia dilakukan, tetapi algoritma Q-Learning tidak berfungsi, atau fungsi reward tidak sepenuhnya didefinisikan. |

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diperhatikan bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.