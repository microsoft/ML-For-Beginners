<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T20:18:21+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "ms"
}
-->
# Dunia yang Lebih Realistik

Dalam situasi kita, Peter dapat bergerak hampir tanpa merasa letih atau lapar. Dalam dunia yang lebih realistik, dia perlu duduk dan berehat dari semasa ke semasa, serta makan untuk mengisi tenaga. Mari kita jadikan dunia kita lebih realistik dengan melaksanakan peraturan berikut:

1. Dengan bergerak dari satu tempat ke tempat lain, Peter kehilangan **tenaga** dan mendapat sedikit **keletihan**.
2. Peter boleh mendapatkan lebih banyak tenaga dengan memakan epal.
3. Peter boleh menghilangkan keletihan dengan berehat di bawah pokok atau di atas rumput (iaitu berjalan ke lokasi papan yang mempunyai pokok atau rumput - padang hijau).
4. Peter perlu mencari dan membunuh serigala.
5. Untuk membunuh serigala, Peter perlu mempunyai tahap tenaga dan keletihan tertentu, jika tidak dia akan kalah dalam pertempuran.

## Arahan

Gunakan notebook asal [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) sebagai titik permulaan untuk penyelesaian anda.

Ubah suai fungsi ganjaran di atas mengikut peraturan permainan, jalankan algoritma pembelajaran pengukuhan untuk mempelajari strategi terbaik untuk memenangi permainan, dan bandingkan hasil jalan rawak dengan algoritma anda dari segi jumlah permainan yang dimenangi dan kalah.

> **Note**: Dalam dunia baru anda, keadaan lebih kompleks, dan selain kedudukan manusia, ia juga merangkumi tahap keletihan dan tenaga. Anda boleh memilih untuk mewakili keadaan sebagai tuple (Board,energy,fatigue), atau mendefinisikan kelas untuk keadaan (anda juga boleh memilih untuk mewarisinya daripada `Board`), atau bahkan mengubah suai kelas `Board` asal dalam [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Dalam penyelesaian anda, sila kekalkan kod yang bertanggungjawab untuk strategi jalan rawak, dan bandingkan hasil algoritma anda dengan jalan rawak pada akhirnya.

> **Note**: Anda mungkin perlu menyesuaikan hiperparameter untuk menjadikannya berfungsi, terutamanya bilangan epoch. Oleh kerana kejayaan permainan (melawan serigala) adalah peristiwa yang jarang berlaku, anda boleh menjangkakan masa latihan yang lebih lama.

## Rubrik

| Kriteria | Cemerlang                                                                                                                                                                                             | Memadai                                                                                                                                                                                | Perlu Penambahbaikan                                                                                                                       |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Notebook disediakan dengan definisi peraturan dunia baru, algoritma Q-Learning dan beberapa penjelasan teks. Q-Learning mampu meningkatkan hasil dengan ketara berbanding jalan rawak.                 | Notebook disediakan, Q-Learning dilaksanakan dan meningkatkan hasil berbanding jalan rawak, tetapi tidak dengan ketara; atau notebook kurang didokumentasikan dan kod tidak tersusun baik | Beberapa usaha untuk mentakrifkan semula peraturan dunia dibuat, tetapi algoritma Q-Learning tidak berfungsi, atau fungsi ganjaran tidak ditakrifkan sepenuhnya                  |

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.