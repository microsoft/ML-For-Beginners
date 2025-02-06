# Dunia yang Lebih Realistik

Dalam situasi kita, Peter dapat bergerak hampir tanpa merasa lelah atau lapar. Dalam dunia yang lebih realistik, dia perlu duduk dan berehat dari semasa ke semasa, dan juga makan untuk mendapatkan tenaga. Mari kita buat dunia kita lebih realistik dengan melaksanakan peraturan berikut:

1. Dengan bergerak dari satu tempat ke tempat lain, Peter kehilangan **tenaga** dan mendapat sedikit **keletihan**.
2. Peter boleh mendapatkan lebih banyak tenaga dengan memakan epal.
3. Peter boleh menghilangkan keletihan dengan berehat di bawah pokok atau di atas rumput (iaitu berjalan ke lokasi papan dengan pokok atau rumput - padang hijau)
4. Peter perlu mencari dan membunuh serigala
5. Untuk membunuh serigala, Peter perlu mempunyai tahap tenaga dan keletihan tertentu, jika tidak, dia akan kalah dalam pertempuran.

## Arahan

Gunakan notebook asal [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) sebagai titik permulaan untuk penyelesaian anda.

Ubah fungsi ganjaran di atas mengikut peraturan permainan, jalankan algoritma pembelajaran pengukuhan untuk mempelajari strategi terbaik untuk memenangi permainan, dan bandingkan hasil jalan rawak dengan algoritma anda dari segi jumlah permainan yang dimenangi dan kalah.

> **Note**: Dalam dunia baru anda, keadaan adalah lebih kompleks, dan selain daripada kedudukan manusia, juga termasuk tahap keletihan dan tenaga. Anda boleh memilih untuk mewakili keadaan sebagai tuple (Board,energy,fatigue), atau mendefinisikan kelas untuk keadaan tersebut (anda juga boleh mengembangkannya daripada `Board`), atau bahkan mengubah kelas asal `Board` dalam [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Dalam penyelesaian anda, sila simpan kod yang bertanggungjawab untuk strategi jalan rawak, dan bandingkan hasil algoritma anda dengan jalan rawak pada akhir.

> **Note**: Anda mungkin perlu menyesuaikan hiperparameter untuk membuatnya berfungsi, terutamanya jumlah epoch. Oleh kerana kejayaan permainan (melawan serigala) adalah kejadian yang jarang berlaku, anda boleh mengharapkan masa latihan yang lebih lama.

## Rubrik

| Kriteria | Cemerlang                                                                                                                                                                                             | Memadai                                                                                                                                                                                | Perlu Penambahbaikan                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Notebook disediakan dengan definisi peraturan dunia baru, algoritma Q-Learning dan beberapa penjelasan teks. Q-Learning mampu meningkatkan hasil dengan ketara berbanding jalan rawak. | Notebook disediakan, Q-Learning dilaksanakan dan meningkatkan hasil berbanding jalan rawak, tetapi tidak dengan ketara; atau notebook kurang didokumentasikan dan kod tidak disusun dengan baik | Beberapa percubaan untuk mentakrifkan semula peraturan dunia dibuat, tetapi algoritma Q-Learning tidak berfungsi, atau fungsi ganjaran tidak sepenuhnya ditakrifkan |

**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI berasaskan mesin. Walaupun kami berusaha untuk ketepatan, sila maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.