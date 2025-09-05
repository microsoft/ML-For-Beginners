<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T20:10:05+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "id"
}
-->
# Pengantar Pembelajaran Penguatan

Pembelajaran penguatan, atau RL, dianggap sebagai salah satu paradigma dasar pembelajaran mesin, selain pembelajaran terawasi dan pembelajaran tak terawasi. RL berfokus pada pengambilan keputusan: membuat keputusan yang tepat atau setidaknya belajar dari keputusan tersebut.

Bayangkan Anda memiliki lingkungan simulasi seperti pasar saham. Apa yang terjadi jika Anda menerapkan suatu regulasi tertentu? Apakah dampaknya positif atau negatif? Jika terjadi sesuatu yang negatif, Anda perlu mengambil _penguatan negatif_, belajar darinya, dan mengubah arah. Jika hasilnya positif, Anda perlu membangun dari _penguatan positif_ tersebut.

![peter dan serigala](../../../8-Reinforcement/images/peter.png)

> Peter dan teman-temannya harus melarikan diri dari serigala yang lapar! Gambar oleh [Jen Looper](https://twitter.com/jenlooper)

## Topik Regional: Peter dan Serigala (Rusia)

[Peter dan Serigala](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) adalah dongeng musikal yang ditulis oleh komposer Rusia [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Ini adalah kisah tentang pionir muda Peter, yang dengan berani keluar dari rumahnya menuju hutan untuk mengejar serigala. Dalam bagian ini, kita akan melatih algoritma pembelajaran mesin yang akan membantu Peter:

- **Menjelajahi** area sekitar dan membangun peta navigasi yang optimal
- **Belajar** menggunakan skateboard dan menjaga keseimbangan di atasnya, agar dapat bergerak lebih cepat.

[![Peter dan Serigala](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klik gambar di atas untuk mendengarkan Peter dan Serigala oleh Prokofiev

## Pembelajaran Penguatan

Pada bagian sebelumnya, Anda telah melihat dua contoh masalah pembelajaran mesin:

- **Terawasi**, di mana kita memiliki dataset yang menyarankan solusi contoh untuk masalah yang ingin kita selesaikan. [Klasifikasi](../4-Classification/README.md) dan [regresi](../2-Regression/README.md) adalah tugas pembelajaran terawasi.
- **Tak terawasi**, di mana kita tidak memiliki data pelatihan yang diberi label. Contoh utama pembelajaran tak terawasi adalah [Pengelompokan](../5-Clustering/README.md).

Dalam bagian ini, kita akan memperkenalkan jenis masalah pembelajaran baru yang tidak memerlukan data pelatihan yang diberi label. Ada beberapa jenis masalah seperti ini:

- **[Pembelajaran semi-terawasi](https://wikipedia.org/wiki/Semi-supervised_learning)**, di mana kita memiliki banyak data yang tidak diberi label yang dapat digunakan untuk pra-pelatihan model.
- **[Pembelajaran penguatan](https://wikipedia.org/wiki/Reinforcement_learning)**, di mana agen belajar bagaimana berperilaku dengan melakukan eksperimen dalam lingkungan simulasi tertentu.

### Contoh - permainan komputer

Misalkan Anda ingin mengajarkan komputer untuk bermain game, seperti catur, atau [Super Mario](https://wikipedia.org/wiki/Super_Mario). Agar komputer dapat bermain game, kita perlu memprediksi langkah apa yang harus diambil dalam setiap keadaan permainan. Meskipun ini mungkin tampak seperti masalah klasifikasi, sebenarnya tidak - karena kita tidak memiliki dataset dengan keadaan dan tindakan yang sesuai. Meskipun kita mungkin memiliki beberapa data seperti pertandingan catur yang ada atau rekaman pemain yang bermain Super Mario, kemungkinan besar data tersebut tidak cukup mencakup sejumlah besar keadaan yang mungkin terjadi.

Alih-alih mencari data game yang ada, **Pembelajaran Penguatan** (RL) didasarkan pada ide *membuat komputer bermain* berkali-kali dan mengamati hasilnya. Jadi, untuk menerapkan Pembelajaran Penguatan, kita membutuhkan dua hal:

- **Sebuah lingkungan** dan **simulator** yang memungkinkan kita bermain game berkali-kali. Simulator ini akan mendefinisikan semua aturan permainan serta keadaan dan tindakan yang mungkin.

- **Fungsi penghargaan**, yang akan memberi tahu kita seberapa baik kita melakukannya selama setiap langkah atau permainan.

Perbedaan utama antara jenis pembelajaran mesin lainnya dan RL adalah bahwa dalam RL kita biasanya tidak tahu apakah kita menang atau kalah sampai kita menyelesaikan permainan. Jadi, kita tidak dapat mengatakan apakah langkah tertentu saja baik atau tidak - kita hanya menerima penghargaan di akhir permainan. Dan tujuan kita adalah merancang algoritma yang memungkinkan kita melatih model dalam kondisi yang tidak pasti. Kita akan belajar tentang salah satu algoritma RL yang disebut **Q-learning**.

## Pelajaran

1. [Pengantar pembelajaran penguatan dan Q-Learning](1-QLearning/README.md)
2. [Menggunakan lingkungan simulasi gym](2-Gym/README.md)

## Kredit

"Pengantar Pembelajaran Penguatan" ditulis dengan ♥️ oleh [Dmitry Soshnikov](http://soshnikov.com)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.