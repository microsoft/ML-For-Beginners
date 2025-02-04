# Pengenalan kepada pembelajaran pengukuhan

Pembelajaran pengukuhan, RL, dianggap sebagai salah satu paradigma pembelajaran mesin asas, selain pembelajaran diselia dan pembelajaran tanpa penyeliaan. RL berkaitan dengan keputusan: membuat keputusan yang betul atau sekurang-kurangnya belajar daripadanya.

Bayangkan anda mempunyai persekitaran simulasi seperti pasaran saham. Apa yang berlaku jika anda mengenakan peraturan tertentu? Adakah ia memberi kesan positif atau negatif? Jika sesuatu yang negatif berlaku, anda perlu mengambil _pengukuhan negatif_ ini, belajar daripadanya, dan mengubah haluan. Jika hasilnya positif, anda perlu membina dari _pengukuhan positif_ itu.

![peter dan serigala](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.ms.png)

> Peter dan kawan-kawannya perlu melarikan diri dari serigala yang lapar! Imej oleh [Jen Looper](https://twitter.com/jenlooper)

## Topik serantau: Peter dan Serigala (Rusia)

[Peter dan Serigala](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) adalah kisah dongeng muzik yang ditulis oleh komposer Rusia [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Ia adalah cerita tentang pelopor muda Peter, yang dengan berani keluar dari rumahnya ke kawasan hutan untuk mengejar serigala. Dalam bahagian ini, kita akan melatih algoritma pembelajaran mesin yang akan membantu Peter:

- **Meneroka** kawasan sekitar dan membina peta navigasi yang optimum
- **Belajar** cara menggunakan papan selaju dan mengimbangi di atasnya, untuk bergerak dengan lebih pantas.

[![Peter dan Serigala](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klik imej di atas untuk mendengar Peter dan Serigala oleh Prokofiev

## Pembelajaran pengukuhan

Dalam bahagian sebelum ini, anda telah melihat dua contoh masalah pembelajaran mesin:

- **Diselia**, di mana kita mempunyai set data yang mencadangkan penyelesaian sampel kepada masalah yang ingin kita selesaikan. [Klasifikasi](../4-Classification/README.md) dan [regresi](../2-Regression/README.md) adalah tugas pembelajaran diselia.
- **Tanpa penyeliaan**, di mana kita tidak mempunyai data latihan berlabel. Contoh utama pembelajaran tanpa penyeliaan ialah [Pengelompokan](../5-Clustering/README.md).

Dalam bahagian ini, kami akan memperkenalkan anda kepada jenis masalah pembelajaran baharu yang tidak memerlukan data latihan berlabel. Terdapat beberapa jenis masalah sedemikian:

- **[Pembelajaran separa diselia](https://wikipedia.org/wiki/Semi-supervised_learning)**, di mana kita mempunyai banyak data tidak berlabel yang boleh digunakan untuk pra-latihan model.
- **[Pembelajaran pengukuhan](https://wikipedia.org/wiki/Reinforcement_learning)**, di mana agen belajar bagaimana untuk bertindak dengan melakukan eksperimen dalam beberapa persekitaran simulasi.

### Contoh - permainan komputer

Katakan anda ingin mengajar komputer untuk bermain permainan, seperti catur, atau [Super Mario](https://wikipedia.org/wiki/Super_Mario). Untuk komputer bermain permainan, kita perlu ia meramalkan langkah mana yang perlu diambil dalam setiap keadaan permainan. Walaupun ini mungkin kelihatan seperti masalah klasifikasi, ia tidak - kerana kita tidak mempunyai set data dengan keadaan dan tindakan yang sepadan. Walaupun kita mungkin mempunyai beberapa data seperti perlawanan catur sedia ada atau rakaman pemain bermain Super Mario, kemungkinan besar data itu tidak akan cukup meliputi sejumlah besar keadaan yang mungkin.

Daripada mencari data permainan sedia ada, **Pembelajaran Pengukuhan** (RL) adalah berdasarkan idea *membuat komputer bermain* banyak kali dan memerhatikan hasilnya. Oleh itu, untuk menggunakan Pembelajaran Pengukuhan, kita memerlukan dua perkara:

- **Persekitaran** dan **simulator** yang membolehkan kita bermain permainan banyak kali. Simulator ini akan menentukan semua peraturan permainan serta keadaan dan tindakan yang mungkin.

- **Fungsi ganjaran**, yang akan memberitahu kita seberapa baik kita melakukannya semasa setiap langkah atau permainan.

Perbezaan utama antara jenis pembelajaran mesin lain dan RL ialah dalam RL kita biasanya tidak tahu sama ada kita menang atau kalah sehingga kita menyelesaikan permainan. Oleh itu, kita tidak boleh mengatakan sama ada langkah tertentu sahaja adalah baik atau tidak - kita hanya menerima ganjaran pada akhir permainan. Dan matlamat kita adalah untuk mereka bentuk algoritma yang akan membolehkan kita melatih model di bawah keadaan yang tidak pasti. Kita akan belajar tentang satu algoritma RL yang dipanggil **Q-learning**.

## Pelajaran

1. [Pengenalan kepada pembelajaran pengukuhan dan Q-Learning](1-QLearning/README.md)
2. [Menggunakan persekitaran simulasi gim](2-Gym/README.md)

## Kredit

"Pengenalan kepada Pembelajaran Pengukuhan" ditulis dengan ♥️ oleh [Dmitry Soshnikov](http://soshnikov.com)

**Penafian**: 
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI berasaskan mesin. Walaupun kami berusaha untuk ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat kritikal, terjemahan manusia profesional disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.