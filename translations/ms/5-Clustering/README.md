<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T19:10:46+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "ms"
}
-->
# Model pengelompokan untuk pembelajaran mesin

Pengelompokan adalah tugas pembelajaran mesin di mana ia mencari objek yang menyerupai satu sama lain dan mengelompokkannya ke dalam kumpulan yang dipanggil kluster. Apa yang membezakan pengelompokan daripada pendekatan lain dalam pembelajaran mesin ialah prosesnya berlaku secara automatik, malah boleh dikatakan ia bertentangan dengan pembelajaran terarah.

## Topik serantau: model pengelompokan untuk citarasa muzik penonton Nigeria 🎧

Penonton Nigeria yang pelbagai mempunyai citarasa muzik yang berbeza-beza. Dengan menggunakan data yang diambil dari Spotify (diilhamkan oleh [artikel ini](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), mari kita lihat beberapa muzik yang popular di Nigeria. Dataset ini merangkumi data tentang skor 'danceability', 'acousticness', kelantangan, 'speechiness', populariti, dan tenaga pelbagai lagu. Ia akan menjadi menarik untuk menemui corak dalam data ini!

![Turntable](../../../5-Clustering/images/turntable.jpg)

> Foto oleh <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> di <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Dalam siri pelajaran ini, anda akan menemui cara baharu untuk menganalisis data menggunakan teknik pengelompokan. Pengelompokan sangat berguna apabila dataset anda tidak mempunyai label. Jika dataset anda mempunyai label, maka teknik klasifikasi seperti yang telah anda pelajari dalam pelajaran sebelumnya mungkin lebih berguna. Tetapi dalam kes di mana anda ingin mengelompokkan data yang tidak berlabel, pengelompokan adalah cara yang hebat untuk menemui corak.

> Terdapat alat low-code yang berguna untuk membantu anda mempelajari cara bekerja dengan model pengelompokan. Cuba [Azure ML untuk tugas ini](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Pelajaran

1. [Pengenalan kepada pengelompokan](1-Visualize/README.md)
2. [Pengelompokan K-Means](2-K-Means/README.md)

## Kredit

Pelajaran ini ditulis dengan 🎶 oleh [Jen Looper](https://www.twitter.com/jenlooper) dengan ulasan berguna oleh [Rishit Dagli](https://rishit_dagli) dan [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Dataset [Lagu Nigeria](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) diperoleh dari Kaggle sebagai hasil pengambilan data dari Spotify.

Contoh K-Means yang berguna yang membantu dalam mencipta pelajaran ini termasuk [eksplorasi iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [notebook pengenalan](https://www.kaggle.com/prashant111/k-means-clustering-with-python), dan [contoh NGO hipotetikal](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.