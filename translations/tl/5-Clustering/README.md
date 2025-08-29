<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-08-29T13:18:57+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "tl"
}
-->
# Mga modelo ng clustering para sa machine learning

Ang clustering ay isang gawain sa machine learning kung saan sinusubukan nitong hanapin ang mga bagay na magkahawig at pagsama-samahin ang mga ito sa mga grupo na tinatawag na clusters. Ang kaibahan ng clustering sa ibang mga pamamaraan sa machine learning ay nangyayari ito nang awtomatiko. Sa katunayan, maituturing na ito ang kabaligtaran ng supervised learning.

## Paksang rehiyonal: mga modelo ng clustering para sa panlasa sa musika ng mga taga-Nigeria 🎧

Ang iba't ibang audience sa Nigeria ay may iba't ibang panlasa sa musika. Gamit ang datos na nakuha mula sa Spotify (inspirado ng [artikulong ito](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), tingnan natin ang ilang musikang sikat sa Nigeria. Ang dataset na ito ay naglalaman ng impormasyon tungkol sa iba't ibang kanta tulad ng 'danceability' score, 'acousticness', lakas ng tunog (loudness), 'speechiness', kasikatan (popularity), at enerhiya. Magiging interesante ang pagtuklas ng mga pattern sa datos na ito!

![Isang turntable](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.tl.jpg)

> Larawan ni <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> sa <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Sa serye ng mga araling ito, matutuklasan mo ang mga bagong paraan ng pagsusuri ng datos gamit ang mga clustering technique. Ang clustering ay partikular na kapaki-pakinabang kapag ang iyong dataset ay walang mga label. Kung mayroon itong mga label, mas magiging kapaki-pakinabang ang mga classification technique tulad ng mga natutunan mo sa mga nakaraang aralin. Ngunit sa mga pagkakataong nais mong pagsama-samahin ang mga datos na walang label, ang clustering ay isang mahusay na paraan upang matuklasan ang mga pattern.

> May mga kapaki-pakinabang na low-code na mga tool na makakatulong sa iyong matutunan ang paggamit ng mga clustering model. Subukan ang [Azure ML para sa gawaing ito](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Mga Aralin

1. [Panimula sa clustering](1-Visualize/README.md)
2. [K-Means clustering](2-K-Means/README.md)

## Mga Kredito

Ang mga araling ito ay isinulat nang may 🎶 ni [Jen Looper](https://www.twitter.com/jenlooper) na may mga kapaki-pakinabang na pagsusuri mula kina [Rishit Dagli](https://rishit_dagli) at [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Ang [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) dataset ay nakuha mula sa Kaggle na kinalap mula sa Spotify.

Ang mga kapaki-pakinabang na halimbawa ng K-Means na tumulong sa paglikha ng araling ito ay kinabibilangan ng [eksplorasyon ng iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ang [introductory notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python), at ang [halimbawang NGO](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, pakitandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa orihinal nitong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.