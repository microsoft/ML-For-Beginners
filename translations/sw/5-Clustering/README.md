<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T15:39:08+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "sw"
}
-->
# Miundo ya Klasta kwa Kujifunza kwa Mashine

Klasta ni kazi ya kujifunza kwa mashine ambapo inatafuta vitu vinavyofanana na kuviweka katika vikundi vinavyoitwa klasta. Kinachotofautisha klasta na mbinu nyingine za kujifunza kwa mashine ni kwamba mambo hufanyika kiotomatiki; kwa kweli, ni sahihi kusema ni kinyume cha kujifunza kwa usimamizi.

## Mada ya Kikanda: Miundo ya Klasta kwa Ladha ya Muziki ya Watazamaji wa Nigeria 🎧

Watazamaji wa Nigeria wana ladha mbalimbali za muziki. Tukitumia data iliyokusanywa kutoka Spotify (iliyopata msukumo kutoka [makala hii](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), hebu tuangalie baadhi ya muziki maarufu nchini Nigeria. Seti hii ya data inajumuisha taarifa kuhusu alama za 'danceability', 'acousticness', sauti, 'speechiness', umaarufu, na nishati ya nyimbo mbalimbali. Itakuwa ya kuvutia kugundua mifumo katika data hii!

![Turntable](../../../5-Clustering/images/turntable.jpg)

> Picha na <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> kwenye <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Katika mfululizo huu wa masomo, utagundua njia mpya za kuchambua data kwa kutumia mbinu za klasta. Klasta ni muhimu hasa pale ambapo seti yako ya data haina lebo. Ikiwa ina lebo, basi mbinu za uainishaji kama zile ulizojifunza katika masomo ya awali zinaweza kuwa na manufaa zaidi. Lakini katika hali ambapo unatafuta kuunda vikundi vya data isiyo na lebo, klasta ni njia nzuri ya kugundua mifumo.

> Kuna zana za kiwango cha chini cha msimbo ambazo zinaweza kusaidia kujifunza kuhusu kufanya kazi na miundo ya klasta. Jaribu [Azure ML kwa kazi hii](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Masomo

1. [Utangulizi wa klasta](1-Visualize/README.md)
2. [K-Means klasta](2-K-Means/README.md)

## Shukrani

Masomo haya yaliandikwa kwa 🎶 na [Jen Looper](https://www.twitter.com/jenlooper) kwa msaada wa ukaguzi wa [Rishit Dagli](https://rishit_dagli) na [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Seti ya data ya [Nyimbo za Nigeria](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ilitolewa kutoka Kaggle kama ilivyokusanywa kutoka Spotify.

Mifano muhimu ya K-Means iliyosaidia kuunda somo hili ni pamoja na [uchambuzi wa iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [notebook ya utangulizi](https://www.kaggle.com/prashant111/k-means-clustering-with-python), na [mfano wa NGO wa kubuni](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.