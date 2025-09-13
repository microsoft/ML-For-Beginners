<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-08-29T17:08:26+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "pa"
}
-->
# ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਲਈ ਕਲੱਸਟਰਿੰਗ ਮਾਡਲ

ਕਲੱਸਟਰਿੰਗ ਇੱਕ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦਾ ਕੰਮ ਹੈ ਜਿਸ ਵਿੱਚ ਇਹ ਉਹਨਾਂ ਚੀਜ਼ਾਂ ਨੂੰ ਲੱਭਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਦਾ ਹੈ ਜੋ ਇੱਕ ਦੂਜੇ ਨਾਲ ਮਿਲਦੀਆਂ ਹਨ ਅਤੇ ਉਨ੍ਹਾਂ ਨੂੰ ਕਲੱਸਟਰਾਂ ਦੇ ਰੂਪ ਵਿੱਚ ਸਮੂਹਬੱਧ ਕਰਦਾ ਹੈ। ਕਲੱਸਟਰਿੰਗ ਨੂੰ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦੇ ਹੋਰ ਪਹੁੰਚਾਂ ਤੋਂ ਵੱਖਰਾ ਬਣਾਉਣ ਵਾਲੀ ਗੱਲ ਇਹ ਹੈ ਕਿ ਇਹ ਸਾਰਾ ਕੰਮ ਆਟੋਮੈਟਿਕ ਤੌਰ 'ਤੇ ਹੁੰਦਾ ਹੈ। ਅਸਲ ਵਿੱਚ, ਇਹ ਕਹਿਣਾ ਸਹੀ ਹੋਵੇਗਾ ਕਿ ਇਹ ਸੁਪਰਵਾਈਜ਼ਡ ਲਰਨਿੰਗ ਦੇ ਉਲਟ ਹੈ।

## ਖੇਤਰੀ ਵਿਸ਼ਾ: ਨਾਈਜੀਰੀਆਈ ਦਰਸ਼ਕਾਂ ਦੇ ਸੰਗੀਤਕ ਰੁਚੀਆਂ ਲਈ ਕਲੱਸਟਰਿੰਗ ਮਾਡਲ 🎧

ਨਾਈਜੀਰੀਆ ਦੇ ਵੱਖ-ਵੱਖ ਦਰਸ਼ਕਾਂ ਦੀਆਂ ਵੱਖ-ਵੱਖ ਸੰਗੀਤਕ ਰੁਚੀਆਂ ਹਨ। Spotify ਤੋਂ ਡਾਟਾ ਇਕੱਠਾ ਕਰਕੇ (ਇਸ ਲੇਖ ਤੋਂ ਪ੍ਰੇਰਿਤ [ਇਥੇ](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), ਆਓ ਨਾਈਜੀਰੀਆ ਵਿੱਚ ਪ੍ਰਸਿੱਧ ਕੁਝ ਸੰਗੀਤਾਂ ਨੂੰ ਵੇਖੀਏ। ਇਸ ਡਾਟਾਸੈਟ ਵਿੱਚ ਵੱਖ-ਵੱਖ ਗੀਤਾਂ ਦੇ 'ਡਾਂਸੇਬਿਲਿਟੀ' ਸਕੋਰ, 'ਅਕੂਸਟਿਕਨੈਸ', ਲਾਊਡਨੈਸ, 'ਸਪੀਚੀਨੈਸ', ਪ੍ਰਸਿੱਧੀ ਅਤੇ ਊਰਜਾ ਬਾਰੇ ਡਾਟਾ ਸ਼ਾਮਲ ਹੈ। ਇਸ ਡਾਟੇ ਵਿੱਚ ਪੈਟਰਨ ਲੱਭਣਾ ਦਿਲਚਸਪ ਹੋਵੇਗਾ!

![ਇੱਕ ਟਰਨਟੇਬਲ](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.pa.jpg)

> ਫੋਟੋ <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਮਾਰਸੇਲਾ ਲਾਸਕੋਸਕੀ</a> ਦੁਆਰਾ <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਅਨਸਪਲੈਸ਼</a> 'ਤੇ
  
ਇਸ ਪਾਠਮਾਲਾ ਦੀ ਲੜੀ ਵਿੱਚ, ਤੁਸੀਂ ਕਲੱਸਟਰਿੰਗ ਤਕਨੀਕਾਂ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਡਾਟੇ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਨ ਦੇ ਨਵੇਂ ਤਰੀਕੇ ਸਿੱਖੋਗੇ। ਕਲੱਸਟਰਿੰਗ ਖਾਸ ਤੌਰ 'ਤੇ ਉਸ ਸਮੇਂ ਲਾਭਦਾਇਕ ਹੁੰਦੀ ਹੈ ਜਦੋਂ ਤੁਹਾਡੇ ਡਾਟਾਸੈਟ ਵਿੱਚ ਲੇਬਲ ਨਹੀਂ ਹੁੰਦੇ। ਜੇ ਇਸ ਵਿੱਚ ਲੇਬਲ ਹਨ, ਤਾਂ ਪਿਛਲੇ ਪਾਠਾਂ ਵਿੱਚ ਸਿੱਖੀਆਂ ਗਈਆਂ ਵਰਗੀਕਰਨ ਤਕਨੀਕਾਂ ਜ਼ਿਆਦਾ ਲਾਭਦਾਇਕ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਪਰ ਜਦੋਂ ਤੁਸੀਂ ਬਿਨਾਂ ਲੇਬਲ ਵਾਲੇ ਡਾਟੇ ਨੂੰ ਸਮੂਹਬੱਧ ਕਰਨ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰ ਰਹੇ ਹੋ, ਤਾਂ ਕਲੱਸਟਰਿੰਗ ਪੈਟਰਨ ਲੱਭਣ ਦਾ ਇੱਕ ਵਧੀਆ ਤਰੀਕਾ ਹੈ।

> ਕੁਝ ਲੋ-ਕੋਡ ਟੂਲ ਲਾਭਦਾਇਕ ਹੋ ਸਕਦੇ ਹਨ ਜੋ ਤੁਹਾਨੂੰ ਕਲੱਸਟਰਿੰਗ ਮਾਡਲਾਂ ਨਾਲ ਕੰਮ ਕਰਨ ਬਾਰੇ ਸਿੱਖਣ ਵਿੱਚ ਮਦਦ ਕਰਦੇ ਹਨ। ਇਸ ਕੰਮ ਲਈ [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ।

## ਪਾਠ

1. [ਕਲੱਸਟਰਿੰਗ ਦਾ ਪਰਿਚਯ](1-Visualize/README.md)
2. [ਕੇ-ਮੀਨਜ਼ ਕਲੱਸਟਰਿੰਗ](2-K-Means/README.md)

## ਸ਼੍ਰੇਯ

ਇਹ ਪਾਠ 🎶 ਨਾਲ [ਜੈਨ ਲੂਪਰ](https://www.twitter.com/jenlooper) ਦੁਆਰਾ ਲਿਖੇ ਗਏ ਹਨ, ਅਤੇ [ਰਿਸ਼ਿਤ ਦਾਗਲੀ](https://rishit_dagli) ਅਤੇ [ਮੁਹੰਮਦ ਸਾਕਿਬ ਖਾਨ ਇਨਾਨ](https://twitter.com/Sakibinan) ਦੁਆਰਾ ਮਦਦਗਾਰ ਸਮੀਖਿਆਵਾਂ ਕੀਤੀਆਂ ਗਈਆਂ ਹਨ।

[ਨਾਈਜੀਰੀਆਈ ਗੀਤਾਂ](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ਡਾਟਾਸੈਟ Kaggle ਤੋਂ Spotify ਤੋਂ ਇਕੱਠਾ ਕੀਤਾ ਗਿਆ ਸੀ।

ਉਪਯੋਗ ਕੇ-ਮੀਨਜ਼ ਉਦਾਹਰਣਾਂ ਜਿਨ੍ਹਾਂ ਨੇ ਇਸ ਪਾਠ ਨੂੰ ਬਣਾਉਣ ਵਿੱਚ ਮਦਦ ਕੀਤੀ, ਵਿੱਚ ਸ਼ਾਮਲ ਹਨ ਇਹ [ਆਇਰਿਸ ਐਕਸਪਲੋਰੇਸ਼ਨ](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ਇਹ [ਪਰਿਚਯਾਤਮਕ ਨੋਟਬੁੱਕ](https://www.kaggle.com/prashant111/k-means-clustering-with-python), ਅਤੇ ਇਹ [ਕਲਪਨਾਤਮਕ NGO ਉਦਾਹਰਣ](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)।

---

**ਅਸਵੀਕਤੀ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀਤਾ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚਤਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।