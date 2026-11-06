# ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਲਈ ਕਲੱਸਟਰਿੰਗ ਮਾਡਲ

ਕਲੱਸਟਰਿੰਗ ਇੱਕ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਟਾਸਕ ਹੈ ਜਿਸ ਵਿੱਚ ਇਹ ਵੇਖਿਆ ਜਾਂਦਾ ਹੈ ਕਿ ਇੱਕ ਦੂਜੇ ਨਾਲ ਮਿਲਦੇ-ਜੁਲਦੇ ਵਸਤੂਆਂ ਨੂੰ ਲੱਭਣਾ ਅਤੇ ਉਹਨਾਂ ਨੂੰ ਕਲੱਸਟਰ ਨਾਮਕ ਗਰੁੱਪਾਂ ਵਿੱਚ ਵੰਡਣਾ। ਜੋ ਗੱਲ ਕਲੱਸਟਰਿੰਗ ਨੂੰ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦੇ ਹੋਰ ਤਰੀਕਿਆਂ ਤੋਂ ਵੱਖਰੀ ਬਣਾਉਂਦੀ ਹੈ, ਉਹ ਇਹ ਹੈ ਕਿ ਇਹ ਆਟੋਮੈਟਿਕ ਤੌਰ 'ਤੇ ਹੁੰਦੀ ਹੈ, ਦਰਅਸਲ, ਇਹ ਕਹਿਣ ਲਈ ਠੀਕ ਹੈ ਕਿ ਇਹ ਸਪਰਵਾਈਜ਼ਡ ਲਰਨਿੰਗ ਦਾ ਉਲਟ ਹੈ।

## ਖੇਤਰੀ ਵਿਸ਼ਾ: ਨਾਈਜੀਰੀਆਈ ਦਰਸ਼ਕਾਂ ਦੀ ਸੰਗੀਤਕ ਪਸੰਦ ਲਈ ਕਲੱਸਟਰਿੰਗ ਮਾਡਲ 🎧

ਨਾਈਜੀਰੀਆ ਦੇ ਵਿਭਿੰਨ ਦਰਸ਼ਕਾਂ ਦੀ ਸੰਗੀਤਕ ਪਸੰਦ ਵੀ ਵੱਖਰੀ ਵੱਖਰੀ ਹੈ। Spotify ਤੋਂ ਸਕ੍ਰੈਪ ਕੀਤੇ ਡੇਟਾ ਦੀ ਵਰਤੋਂ ਕਰਦਿਆਂ (ਇਸ ਲੇਖ ਤੋਂ ਪ੍ਰੇਰਿਤ [this article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), ਆਓ ਨਾਈਜੀਰੀਆ ਵਿੱਚ ਕੁਝ ਲੋਕਪਰੀਅ ਸੰਗੀਤ ਨੂੰ ਦੇਖੀਏ। ਇਸ ਡੇਟਾਸੈੱਟ ਵਿੱਚ ਵੱਖ-ਵੱਖ ਗਾਣਿਆਂ ਦੇ 'ਡਾਂਸੇਬਿਲਿਟੀ' ਸਕੋਰ, 'ਅਕੂਸਟਿਕਨੈੱਸ', ਸ਼ੋਰਗੁਲ, 'ਸਪੀਚੀਨੈੱਸ', ਪ੍ਰਸਿੱਧੀ ਅਤੇ ਊਰਜਾ ਬਾਰੇ ਡੇਟਾ ਸ਼ਾਮਲ ਹੈ। ਇਸ ਡੇਟਾ ਵਿੱਚ ਪੈਟਰਨ ਲੱਭਣਾ ਦਿਲਚਸਪ ਹੋਵੇਗਾ!

![ਇੱਕ ਟਰਨਟੇਬਲ](../../../translated_images/pa/turntable.f2b86b13c53302dc.webp)

> ਫੋਟੋ <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਮਾਰਸੇਲਾ ਲਾਸਕੋਸਕੀ</a> ਦੁਆਰਾ <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਅਨਸਪਲੈਸ਼</a> 'ਤੇ
  
ਇਸ ਲੜੀਵਾਰ ਪਾਠਾਂ ਵਿੱਚ, ਤੁਸੀਂ ਕਲੱਸਟਰਿੰਗ ਤਕਨੀਕਾਂ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਡੇਟਾ ਦਾ ਨਵਾਂ ਢੰਗ ਨਾਲ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਨਾ ਸਿੱਖੋਗੇ। ਜਦੋਂ ਤੁਹਾਡੇ ਡੇਟਾਸੈੱਟ ਵਿੱਚ ਲੇਬਲ ਨਹੀਂ ਹੁੰਦੇ, ਕਲੱਸਟਰਿੰਗ ਵਿਸ਼ੇਸ਼ ਤੌਰ 'ਤੇ ਲਾਭਦਾਇਕ ਹੈ। ਜੇ ਲੇਬਲ ਹੁੰਦੇ ਹਨ, ਤਾਂ ਪਿਛਲੇ ਪਾਠਾਂ ਵਿੱਚ ਸਿੱਖੀ ਗਈ ਵਰਗੀ ਕਲਾਸੀਫਿਕੇਸ਼ਨ ਤਕਨੀਕਾਂ ਜ਼ਿਆਦਾ ਲਾਭਦਾਇਕ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਪਰ ਜਦੋਂ ਤੁਸੀਂ ਅਣਲੇਬਲਡ ਡੇਟਾ ਨੂੰ ਗਰੁੱਪ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ, ਤਾਂ ਕਲੱਸਟਰਿੰਗ ਪੈਟਰਨ ਲੱਭਣ ਦਾ ਬਿਹਤਰ ਤਰੀਕਾ ਹੈ।

> ਕੁਝ ਲੋ-ਕੋਡ ਸੰਦ ਮੌਜੂਦ ਹਨ ਜੋ ਤੁਹਾਨੂੰ ਕਲੱਸਟਰਿੰਗ ਮਾਡਲਾਂ ਨਾਲ ਕੰਮ ਕਰਨ ਬਾਰੇ ਸਿੱਖਣ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕਦੇ ਹਨ। ਇਸ ਟਾਸਕ ਲਈ [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ

## ਪਾਠ

1. [ਕਲੱਸਟਰਿੰਗ ਦਾ ਪ੍ਰਿੰਤਭੂਤ ਪਰੀਚੈ](1-Visualize/README.md)
2. [ਕੇ-ਮੀਨਜ਼ ਕਲੱਸਟਰਿੰਗ](2-K-Means/README.md)

## ਸ਼ੁਕਰਾਨੇ

ਇਹ ਪਾਠ 🎶 ਦੀ ਸਹਾਇਤਾ ਨਾਲ [ਜੇਨ ਲੂਪਰ](https://www.twitter.com/jenlooper) ਵੱਲੋਂ ਲਿਖੇ ਗਏ ਹਨ ਜਿਨ੍ਹਾਂ ਨੂੰ [ਰਿਸ਼ਿਤ ਦਾਗਲੀ](https://rishit_dagli/) ਅਤੇ [ਮੁਹੰਮਦ ਸਾਕਿਬ ਖਾਨ ਇਨਾਨ](https://twitter.com/Sakibinan) ਦੀ ਮਦਦ ਨਾਲ ਸਮੀਖਿਆ ਮਿਲੀ ਹੈ।

[ਨਾਈਜੀਰੀਆਈ ਗਾਣੇ](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ਡੇਟਾਸੈੱਟ ਨੂੰ Kaggle ਤੋਂ, ਜੋ Spotify ਤੋਂ ਸਕ੍ਰੈਪ ਕੀਤਾ ਗਿਆ ਸੀ, ਪ੍ਰਾਪਤ ਕੀਤਾ ਗਿਆ।

ਕੇ-ਮੀਨਜ਼ ਦੇ ਕੁਝ ਲਾਭਦਾਇਕ ਉਦਾਹਰਨ ਜੋ ਇਸ ਪਾਠ ਨੂੰ ਬਣਾਉਣ ਵਿੱਚ ਮਦਦਗਾਰ ਸਾਬਤ ਹੋਈਆਂ, ਉਨ੍ਹਾਂ ਵਿੱਚ ਇਸ [ਆਇਰਿਸ ਖੋਜ](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ਇਹ [ਪ੍ਰਸਤਾਵਿਕ ਨੋਟਬੁੱਕ](https://www.kaggle.com/prashant111/k-means-clustering-with-python), ਅਤੇ ਇਹ [ਕਲਪਨਾਤਮਕ NGO ਉਦਾਹਰਨ](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) ਸ਼ਾਮਲ ਹਨ।

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ਅਸਵੀਕਾਰੋਪਣ**:
ਇਸ ਦਸਤਾਵੇਜ਼ ਦਾ ਅਨੁਵਾਦ ਏਆਈ ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀਤਾਵਾਂ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਰੱਖੋ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸਮੱਤਿਆਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਆਪਣੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਕ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਜਰੂਰੀ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫ਼ਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਅਸੀਂ ਇਸ ਅਨੁਵਾਦ ਦੇ ਉਪਯੋਗ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੀਆਂ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀਆਂ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆਵਾਂ ਲਈ ਜਵਾਬਦੇਹ ਨਹੀਂ ਹਾਂ।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->