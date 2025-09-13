<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-08-29T17:44:46+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "pa"
}
-->
# ਆਪਣਾ ML ਮਾਡਲ ਵਰਤਣ ਲਈ ਇੱਕ ਵੈੱਬ ਐਪ ਬਣਾਓ

ਇਸ ਪਾਠਕ੍ਰਮ ਦੇ ਇਸ ਹਿੱਸੇ ਵਿੱਚ, ਤੁਹਾਨੂੰ ਇੱਕ ਲਾਗੂ ਕੀਤੀ ਗਈ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਵਿਸ਼ੇ ਨਾਲ ਜਾਣੂ ਕਰਵਾਇਆ ਜਾਵੇਗਾ: ਆਪਣੇ Scikit-learn ਮਾਡਲ ਨੂੰ ਇੱਕ ਫਾਈਲ ਵਜੋਂ ਕਿਵੇਂ ਸੇਵ ਕਰਨਾ ਹੈ ਜੋ ਵੈੱਬ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ ਅਨੁਮਾਨ ਲਗਾਉਣ ਲਈ ਵਰਤੀ ਜਾ ਸਕਦੀ ਹੈ। ਜਦੋਂ ਮਾਡਲ ਸੇਵ ਹੋ ਜਾਂਦਾ ਹੈ, ਤੁਸੀਂ ਸਿੱਖੋਗੇ ਕਿ ਇਸਨੂੰ Flask ਵਿੱਚ ਬਣਾਈ ਗਈ ਵੈੱਬ ਐਪ ਵਿੱਚ ਕਿਵੇਂ ਵਰਤਣਾ ਹੈ। ਸਭ ਤੋਂ ਪਹਿਲਾਂ, ਤੁਸੀਂ ਇੱਕ ਮਾਡਲ ਬਣਾਓਗੇ ਜੋ UFO ਦੇ ਨਜ਼ਾਰਿਆਂ ਬਾਰੇ ਕੁਝ ਡਾਟਾ ਵਰਤਦਾ ਹੈ! ਫਿਰ, ਤੁਸੀਂ ਇੱਕ ਵੈੱਬ ਐਪ ਬਣਾਓਗੇ ਜੋ ਤੁਹਾਨੂੰ ਸਕਿੰਟਾਂ ਦੀ ਗਿਣਤੀ, ਲੈਟੀਟਿਊਡ ਅਤੇ ਲੌਂਗਿਟਿਊਡ ਦੀ ਮੁੱਲ ਦਾਖਲ ਕਰਨ ਦੀ ਆਗਿਆ ਦੇਵੇਗਾ ਤਾਂ ਜੋ ਇਹ ਅਨੁਮਾਨ ਲਗਾਇਆ ਜਾ ਸਕੇ ਕਿ ਕਿਸ ਦੇਸ਼ ਨੇ UFO ਦੇਖਣ ਦੀ ਰਿਪੋਰਟ ਕੀਤੀ।

![UFO Parking](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.pa.jpg)

ਫੋਟੋ <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਮਾਈਕਲ ਹੇਰਨ</a> ਦੁਆਰਾ <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਅਨਸਪਲੈਸ਼</a> 'ਤੇ

## ਪਾਠ

1. [ਵੈੱਬ ਐਪ ਬਣਾਓ](1-Web-App/README.md)

## ਸ਼੍ਰੇਯ

"ਵੈੱਬ ਐਪ ਬਣਾਓ" ਨੂੰ [ਜੈਨ ਲੂਪਰ](https://twitter.com/jenlooper) ਦੁਆਰਾ ♥️ ਨਾਲ ਲਿਖਿਆ ਗਿਆ ਸੀ।

♥️ ਕਵਿਜ਼ ਰੋਹਨ ਰਾਜ ਦੁਆਰਾ ਲਿਖੇ ਗਏ ਸਨ।

ਡਾਟਾਸੈਟ [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) ਤੋਂ ਪ੍ਰਾਪਤ ਕੀਤਾ ਗਿਆ ਹੈ।

ਵੈੱਬ ਐਪ ਆਰਕੀਟੈਕਚਰ ਨੂੰ ਹਿੱਸੇ ਵਿੱਚ [ਇਸ ਲੇਖ](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) ਅਤੇ [ਇਸ ਰਿਪੋ](https://github.com/abhinavsagar/machine-learning-deployment) ਦੁਆਰਾ ਅਭਿਨਵ ਸਾਗਰ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਗਈ ਸੀ।

---

**ਅਸਵੀਕਤੀ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀਤਾ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚੀਤਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਇਸ ਦਸਤਾਵੇਜ਼ ਦਾ ਮੂਲ ਰੂਪ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।