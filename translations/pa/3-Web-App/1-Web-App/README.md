<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-06T07:07:24+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "pa"
}
-->
# ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲ ਵਰਤਣ ਲਈ ਵੈੱਬ ਐਪ ਬਣਾਓ

ਇਸ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਇੱਕ ਡਾਟਾ ਸੈੱਟ 'ਤੇ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲ ਨੂੰ ਟ੍ਰੇਨ ਕਰੋਗੇ ਜੋ ਬਹੁਤ ਹੀ ਦਿਲਚਸਪ ਹੈ: _ਪਿਛਲੇ ਸਦੀ ਦੇ ਦੌਰਾਨ ਦੇਖੇ ਗਏ UFO_, ਜੋ ਕਿ NUFORC ਦੇ ਡਾਟਾਬੇਸ ਤੋਂ ਲਿਆ ਗਿਆ ਹੈ।

ਤੁਸੀਂ ਸਿੱਖੋਗੇ:

- ਟ੍ਰੇਨ ਕੀਤੇ ਮਾਡਲ ਨੂੰ 'pickle' ਕਿਵੇਂ ਕਰਨਾ ਹੈ
- ਉਸ ਮਾਡਲ ਨੂੰ Flask ਐਪ ਵਿੱਚ ਕਿਵੇਂ ਵਰਤਣਾ ਹੈ

ਅਸੀਂ ਨੋਟਬੁੱਕਸ ਦੀ ਵਰਤੋਂ ਜਾਰੀ ਰੱਖਾਂਗੇ ਡਾਟਾ ਸਾਫ ਕਰਨ ਅਤੇ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰਨ ਲਈ, ਪਰ ਤੁਸੀਂ ਇਸ ਪ੍ਰਕਿਰਿਆ ਨੂੰ ਇੱਕ ਕਦਮ ਅੱਗੇ ਲੈ ਜਾ ਸਕਦੇ ਹੋ ਅਤੇ ਮਾਡਲ ਨੂੰ 'ਜੰਗਲੀ' ਤੌਰ 'ਤੇ ਵਰਤਣ ਦੀ ਖੋਜ ਕਰ ਸਕਦੇ ਹੋ, ਜਿਵੇਂ ਕਿ ਇੱਕ ਵੈੱਬ ਐਪ ਵਿੱਚ।

ਇਹ ਕਰਨ ਲਈ, ਤੁਹਾਨੂੰ Flask ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਇੱਕ ਵੈੱਬ ਐਪ ਬਣਾਉਣ ਦੀ ਲੋੜ ਹੈ।

## [ਪਾਠ-ਪਹਿਲਾਂ ਕਵੀਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਐਪ ਬਣਾਉਣਾ

ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲਾਂ ਨੂੰ ਵਰਤਣ ਲਈ ਵੈੱਬ ਐਪ ਬਣਾਉਣ ਦੇ ਕਈ ਤਰੀਕੇ ਹਨ। ਤੁਹਾਡੀ ਵੈੱਬ ਆਰਕੀਟੈਕਚਰ ਇਸ ਗੱਲ ਨੂੰ ਪ੍ਰਭਾਵਿਤ ਕਰ ਸਕਦੀ ਹੈ ਕਿ ਤੁਹਾਡਾ ਮਾਡਲ ਕਿਵੇਂ ਟ੍ਰੇਨ ਕੀਤਾ ਗਿਆ ਹੈ। ਕਲਪਨਾ ਕਰੋ ਕਿ ਤੁਸੀਂ ਇੱਕ ਕਾਰੋਬਾਰ ਵਿੱਚ ਕੰਮ ਕਰ ਰਹੇ ਹੋ ਜਿੱਥੇ ਡਾਟਾ ਸਾਇੰਸ ਗਰੁੱਪ ਨੇ ਇੱਕ ਮਾਡਲ ਟ੍ਰੇਨ ਕੀਤਾ ਹੈ ਜੋ ਉਹ ਚਾਹੁੰਦੇ ਹਨ ਕਿ ਤੁਸੀਂ ਇੱਕ ਐਪ ਵਿੱਚ ਵਰਤੋ।

### ਵਿਚਾਰ

ਤੁਹਾਨੂੰ ਕਈ ਸਵਾਲ ਪੁੱਛਣ ਦੀ ਲੋੜ ਹੈ:

- **ਕੀ ਇਹ ਵੈੱਬ ਐਪ ਹੈ ਜਾਂ ਮੋਬਾਈਲ ਐਪ?** ਜੇ ਤੁਸੀਂ ਮੋਬਾਈਲ ਐਪ ਬਣਾ ਰਹੇ ਹੋ ਜਾਂ IoT ਸੰਦਰਭ ਵਿੱਚ ਮਾਡਲ ਦੀ ਵਰਤੋਂ ਕਰਨ ਦੀ ਲੋੜ ਹੈ, ਤਾਂ ਤੁਸੀਂ [TensorFlow Lite](https://www.tensorflow.org/lite/) ਦੀ ਵਰਤੋਂ ਕਰ ਸਕਦੇ ਹੋ ਅਤੇ ਮਾਡਲ ਨੂੰ Android ਜਾਂ iOS ਐਪ ਵਿੱਚ ਵਰਤ ਸਕਦੇ ਹੋ।
- **ਮਾਡਲ ਕਿੱਥੇ ਹੋਵੇਗਾ?** ਕਲਾਉਡ ਵਿੱਚ ਜਾਂ ਲੋਕਲ ਤੌਰ 'ਤੇ?
- **ਆਫਲਾਈਨ ਸਹਾਇਤਾ।** ਕੀ ਐਪ ਨੂੰ ਆਫਲਾਈਨ ਕੰਮ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ?
- **ਮਾਡਲ ਟ੍ਰੇਨ ਕਰਨ ਲਈ ਕਿਹੜੀ ਤਕਨਾਲੋਜੀ ਵਰਤੀ ਗਈ ਸੀ?** ਚੁਣੀ ਗਈ ਤਕਨਾਲੋਜੀ ਤੁਹਾਨੂੰ ਵਰਤਣ ਵਾਲੇ ਟੂਲਿੰਗ ਨੂੰ ਪ੍ਰਭਾਵਿਤ ਕਰ ਸਕਦੀ ਹੈ।
    - **TensorFlow ਦੀ ਵਰਤੋਂ।** ਜੇ ਤੁਸੀਂ TensorFlow ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰ ਰਹੇ ਹੋ, ਉਦਾਹਰਣ ਲਈ, ਉਹ ਪਰਿਸਰ [TensorFlow.js](https://www.tensorflow.org/js/) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਵੈੱਬ ਐਪ ਵਿੱਚ ਵਰਤਣ ਲਈ ਇੱਕ TensorFlow ਮਾਡਲ ਨੂੰ ਕਨਵਰਟ ਕਰਨ ਦੀ ਸਮਰੱਥਾ ਪ੍ਰਦਾਨ ਕਰਦਾ ਹੈ।
    - **PyTorch ਦੀ ਵਰਤੋਂ।** ਜੇ ਤੁਸੀਂ [PyTorch](https://pytorch.org/) ਵਰਗੇ ਲਾਇਬ੍ਰੇਰੀ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਮਾਡਲ ਬਣਾ ਰਹੇ ਹੋ, ਤਾਂ ਤੁਹਾਡੇ ਕੋਲ ਇਸਨੂੰ [ONNX](https://onnx.ai/) (Open Neural Network Exchange) ਫਾਰਮੈਟ ਵਿੱਚ ਐਕਸਪੋਰਟ ਕਰਨ ਦਾ ਵਿਕਲਪ ਹੈ, ਜਿਸਨੂੰ ਜਾਵਾਸਕ੍ਰਿਪਟ ਵੈੱਬ ਐਪ ਵਿੱਚ ਵਰਤਿਆ ਜਾ ਸਕਦਾ ਹੈ ਜੋ [Onnx Runtime](https://www.onnxruntime.ai/) ਦੀ ਵਰਤੋਂ ਕਰਦਾ ਹੈ। ਇਸ ਵਿਕਲਪ ਦੀ ਖੋਜ ਅਗਲੇ ਪਾਠ ਵਿੱਚ ਕੀਤੀ ਜਾਵੇਗੀ ਜਿੱਥੇ Scikit-learn-ਟ੍ਰੇਨ ਮਾਡਲ ਦੀ ਵਰਤੋਂ ਕੀਤੀ ਜਾਵੇਗੀ।
    - **Lobe.ai ਜਾਂ Azure Custom Vision ਦੀ ਵਰਤੋਂ।** ਜੇ ਤੁਸੀਂ [Lobe.ai](https://lobe.ai/) ਜਾਂ [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) ਵਰਗੇ ML SaaS (Software as a Service) ਸਿਸਟਮ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰ ਰਹੇ ਹੋ, ਤਾਂ ਇਸ ਤਰ੍ਹਾਂ ਦਾ ਸੌਫਟਵੇਅਰ ਕਈ ਪਲੇਟਫਾਰਮਾਂ ਲਈ ਮਾਡਲ ਐਕਸਪੋਰਟ ਕਰਨ ਦੇ ਤਰੀਕੇ ਪ੍ਰਦਾਨ ਕਰਦਾ ਹੈ, ਜਿਸ ਵਿੱਚ ਕਲਾਉਡ ਵਿੱਚ ਤੁਹਾਡੇ ਆਨਲਾਈਨ ਐਪਲੀਕੇਸ਼ਨ ਦੁਆਰਾ ਪੁੱਛੇ ਜਾਣ ਵਾਲੇ ਬੇਸਪੋਕ API ਬਣਾਉਣਾ ਸ਼ਾਮਲ ਹੈ।

ਤੁਹਾਡੇ ਕੋਲ ਪੂਰੇ Flask ਵੈੱਬ ਐਪ ਨੂੰ ਬਣਾਉਣ ਦਾ ਮੌਕਾ ਵੀ ਹੈ ਜੋ ਵੈੱਬ ਬ੍ਰਾਊਜ਼ਰ ਵਿੱਚ ਖੁਦ ਮਾਡਲ ਨੂੰ ਟ੍ਰੇਨ ਕਰ ਸਕਦਾ ਹੈ। ਇਹ ਕੰਮ ਜਾਵਾਸਕ੍ਰਿਪਟ ਸੰਦਰਭ ਵਿੱਚ TensorFlow.js ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਵੀ ਕੀਤਾ ਜਾ ਸਕਦਾ ਹੈ।

ਸਾਡੇ ਮਕਸਦ ਲਈ, ਕਿਉਂਕਿ ਅਸੀਂ Python-ਅਧਾਰਿਤ ਨੋਟਬੁੱਕਸ ਨਾਲ ਕੰਮ ਕਰ ਰਹੇ ਹਾਂ, ਆਓ ਉਹ ਕਦਮ ਖੋਜੀਏ ਜੋ ਤੁਹਾਨੂੰ ਇੱਕ ਟ੍ਰੇਨ ਕੀਤੇ ਮਾਡਲ ਨੂੰ ਨੋਟਬੁੱਕ ਤੋਂ Python-ਨਿਰਮਿਤ ਵੈੱਬ ਐਪ ਦੁਆਰਾ ਪੜ੍ਹਨਯੋਗ ਫਾਰਮੈਟ ਵਿੱਚ ਐਕਸਪੋਰਟ ਕਰਨ ਲਈ ਲੈਣ ਦੀ ਲੋੜ ਹੈ।

## ਟੂਲ

ਇਸ ਕੰਮ ਲਈ, ਤੁਹਾਨੂੰ ਦੋ ਟੂਲਾਂ ਦੀ ਲੋੜ ਹੈ: Flask ਅਤੇ Pickle, ਜੋ ਦੋਵੇਂ Python 'ਤੇ ਚਲਦੇ ਹਨ।

✅ [Flask](https://palletsprojects.com/p/flask/) ਕੀ ਹੈ? ਇਸਦੇ ਨਿਰਮਾਤਾਵਾਂ ਦੁਆਰਾ 'ਮਾਈਕ੍ਰੋ-ਫ੍ਰੇਮਵਰਕ' ਵਜੋਂ ਪਰਿਭਾਸ਼ਿਤ, Flask ਵੈੱਬ ਫ੍ਰੇਮਵਰਕਸ ਦੀ ਮੁੱਢਲੀ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ ਪ੍ਰਦਾਨ ਕਰਦਾ ਹੈ ਜੋ Python ਅਤੇ ਇੱਕ ਟੈਂਪਲੇਟਿੰਗ ਇੰਜਣ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਵੈੱਬ ਪੰਨਿਆਂ ਨੂੰ ਬਣਾਉਂਦਾ ਹੈ। [ਇਸ ਲਰਨ ਮਾਡਿਊਲ](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) ਨੂੰ ਵੇਖੋ Flask ਨਾਲ ਬਣਾਉਣ ਦੀ ਅਭਿਆਸ ਕਰਨ ਲਈ।

✅ [Pickle](https://docs.python.org/3/library/pickle.html) ਕੀ ਹੈ? Pickle 🥒 ਇੱਕ Python ਮੋਡੀਊਲ ਹੈ ਜੋ Python ਆਬਜੈਕਟ ਸਟ੍ਰਕਚਰ ਨੂੰ ਸੀਰੀਅਲਾਈਜ਼ ਅਤੇ ਡੀ-ਸੀਰੀਅਲਾਈਜ਼ ਕਰਦਾ ਹੈ। ਜਦੋਂ ਤੁਸੀਂ ਮਾਡਲ ਨੂੰ 'pickle' ਕਰਦੇ ਹੋ, ਤੁਸੀਂ ਇਸਦੀ ਸਟ੍ਰਕਚਰ ਨੂੰ ਵੈੱਬ 'ਤੇ ਵਰਤਣ ਲਈ ਸੀਰੀਅਲਾਈਜ਼ ਜਾਂ ਫਲੈਟ ਕਰਦੇ ਹੋ। ਧਿਆਨ ਰੱਖੋ: pickle ਅੰਦਰੂਨੀ ਤੌਰ 'ਤੇ ਸੁਰੱਖਿਅਤ ਨਹੀਂ ਹੈ, ਇਸ ਲਈ ਜੇ ਕਿਸੇ ਫਾਈਲ ਨੂੰ 'un-pickle' ਕਰਨ ਲਈ ਕਿਹਾ ਜਾਵੇ ਤਾਂ ਸਾਵਧਾਨ ਰਹੋ। ਇੱਕ pickled ਫਾਈਲ ਦਾ ਸੁਫਿਕਸ `.pkl` ਹੁੰਦਾ ਹੈ।

## ਅਭਿਆਸ - ਆਪਣਾ ਡਾਟਾ ਸਾਫ ਕਰੋ

ਇਸ ਪਾਠ ਵਿੱਚ ਤੁਸੀਂ 80,000 UFO ਦੇਖਣਾਂ ਦੇ ਡਾਟਾ ਦੀ ਵਰਤੋਂ ਕਰੋਗੇ, ਜੋ ਕਿ [NUFORC](https://nuforc.org) (The National UFO Reporting Center) ਦੁਆਰਾ ਇਕੱਠਾ ਕੀਤਾ ਗਿਆ ਹੈ। ਇਸ ਡਾਟਾ ਵਿੱਚ UFO ਦੇਖਣਾਂ ਦੇ ਕੁਝ ਦਿਲਚਸਪ ਵਰਣਨ ਹਨ, ਉਦਾਹਰਣ ਲਈ:

- **ਲੰਬਾ ਵਰਣਨ।** "ਇੱਕ ਆਦਮੀ ਰਾਤ ਨੂੰ ਇੱਕ ਘਾਸ ਵਾਲੇ ਖੇਤਰ 'ਤੇ ਚਮਕ ਰਹੀ ਰੌਸ਼ਨੀ ਦੀ ਕਿਰਣ ਤੋਂ ਬਾਹਰ ਆਉਂਦਾ ਹੈ ਅਤੇ ਉਹ Texas Instruments ਦੀ ਪਾਰਕਿੰਗ ਲਾਟ ਵੱਲ ਦੌੜਦਾ ਹੈ।"
- **ਛੋਟਾ ਵਰਣਨ।** "ਰੌਸ਼ਨੀ ਸਾਡੇ ਪਿੱਛੇ ਆਈ।"

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) ਸਪ੍ਰੈਡਸ਼ੀਟ ਵਿੱਚ ਉਹ ਕਾਲਮ ਸ਼ਾਮਲ ਹਨ ਜਿੱਥੇ ਦੇਖਣ ਹੋਏ, ਜਿਵੇਂ `city`, `state`, ਅਤੇ `country`, ਆਬਜੈਕਟ ਦਾ `shape`, ਅਤੇ ਇਸਦਾ `latitude` ਅਤੇ `longitude`।

ਖਾਲੀ [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) ਵਿੱਚ:

1. ਪਿਛਲੇ ਪਾਠਾਂ ਵਿੱਚ ਵਰਤੇ ਗਏ ਤਰੀਕੇ ਨਾਲ `pandas`, `matplotlib`, ਅਤੇ `numpy` ਨੂੰ ਇੰਪੋਰਟ ਕਰੋ ਅਤੇ ufos ਸਪ੍ਰੈਡਸ਼ੀਟ ਨੂੰ ਇੰਪੋਰਟ ਕਰੋ। ਤੁਸੀਂ ਡਾਟਾ ਸੈੱਟ ਦਾ ਨਮੂਨਾ ਵੇਖ ਸਕਦੇ ਹੋ:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos ਡਾਟਾ ਨੂੰ ਨਵੇਂ ਸਿਰਲੇਖਾਂ ਨਾਲ ਇੱਕ ਛੋਟੇ ਡਾਟਾ ਫ੍ਰੇਮ ਵਿੱਚ ਕਨਵਰਟ ਕਰੋ। `Country` ਫੀਲਡ ਵਿੱਚ ਵਿਲੱਖਣ ਮੁੱਲਾਂ ਦੀ ਜਾਂਚ ਕਰੋ।

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. ਹੁਣ, ਤੁਸੀਂ ਸਾਡੇ ਲਈ ਲੋੜੀਂਦੇ ਡਾਟਾ ਦੀ ਮਾਤਰਾ ਨੂੰ ਘਟਾ ਸਕਦੇ ਹੋ, ਕਿਸੇ ਵੀ null ਮੁੱਲਾਂ ਨੂੰ ਹਟਾ ਕੇ ਅਤੇ ਸਿਰਫ 1-60 ਸਕਿੰਟ ਦੇ ਵਿਚਕਾਰ ਦੇਖਣਾਂ ਨੂੰ ਇੰਪੋਰਟ ਕਰਕੇ:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn ਦੀ `LabelEncoder` ਲਾਇਬ੍ਰੇਰੀ ਨੂੰ ਇੰਪੋਰਟ ਕਰੋ ਤਾਂ ਜੋ ਦੇਸ਼ਾਂ ਲਈ ਟੈਕਸਟ ਮੁੱਲਾਂ ਨੂੰ ਨੰਬਰਾਂ ਵਿੱਚ ਕਨਵਰਟ ਕੀਤਾ ਜਾ ਸਕੇ:

    ✅ LabelEncoder ਡਾਟਾ ਨੂੰ ਵਰਣਮਾਲਾ ਅਨੁਸਾਰ ਕੋਡ ਕਰਦਾ ਹੈ

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    ਤੁਹਾਡਾ ਡਾਟਾ ਇਸ ਤਰ੍ਹਾਂ ਦਿਖਾਈ ਦੇਣਾ ਚਾਹੀਦਾ ਹੈ:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## ਅਭਿਆਸ - ਆਪਣਾ ਮਾਡਲ ਬਣਾਓ

ਹੁਣ ਤੁਸੀਂ ਡਾਟਾ ਨੂੰ ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟਿੰਗ ਗਰੁੱਪ ਵਿੱਚ ਵੰਡ ਕੇ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰਨ ਲਈ ਤਿਆਰ ਹੋ।

1. ਉਹ ਤਿੰਨ ਫੀਚਰ ਚੁਣੋ ਜਿਨ੍ਹਾਂ 'ਤੇ ਤੁਸੀਂ ਟ੍ਰੇਨ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ ਆਪਣੇ X ਵੇਕਟਰ ਵਜੋਂ, ਅਤੇ y ਵੇਕਟਰ `Country` ਹੋਵੇਗਾ। ਤੁਸੀਂ ਚਾਹੁੰਦੇ ਹੋ ਕਿ ਤੁਸੀਂ `Seconds`, `Latitude`, ਅਤੇ `Longitude` ਦਾਖਲ ਕਰੋ ਅਤੇ ਇੱਕ ਦੇਸ਼ ਦਾ ਕੋਡ ਪ੍ਰਾਪਤ ਕਰੋ।

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਆਪਣਾ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰੋ:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

ਸਹੀਤਾ ਬੁਰੀ ਨਹੀਂ ਹੈ **(ਲਗਭਗ 95%)**, ਜੋ ਹੈਰਾਨੀਜਨਕ ਨਹੀਂ ਹੈ, ਕਿਉਂਕਿ `Country` ਅਤੇ `Latitude/Longitude` ਸੰਬੰਧਿਤ ਹਨ।

ਤੁਹਾਡੇ ਦੁਆਰਾ ਬਣਾਇਆ ਗਿਆ ਮਾਡਲ ਬਹੁਤ ਵੱਖਰਾ ਨਹੀਂ ਹੈ ਕਿਉਂਕਿ ਤੁਸੀਂ `Latitude` ਅਤੇ `Longitude` ਤੋਂ ਇੱਕ `Country` ਦਾ ਅਨੁਮਾਨ ਲਗਾ ਸਕਦੇ ਹੋ, ਪਰ ਇਹ ਇੱਕ ਵਧੀਆ ਅਭਿਆਸ ਹੈ ਕੱਚੇ ਡਾਟਾ ਤੋਂ ਟ੍ਰੇਨ ਕਰਨ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਨ ਲਈ, ਜਿਸਨੂੰ ਤੁਸੀਂ ਸਾਫ ਕੀਤਾ, ਐਕਸਪੋਰਟ ਕੀਤਾ, ਅਤੇ ਫਿਰ ਇਸ ਮਾਡਲ ਨੂੰ ਇੱਕ ਵੈੱਬ ਐਪ ਵਿੱਚ ਵਰਤਿਆ।

## ਅਭਿਆਸ - ਆਪਣੇ ਮਾਡਲ ਨੂੰ 'pickle' ਕਰੋ

ਹੁਣ, ਸਮਾਂ ਆ ਗਿਆ ਹੈ ਕਿ ਤੁਸੀਂ ਆਪਣੇ ਮਾਡਲ ਨੂੰ _pickle_ ਕਰੋ! ਤੁਸੀਂ ਇਹ ਕੁਝ ਲਾਈਨਾਂ ਦੇ ਕੋਡ ਵਿੱਚ ਕਰ ਸਕਦੇ ਹੋ। ਜਦੋਂ ਇਹ _pickled_ ਹੋ ਜਾਵੇ, ਆਪਣੇ pickled ਮਾਡਲ ਨੂੰ ਲੋਡ ਕਰੋ ਅਤੇ ਇਸਨੂੰ ਸਕਿੰਟਾਂ, ਲੈਟੀਟਿਊਡ ਅਤੇ ਲੌਂਗਿਟਿਊਡ ਲਈ ਮੁੱਲਾਂ ਵਾਲੇ ਨਮੂਨਾ ਡਾਟਾ ਐਰੇ ਦੇ ਖਿਲਾਫ ਟੈਸਟ ਕਰੋ,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

ਮਾਡਲ **'3'** ਵਾਪਸ ਕਰਦਾ ਹੈ, ਜੋ ਕਿ ਯੂਕੇ ਲਈ ਦੇਸ਼ ਕੋਡ ਹੈ। ਹੈਰਾਨੀਜਨਕ! 👽

## ਅਭਿਆਸ - Flask ਐਪ ਬਣਾਓ

ਹੁਣ ਤੁਸੀਂ ਇੱਕ Flask ਐਪ ਬਣਾ ਸਕਦੇ ਹੋ ਜੋ ਤੁਹਾਡੇ ਮਾਡਲ ਨੂੰ ਕਾਲ ਕਰੇ ਅਤੇ ਇਸੇ ਤਰ੍ਹਾਂ ਦੇ ਨਤੀਜੇ ਵਾਪਸ ਕਰੇ, ਪਰ ਇੱਕ ਹੋਰ ਦ੍ਰਿਸ਼ਟੀਗੋਚੀ ਤਰੀਕੇ ਨਾਲ।

1. _notebook.ipynb_ ਫਾਈਲ ਦੇ ਕੋਲ ਇੱਕ ਫੋਲਡਰ **web-app** ਬਣਾਓ ਜਿੱਥੇ ਤੁਹਾਡੀ _ufo-model.pkl_ ਫਾਈਲ ਮੌਜੂਦ ਹੈ।

1. ਉਸ ਫੋਲਡਰ ਵਿੱਚ ਤਿੰਨ ਹੋਰ ਫੋਲਡਰ ਬਣਾਓ: **static**, ਜਿਸ ਵਿੱਚ ਇੱਕ ਫੋਲਡਰ **css** ਹੈ, ਅਤੇ **templates**। ਹੁਣ ਤੁਹਾਡੇ ਕੋਲ ਹੇਠਾਂ ਦਿੱਤੇ ਫਾਈਲਾਂ ਅਤੇ ਡਾਇਰੈਕਟਰੀਆਂ ਹੋਣੀਆਂ ਚਾਹੀਦੀਆਂ ਹਨ:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ ਤਿਆਰ ਐਪ ਦੇ ਦ੍ਰਿਸ਼ਟੀਕੋਣ ਲਈ ਹੱਲ ਫੋਲਡਰ ਨੂੰ ਵੇਖੋ

1. _web-app_ ਫੋਲਡਰ ਵਿੱਚ ਬਣਾਉਣ ਲਈ ਪਹਿਲੀ ਫਾਈਲ **requirements.txt** ਹੈ। ਜਿਵੇਂ ਕਿ ਜਾਵਾਸਕ੍ਰਿਪਟ ਐਪ ਵਿੱਚ _package.json_, ਇਹ ਫਾਈਲ ਐਪ ਦੁਆਰਾ ਲੋੜੀਂਦੇ ਡਿਪੈਂਡੈਂਸੀਜ਼ ਦੀ ਸੂਚੀ ਦਿੰਦੀ ਹੈ। **requirements.txt** ਵਿੱਚ ਹੇਠਾਂ ਦਿੱਤੀਆਂ ਲਾਈਨਾਂ ਸ਼ਾਮਲ ਕਰੋ:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. ਹੁਣ, ਇਸ ਫਾਈਲ ਨੂੰ _web-app_ ਵਿੱਚ ਨੈਵੀਗੇਟ ਕਰਕੇ ਚਲਾਓ:

    ```bash
    cd web-app
    ```

1. ਆਪਣੇ ਟਰਮੀਨਲ ਵਿੱਚ `pip install` ਟਾਈਪ ਕਰੋ, ਤਾਂ ਜੋ _requirements.txt_ ਵਿੱਚ ਸੂਚੀਬੱਧ ਲਾਇਬ੍ਰੇਰੀਆਂ ਨੂੰ ਇੰਸਟਾਲ ਕੀਤਾ ਜਾ ਸਕੇ:

    ```bash
    pip install -r requirements.txt
    ```

1. ਹੁਣ, ਤੁਸੀਂ ਐਪ ਨੂੰ ਪੂਰਾ ਕਰਨ ਲਈ ਤਿੰਨ ਹੋਰ ਫਾਈਲਾਂ ਬਣਾਉਣ ਲਈ ਤਿਆਰ ਹੋ:

    1. **app.py** ਨੂੰ ਰੂਟ ਵਿੱਚ ਬਣਾਓ।
    2. _templates_ ਡਾਇਰੈਕਟਰੀ ਵਿੱਚ **index.html** ਬਣਾਓ।
    3. _static/css_ ਡਾਇਰੈਕਟਰੀ ਵਿੱਚ **styles.css** ਬਣਾਓ।

1. _styles.css_ ਫਾਈਲ ਵਿੱਚ ਕੁਝ ਸਟਾਈਲਾਂ ਸ਼ਾਮਲ ਕਰੋ:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. ਅਗਲੇ ਕਦਮ ਵਿੱਚ, _index.html_ ਫਾਈਲ ਬਣਾਓ:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    ਇਸ ਫਾਈਲ ਵਿੱਚ ਟੈਂਪਲੇਟਿੰਗ ਨੂੰ ਵੇਖੋ। ਧਿਆਨ ਦਿਓ ਕਿ ਇਸ ਵਿੱਚ ਮਾਡਲ ਦੁਆਰਾ ਪ੍ਰਦਾਨ ਕੀਤੇ ਜਾਣ ਵਾਲੇ ਵੈਰੀਏਬਲਾਂ ਲਈ 'mustache' ਸਿੰਟੈਕਸ ਵਰਤਿਆ ਗਿਆ ਹੈ, ਜਿਵੇਂ ਕਿ ਪ੍ਰਡਿਕਸ਼ਨ ਟੈਕਸਟ: `{{}}`। ਇੱਥੇ ਇੱਕ ਫਾਰਮ ਵੀ ਹੈ ਜੋ `/predict` ਰੂਟ ਨੂੰ ਪ੍ਰਡਿਕਸ਼ਨ ਭੇਜਦਾ ਹੈ।

    ਅੰਤ ਵਿੱਚ, ਤੁਸੀਂ Python ਫਾਈਲ ਬਣਾਉਣ ਲਈ ਤਿਆਰ ਹੋ ਜੋ ਮਾਡਲ ਦੀ ਖਪਤ ਅਤੇ ਪ੍ਰਡਿਕਸ਼ਨ ਦੇ ਪ੍ਰਦਰਸ਼ਨ ਨੂੰ ਚਲਾਉਂਦੀ ਹੈ:

1. `app.py` ਵਿੱਚ ਸ਼ਾਮਲ ਕਰੋ:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 ਟਿੱਪ: ਜਦੋਂ ਤੁਸੀਂ Flask ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਵੈੱਬ ਐਪ ਚਲਾਉਂਦੇ ਹੋ, ਤਾਂ [debug=True](https://www.askpython.com/python-modules/flask/flask-debug-mode) ਸ਼ਾਮਲ ਕਰਨ 'ਤੇ, ਤੁਹਾਡੇ ਐਪਲੀਕੇਸ਼ਨ ਵਿੱਚ ਕੀਤੇ ਗਏ ਕਿਸੇ ਵੀ ਬਦਲਾਅ ਨੂੰ ਤੁਰੰਤ ਦਰਸਾਇਆ ਜਾਵੇਗਾ ਬਿਨਾਂ ਸਰਵਰ ਨੂੰ ਦੁਬਾਰਾ ਸ਼ੁਰੂ ਕਰਨ ਦੀ ਲੋੜ। ਸਾਵਧਾਨ ਰਹੋ! ਇਸ ਮੋਡ ਨੂੰ ਪ੍ਰੋਡਕਸ਼ਨ ਐਪ ਵਿੱਚ ਚਾਲੂ ਨਾ ਕਰੋ।

ਜੇ ਤੁਸੀਂ `python app.py` ਜਾਂ `python3 app.py` ਚਲਾਉਂਦੇ ਹੋ - ਤੁਹਾਡਾ ਵੈੱਬ ਸਰਵਰ ਸਥਾਨਕ ਤੌਰ 'ਤੇ ਸ਼ੁਰੂ ਹੋ ਜਾਂਦਾ ਹੈ, ਅਤੇ ਤੁਸੀਂ ਇੱਕ ਛੋਟਾ ਫਾਰਮ ਭਰ ਸਕਦੇ ਹੋ ਤਾਂ ਜੋ ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਮਿਲ ਸਕੇ ਕਿ UFO ਕਿੱਥੇ ਦੇਖੇ ਗਏ ਹਨ!

ਇਹ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ, `app.py` ਦੇ ਹਿੱਸਿਆਂ ਨੂੰ ਵੇਖੋ:

1. ਪਹਿਲਾਂ, ਡਿਪੈਂਡੈਂਸੀਜ਼ ਲੋਡ ਕੀਤੀਆਂ ਜਾਂਦੀਆਂ ਹਨ ਅਤੇ ਐਪ ਸ਼ੁਰੂ ਹੁੰਦਾ ਹੈ।
1. ਫਿਰ, ਮਾਡਲ ਇੰਪੋਰਟ ਕੀਤਾ ਜਾਂਦਾ ਹੈ।
1. ਫਿਰ, ਮੁੱਖ ਰੂਟ 'ਤੇ index.html ਰੈਂਡਰ ਕੀਤਾ ਜਾਂਦਾ ਹੈ।

`/predict` ਰੂਟ 'ਤੇ, ਜਦੋਂ ਫਾਰ

---

**ਅਸਵੀਕਾਰਨਾ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਹਾਲਾਂਕਿ ਅਸੀਂ ਸਹੀਅਤਾ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁੱਤੀਆਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤ ਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।