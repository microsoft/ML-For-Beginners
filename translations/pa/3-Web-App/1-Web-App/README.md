# ਇਕ ML ਮਾਡਲ ਦੀ ਵਰਤੋਂ ਕਰਨ ਲਈ ਵੈੱਬ ਐਪ ਬਣਾਓ

ਇਸ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਇੱਕ ਐਸਾ ਡਾਟਾ ਸੈੱਟ ਤੇ ML ਮਾਡਲ ਸ਼ਿਕਸ਼ਿਤ ਕਰੋਗੇ ਜੋ ਦੁਨੀਆ ਤੋਂ ਬਾਹਰ ਹੈ: _ਪਿਛਲੇ ਸਦੀ ਵਿੱਚ UFO ਦ੍ਰਿਸ਼ਟੀਗੋਚਰ_, ਜੋ NUFORC ਦੇ ਡੇਟਾਬੇਸ ਤੋਂ ਲਿਆ ਗਿਆ ਹੈ।

ਤੁਸੀਂ ਸਿੱਖੋਗੇ:

- ਇੱਕ ਤਿਆਰ ਕੀਤੇ ਮਾਡਲ ਨੂੰ 'ਪਿਕਲ' ਕਿਵੇਂ ਕਰਨਾ ਹੈ
- ਉਸ ਮਾਡਲ ਨੂੰ ਇੱਕ Flask ਐਪ ਵਿੱਚ ਕਿਵੇਂ ਵਰਤਣਾ ਹੈ

ਅਸੀਂ ਡੇਟਾ ਸਾਫ਼ ਕਰਨ ਅਤੇ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰਨ ਲਈ ਨੋਟਬੁੱਕਸ ਦੀ ਵਰਤੋਂ ਜਾਰੀ ਰੱਖਾਂਗੇ, ਪਰ ਤੁਸੀਂ ਇਸ ਪ੍ਰਕਿਰਿਆ ਨੂੰ ਇੱਕ ਕਦਮ ਅੱਗੇ ਵਧਾ ਸਕਦੇ ਹੋ ਇਹ ਵੇਖ ਕੇ ਕਿ ਕਿਸ ਤਰ੍ਹਾਂ ਮਾਡਲ ਨੂੰ 'ਜੰਗਲ' ਵਿੱਚ ਵਰਤਿਆ ਜਾ ਸਕਦਾ ਹੈ: ਵੈੱਬ ਐਪ ਵਿੱਚ।

ਇਸ ਲਈ, ਤੁਹਾਨੂੰ Flask ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਇੱਕ ਵੈੱਬ ਐਪ ਬਣਾਉਣੀ ਪਵੇਗੀ।

## [ਪ੍ਰੀ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਐਪ ਬਣਾਉਣਾ

ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲਾਂ ਦੀ ਖਪਤ ਲਈ ਵੈੱਬ ਐਪ ਬਣਾਉਣ ਦੇ ਕਈ ਤਰੀਕੇ ਹਨ। ਤੁਹਾਡੀ ਵੈੱਬ ਆਰਕੀਟੈਕਚਰ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰਨ ਦੇ ਤਰੀਕੇ ਨੂੰ ਪ੍ਰਭਾਵਿਤ ਕਰ ਸਕਦੀ ਹੈ। ਸੋਚੋ ਕਿ ਤੁਸੀਂ ਉਸ ਕਾਰੋਬਾਰ ਵਿੱਚ ਕੰਮ ਕਰ ਰਹੇ ਹੋ ਜਿਥੇ ਡੇਟਾ ਸਾਇੰਸ ਗਰੁੱਪ ਨੇ ਇੱਕ ਮਾਡਲ ਟ੍ਰੇਨ ਕੀਤਾ ਹੈ ਜਿਸਦਾ ਤੁਸੀਂ ਐਪ ਵਿੱਚ ਉਪਯੋਗ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ।

### ਵਿਚਾਰ

ਤੁਹਾਨੂੰ ਕਈ ਸਵਾਲ ਪੁੱਛਣੇ ਚਾਹੀਦੇ ਹਨ:

- **ਕੀ ਇਹ ਵੈੱਬ ਐਪ ਹੈ ਜਾਂ ਮੋਬਾਈਲ ਐਪ?** ਜੇਕਰ ਤੁਸੀਂ ਮੋਬਾਈਲ ਐਪ ਬਣਾ ਰਹੇ ਹੋ ਜਾਂ ਮਾਡਲ ਨੂੰ IoT ਸੰਦਰਭ ਵਿੱਚ ਵਰਤਣਾ ਹੈ, ਤਾਂ ਤੁਸੀਂ [TensorFlow Lite](https://www.tensorflow.org/lite/) ਦੀ ਵਰਤੋਂ ਕਰ ਸਕਦੇ ਹੋ ਅਤੇ ਮਾਡਲ ਨੂੰ ਐਂਡਰਾਇਡ ਜਾਂ iOS ਐਪ ਵਿੱਚ ਵਰਤ ਸਕਦੇ ਹੋ।
- **ਮਾਡਲ ਕਿੱਥੇ ਰਹੇਗਾ?** ਕਲਾਊਡ ਵਿੱਚ ਜਾਂ ਲੋਕਲ ਹੀ।
- **ਆਫਲਾਇਨ ਸਹਿਯੋਗ।** ਕੀ ਐਪ ਨੂੰ ਆਫਲਾਇਨ ਕੰਮ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ?
- **ਮਾਡਲ ਟ੍ਰੇਨ ਕਰਨ ਲਈ ਕਿਸ ਤਕਨਾਲੋਜੀ ਦੀ ਵਰਤੋਂ ਕੀਤੀ ਗਈ ਸੀ?** ਚੁਣੀ ਹੋਈ ਤਕਨਾਲੋਜੀ ਤੁਹਾਨੂੰ ਲੋੜੀਂਦੇ ਟੂਲਾਂ ਨੂੰ ਪ੍ਰਭਾਵਿਤ ਕਰ ਸਕਦੀ ਹੈ।
    - **TensorFlow ਦੀ ਵਰਤੋਂ।** ਜੇ ਤੁਸੀਂ ਉਦਾਹਰਨ ਵਜੋਂ TensorFlow ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰ ਰਹੇ ਹੋ, ਤਾਂ ਉਹ ਇੱਕ ਐਸਾ ਈਕੋਸਿਸਟਮ ਮੁਹੱਈਆ ਕਰਦਾ ਹੈ ਜੋ ਮਾਡਲ ਨੂੰ ਵੈੱਬ ਐਪ ਵਿੱਚ ਵਰਤਣ ਲਈ [TensorFlow.js](https://www.tensorflow.org/js/) ਵਿੱਚ ਹਵਾਲੇ ਨਾਲ ਬਦਲ ਸਕਦਾ ਹੈ।
    - **PyTorch ਦੀ ਵਰਤੋਂ।** ਜੇ ਤੁਸੀਂ [PyTorch](https://pytorch.org/) ਵਰਗੇ ਲਾਇਬਰੇਰੀ ਨਾਲ ਮਾਡਲ ਬਣਾ ਰਹੇ ਹੋ, ਤਾਂ ਤੁਹਾਡੇ ਕੋਲ ਇਸਨੂੰ ਜਾਵਾਸਕ੍ਰਿਪਟ ਵੈੱਬ ਐਪਸ ਵਿੱਚ ਵਰਤਣ ਲਈ [ONNX](https://onnx.ai/) ਫਾਰਮੈਟ ਵਿੱਚ ਐਕਸਪੋਰਟ ਕਰਨ ਦਾ ਵਿਕਲਪ ਹੈ, ਜੋ [Onnx Runtime](https://www.onnxruntime.ai/) ਵਰਤ ਸਕਦਾ ਹੈ। ਇਹ ਵਿਕਲਪ ਭਵਿੱਖ ਦੇ ਪਾਠ ਵਿੱਚ Scikit-learn-ਟੀਟ ਮਾਡਲ ਲਈ ਖੋਜਿਆ ਜਾਵੇਗਾ।
    - **Lobe.ai ਜਾਂ Azure Custom Vision ਦੀ ਵਰਤੋਂ।** ਜੇ ਤੁਸੀਂ [Lobe.ai](https://lobe.ai/) ਜਾਂ [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) ਵਰਗੇ ML SaaS (ਸਾਫਟਵੇਅਰ ਐਜ਼ ਏ ਸਰਵਿਸ) ਪ੍ਰਣਾਲੀ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰ ਰਹੇ ਹੋ, ਤਾਂ ਇਹ ਸਾਫਟਵੇਅਰ ਬਹੁਤ ਸਾਰੀਆਂ ਪਲੇਟਫਾਰਮਾਂ ਲਈ ਮਾਡਲ ਐਕਸਪੋਰਟ ਕਰਨ ਦੇ ਤਰੀਕੇ ਮੁਹੱਈਆ ਕਰਵਾਉਂਦਾ ਹੈ, ਜਿਸ ਵਿੱਚ ਇੱਕ ਵਿਸ਼ੇਸ਼ API ਬਣਾਉਣਾ ਵੀ ਸ਼ਾਮਲ ਹੈ ਜੋ ਤੁਹਾਡੇ ਆਨਲਾਈਨ ਐਪਲੀਕੇਸ਼ਨ ਵੱਲੋਂ ਕਲਾਊਡ ਵਿੱਚ ਪੁੱਛੇ ਜਾ ਸਕਦਾ ਹੈ।

ਤੁਹਾਡੇ ਕੋਲ ਇੱਕ ਪੂਰੀ Flask ਵੈੱਬ ਐਪ ਬਣਾਉਣ ਦਾ ਵੀ ਮੌਕਾ ਹੈ ਜੋ ਖੁਦ ਬਰਾਊਜ਼ਰ ਵਿੱਚ ਮਾਡਲ ਨੂੰ ਟ੍ਰੇਨ ਕਰਨ ਯੋਗ ਹੋਵੇ। ਇਹ JavaScript ਸੰਦਰਭ ਵਿੱਚ TensorFlow.js ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਵੀ ਕੀਤਾ ਜਾ ਸਕਦਾ ਹੈ।

ਸਾਡੇ ਮਕਸਦ ਲਈ, ਕਿਉਂਕਿ ਅਸੀਂ Python-ਅਧਾਰਿਤ ਨੋਟਬੁੱਕਸ ਨਾਲ ਕੰਮ ਕਰ ਰਹੇ ਹਾਂ, ਆਓ ਉਹ ਕਦਮ ਵੇਖੀਏ ਜੋ ਤੁਹਾਨੂੰ ਇੱਕ ਤਿਆਰ ਕੀਤੇ ਮਾਡਲ ਨੂੰ ਨੋਟਬੁੱਕ ਤੋਂ Python ਬਣੀ ਵੈੱਬ ਐਪ ਵਿੱਚ ਪੱਠਿਅੋਗ ਫਾਰਮੈਟ ਵਿੱਚ ਐਕਸਪੋਰਟ ਕਰਨ ਲਈ ਲੈਣਾ ਪਏਗਾ।

## ਟੂਲ

ਇਸ ਕਾਰਜ ਲਈ, ਤੁਹਾਨੂੰ ਦੋ ਟੂਲਾਂ ਦੀ ਲੋੜ ਹੈ: Flask ਅਤੇ Pickle, ਦੋਹਾਂ Python ਤੇ ਚਲਦੇ ਹਨ।

✅ [Flask](https://palletsprojects.com/p/flask/) ਕੀ ਹੈ? ਆਪਣੇ ਬਣਾਉਣ ਵਾਲਿਆਂ ਦੁਆਰਾ 'ਮਾਈਕ੍ਰੋ-ਫਰੇਮਵਰਕ' ਵਜੋਂ ਪਰਿਭਾਸ਼ਿਤ ਕੀਤਾ ਗਿਆ, Flask Python ਅਤੇ ਟੈਮਪਲੇਟਿੰਗ ਇੰਜਣ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਵੈੱਬ ਫਰੇਮਵਰਕ ਦੇ ਬੁਨਿਆਦੀ ਫੀਚਰ ਮੁਹੱਈਆ ਕਰਵਾਉਂਦਾ ਹੈ। Flask ਨਾਲ ਬਣਾਉਣ ਦਾ ਅਭਿਆਸ ਕਰਨ ਲਈ [ਇਸ Learn ਮੋਡੀਊਲ](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) ਨੂੰ ਵੇਖੋ।

✅ [Pickle](https://docs.python.org/3/library/pickle.html) ਕੀ ਹੈ? Pickle 🥒 ਇੱਕ Python ਮਾਡਿਊਲ ਹੈ ਜੋ Python ਓਬਜੈਕਟ ਸਟ੍ਰਕਚਰ ਨੂੰ ਸੀਰੀਅਲਾਈਜ਼ ਅਤੇ ਡੀ-ਸੀਰੀਅਲਾਈਜ਼ ਕਰਦਾ ਹੈ। ਜਦੋਂ ਤੁਸੀਂ ਮਾਡਲ ਨੂੰ 'ਪਿਕਲ' ਕਰਦੇ ਹੋ, ਤਾਂ ਤੁਸੀਂ ਇਸ ਦੀ ਬਣਤਰ ਨੂੰ ਸੀਰੀਅਲਾਈਜ਼ ਜਾਂ ਸਮਤਲ ਕਰਦੇ ਹੋ ਤਾਂ ਜੋ ਵੈੱਬ ਵਿੱਚ ਵਰਤਿਆ ਜਾ ਸਕੇ। ਧਿਆਨ ਰੱਖੋ: pickle ਸੁਰੱਖਿਅਤ ਨਹੀਂ ਹੈ, ਇਸ ਲਈ 'ਅਨਪਿਕਲ' ਫਾਇਲ ਕਰਦੇ ਸਮੇਂ ਸਾਵਧਾਨ ਰਹੋ। ਪਿਕਲ ਕੀਤੀ ਫਾਈਲ ਦਾ ਸਫਿਕਸ `.pkl` ਹੁੰਦਾ ਹੈ।

## ਅਭਿਆਸ - ਆਪਣਾ ਡੇਟਾ ਸਾਫ਼ ਕਰੋ

ਇਸ ਪਾਠ ਵਿੱਚ ਤੁਸੀਂ 80,000 UFO ਦ੍ਰਿਸ਼ਟੀਗੋਚਰਾਂ ਤੋਂ ਡੇਟਾ ਵਰਤੋਂਗੇ, ਜੋ [NUFORC](https://nuforc.org) (ਰਾਸ਼ਟਰੀ UFO ਰਿਪੋਰਟਿੰਗ ਸੈਂਟਰ) ਨੇ ਇਕੱਠਾ ਕੀਤਾ ਹੈ। ਇਸ ਡੇਟਾ ਵਿੱਚ UFO ਦੇ ਦ੍ਰਿਸ਼ਟੀਗੋਚਰਾਂ ਦੇ ਕੁਝ ਦਿਲਚਸਪ ਵਰਣਨਾਂ ਹਨ, ਉਦਾਹਰਨ ਵਜੋਂ:

- **ਲੰਬੀ ਉਦਾਹਰਨ ਦੀ ਵਰਣਨਾ।** "ਇੱਕ ਆਦਮੀ ਇੱਕ ਰੋਸ਼ਨੀ ਦੀ ਕਿਰਣ ਵਿੱਚੋਂ ਨਿਕਲਦਾ ਹੈ ਜੋ ਰਾਤ ਨੂੰ ਘਾਂਸ ਵਾਲੇ ਖੇਤ 'ਤੇ ਚਮਕਦੀ ਹੈ ਅਤੇ ਉਹ ਟੈਕਸਾਸ ਇੰਸਟਰੂਮੈਂਟਸ ਦੇ ਪਾਰਕਿੰਗ ਲਾਟ ਵੱਲ ਦੌੜਦਾ ਹੈ"।
- **ਛੋਟੀ ਉਦਾਹਰਨ ਦੀ ਵਰਣਨਾ।** "ਲਾਈਟਸ ਸਾਡੇ ਪਿੱਛੇ ਦੌੜੀ।"

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) ਸਪਰੇਡਸ਼ੀਟ ਵਿੱਚ ਕਾਲਮ ਹਨ ਜਿਨ੍ਹਾਂ ਵਿੱਚ `city`, `state` ਅਤੇ `country` ਹਨ ਜਿੱਥੇ ਦ੍ਰਿਸ਼ਟੀਗੋਚਰ ਹੋਇਆ, ਵਸਤੂ ਦਾ `shape` ਅਤੇ ਇਸ ਦਾ `latitude` ਅਤੇ `longitude` ਵੀ ਸ਼ਾਮਲ ਹਨ।

ਇਸ ਪਾਠ ਵਿੱਚ ਸ਼ਾਮਲ ਸਾਫ਼ ਸੂਤਰ [notebook](notebook.ipynb) ਵਿੱਚ:

1. ਪਹਿਲਾਂ ਹੀ ਕੀਤੇ ਗਏ ਪਾਠਾਂ ਵਾਂਗ `pandas`, `matplotlib`, ਅਤੇ `numpy` ਇੰਪੋਰਟ ਕਰੋ ਅਤੇ ufos ਸਪਰੇਡਸ਼ੀਟ ਨੂੰ ਇੰਪੋਰਟ ਕਰੋ। ਤੁਸੀਂ ਨਮੂਨਾ ਡੇਟਾਸੈਟ ਦੇਖ ਸਕਦੇ ਹੋ:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos ਡੇਟਾ ਨੂੰ ਇੱਕ ਛੋਟੇ ਡੇਟਾਫ਼ਰੇਮ ਵਿੱਚ ਬਦਲੋ ਜਿਨ੍ਹਾਂ ਟਾਈਟਲ ਤਾਜ਼ਾ ਹਨ। `Country` ਫੀਲਡ ਵਿੱਚ ਯੂਨੀਕ ਮੁੱਲ ਚੈੱਕ ਕਰੋ।

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. ਹੁਣ, ਤੁਸੀਂ ਡੇਟਾ ਦੀ ਮਾਤਰਾ ਘਟਾ ਸਕਦੇ ਹੋ ਬੱਸਗੁੰਨੇ ਮੁੱਲਾਂ ਨੂੰ ਹਟਾ ਕੇ ਅਤੇ ਕੇਵਲ ਉਹ ਦ੍ਰਿਸ਼ਟੀਗੋਚਰ ਲੈ ਕੇ ਜੋ 1-60 ਸਕਿੰਟ ਵਿੱਚ ਆਉਂਦੇ ਹਨ:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. ਸਕਾਇਕੀਟ-ਲਰਨ ਦੇ `LabelEncoder` ਲਾਇਬਰੇਰੀ ਨੂੰ ਇੰਪੋਰਟ ਕਰੋ ताकि ਦੇਸ਼ਾਂ ਦੇ ਪਾਠ ਮੁੱਲਾਂ ਨੂੰ ਨੰਬਰਾਂ ਵਿੱਚ ਬਦਲਿਆ ਜਾ ਸਕੇ:

    ✅ LabelEncoder ਅੱਖਰਾਂ ਦੇ ਕ੍ਰਮ ਵਿੱਚ ਡੇਟਾ ਨੂੰ ਕੋਡ ਕਰਦਾ ਹੈ

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    ਤੁਹਾਡਾ ਡੇਟਾ ਇਸ ਤਰ੍ਹਾਂ ਦਿੱਸਣਾ ਚਾਹੀਦਾ ਹੈ:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## ਅਭਿਆਸ - ਆਪਣੇ ਮਾਡਲ ਨੂੰ ਬਣਾਓ

ਹੁਣ ਤੁਸੀਂ ਮਾਡਲ ਨੂੰ ਚੰਗੀ ਤਰ੍ਹਾਂ ਟ੍ਰੇਨ ਕਰਨ ਲਈ ਡੇਟਾ ਨੂੰ ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟਿੰਗ ਗਰੁੱਪਾਂ ਵਿੱਚ ਵੰਡਣ ਲਈ ਤਿਆਰ ਹੋ।

1. ਉਹ ਤਿੰਨ ਫੀਚਰ ਚੁਣੋ ਜਿਨ੍ਹਾਂ ਤੇ ਤੁਸੀਂ ਟ੍ਰੇਨ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ ਜਿਵੇਂ ਕਿ ਤੁਹਾਡੀ X ਵੇਕਟਰ ਹੋਏਗੀ, ਅਤੇ y ਵੇਕਟਰ `Country` ਹੋਵੇਗਾ। ਤੁਸੀਂ ਇੰਜਪੁਟ ਵਿੱਚ `Seconds`, `Latitude` ਅਤੇ `Longitude` ਦੇ ਕੇ ਇੱਕ ਦੇਸ਼ ਦੀ ਪਛਾਣ ਪ੍ਰਾਪਤ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ।

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. ਲੋਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਆਪਣੇ ਮਾਡਲ ਨੂੰ ਟ੍ਰੇਨ ਕਰੋ:

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

ਇਹ ਸਹੀ ਹੈ **(ਲਗਭਗ 95%)**, ਬੇ-ਜੁੜੇ ਤੌਰ 'ਤੇ, ਕਿਉਂਕਿ `Country` ਅਤੇ `Latitude/Longitude` ਵਿੱਚ ਸੰਬੰਧ ਹੈ।

ਜੋ ਮਾਡਲ ਤੁਸੀਂ ਬਣਾਇਆ ਹੈ ਉਹ ਜ਼ਿਆਦਾ ਇਨਕਲਾਬੀ ਨਹੀਂ ਹੈ ਕਿਉਂਕਿ ਤੁਸੀਂ `Latitude` ਅਤੇ `Longitude` ਦੇ ਆਧਾਰ ਤੇ `Country` ਦਾ ਅੰਦਾਜ਼ਾ ਲਗਾ ਸਕਦੇ ਹੋ, ਪਰ ਇਹ ਕੱਚੇ ਡੇਟਾ ਤੋਂ ਸਾਫ, ਐਕਸਪੋਰਟ ਅਤੇ ਫਿਰ ਵੈੱਬ ਐਪ ਵਿੱਚ ਮਾਡਲ ਵਰਤਣ ਦਾ ਵਧੀਆ ਅਭਿਆਸ ਹੈ।

## ਅਭਿਆਸ - ਆਪਣੇ ਮਾਡਲ ਨੂੰ 'ਪਿਕਲ' ਕਰੋ

ਹੁਣ ਸਮਾਂ ਆ ਗਿਆ ਹੈ ਕਿ ਤੁਸੀਂ ਆਪਣੇ ਮਾਡਲ ਨੂੰ _ਪਿਕਲ_ ਕਰੋ! ਤੁਸੀਂ ਇਹ ਕੁਝ ਕੋਡ ਲਾਈਨਾਂ ਵਿੱਚ ਕਰ ਸਕਦੇ ਹੋ। ਇਕ ਵਾਰ ਜਦੋਂ ਇਹ ਪਿਕਲ ਹੋ ਜਾਵੇ, ਆਪਣੇ ਪਿਕਲ ਕੀਤੇ ਮਾਡਲ ਨੂੰ ਲੋਡ ਕਰੋ ਅਤੇ ਉਸਨੂੰ ਸਕਿੰਟ, ਅੱਖਾਂ ਦੀ ਚੌੜਾਈ ਅਤੇ ਲਾਂਬਾਈ ਵਾਲੇ ਨਮੂਨਾ ਡੇਟਾ ਐਰੇ ਨਾਲ ਟੈਸਟ ਕਰੋ,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

ਮਾਡਲ **'3'** ਵਾਪਿਸ ਕਰਦਾ ਹੈ, ਜੋ UK ਦਾ ਦੇਸ਼ ਕੋਡ ਹੈ। ਵਾਹ! 👽

## ਅਭਿਆਸ - ਇੱਕ Flask ਐਪ ਬਣਾਓ

ਹੁਣ ਤੁਸੀਂ ਆਪਣੇ ਮਾਡਲ ਨੂੰ ਕਾਲ ਕਰਨ ਅਤੇ ਇਸੇ ਤਰ੍ਹਾਂ ਦੇ ਨਤੀਜੇ ਵਾਪਸ ਕਰਨ ਲਈ Flask ਐਪ ਬਣਾ ਸਕਦੇ ਹੋ, ਪਰ ਇੱਕ ਜ਼ਿਆਦਾ ਦਿਖਾਈ ਦੇਣ ਵਾਲੇ ਢੰਗ ਨਾਲ।

1. _notebook.ipynb_ ਫਾਇਲ ਦੇ ਨਾਲ ਇੱਕ **web-app** ਨਾਂ ਦਾ ਫੋਲਡਰ ਬਣਾਓ ਜਿੱਥੇ ਤੁਹਾਡੀ _ufo-model.pkl_ ਫਾਇਲ ਹੈ।

1. ਉਸ ਫੋਲਡਰ ਵਿੱਚ ਤਿੰਨ ਹੋਰ ਫੋਲਡਰ ਬਣਾਓ: **static** ਜਿਸ ਦੇ ਅੰਦਰ **css** ਫੋਲਡਰ ਹੋਵੇ, ਅਤੇ **templates**। ਹੁਣ ਤੁਹਾਡੇ ਕੋਲ ਇਹ ਫਾਇਲਾਂ ਅਤੇ ਡਾਇਰੈਕਟਰੀਜ਼ ਹੋਣ:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ ਬਣੇ ਹੋਏ ਐਪ ਦੀ ਨਜ਼ਰ ਲਈ ਹੱਲ ਫੋਲਡਰ ਨੂੰ ਵੇਖੋ

1. _web-app_ ਫੋਲਡਰ ਵਿੱਚ ਬਣਾਉਣ ਵਾਲੀ ਪਹਿਲੀ ਫਾਈਲ **requirements.txt** ਹੈ। ਜਿਵੇਂ ਕਿ ਜਾਵਾਸਕ੍ਰਿਪਟ ਐਪ ਵਿੱਚ _package.json_ ਹੁੰਦਾ ਹੈ, ਇਹ ਫਾਇਲ ਐਪ ਲਈ ਲੋੜੀਂਦੇ ਡਿਪੈਂਡੇਨਸੀਜ਼ ਦੀ ਸੂਚੀ ਦਿੰਦੀ ਹੈ। **requirements.txt** ਵਿੱਚ ਇਹ ਲਾਈਨਾਂ ਸ਼ਾਮਲ ਕਰੋ:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. ਹੁਣ, ਇਸ ਫਾਇਲ ਨੂੰ _web-app_ ਵਿੱਚ ਜਾਂਕੇ ਚਲਾਓ:

    ```bash
    cd web-app
    ```

1. ਆਪਣੇ ਟਰਮੀਨਲ ਵਿੱਚ `pip install` ਲਿਖੋ ਤਾਂ ਜੋ _requirements.txt_ ਵਿੱਚ ਦਿੱਤੀਆਂ ਲਾਇਬਰੇਰੀਆਂ ਇੰਸਟਾਲ ਹੋ ਜਾਣ:

    ```bash
    pip install -r requirements.txt
    ```

1. ਹੁਣ, ਆਪਣੇ ਐਪ ਨੂੰ ਖਤਮ ਕਰਨ ਲਈ ਤਿੰਨ ਹੋਰ ਫਾਇਲਾਂ ਬਣਾਉਣ ਲਈ ਤਿਆਰ ਹੋ:

    1. ਰੂਟ ਵਿੱਚ **app.py** ਬਣਾਓ।
    2. _templates_ ਡਾਇਰੈਕਟਰੀ ਵਿੱਚ **index.html** ਬਣਾਓ।
    3. _static/css_ ਡਾਇਰੈਕਟਰੀ ਵਿੱਚ **styles.css** ਬਣਾਓ।

1. _styles.css_ ਫਾਇਲ ਵਿੱਚ ਕੁਝ ਸਟਾਈਲਜ਼ ਬਣਾਓ:

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

1. ਅਗਲੇ, _index.html_ ਫਾਇਲ ਬਣਾਓ:

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

    ਇਸ ਫਾਇਲ ਵਿੱਚ ਟੈਮਪਲੇਟਿੰਗ ਨੂੰ ਦੇਖੋ। ਧਿਆਨ ਦਿਓ ਕਿ ਵੇਰੀਏਬਲਾਂ ਦੇ ਆਲੇ-ਦੁਆਲੇ 'ਮਸਟੈਚ' ਸਿੰਟੈਕਸ ਹੈ ਜੋ ਐਪ ਤੋਂ ਮੈਲ-ਮਿਲਾਪ ਕਰਦੇ ਹਨ, ਜਿਵੇਂ ਅਨੁਮਾਨ ਟੈਕਸਟ: `{{}}`। ਇੱਕ ਫਾਰਮ ਵੀ ਹੈ ਜੋ `/predict` ਰੂਟ ਨੂੰ ਅਨੁਮਾਨ ਪੁੱਟਦਾ ਹੈ।

    ਅੰਤ ਵਿੱਚ, ਤੁਸੀਂ ਉਸ ਪਾਈਥਨ ਫਾਇਲ ਨੂੰ ਬਣਾਉਣ ਲਈ ਤਿਆਰ ਹੋ ਜੋ ਮਾਡਲ ਦੀ ਖਪਤ ਅਤੇ ਅਨੁਮਾਨਾਂ ਦੀ ਪ੍ਰਦਰਸ਼ਨੀ ਚਲਾਉਂਦੀ ਹੈ:

1. `app.py` ਵਿੱਚ ਜੋੜੋ:

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

    > 💡 ਸੁਝਾਅ: ਜਦੋਂ ਤੁਸੀਂ Flask ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਵੈੱਬ ਐਪ ਚਲਾ ਰਹੇ ਹੋ ਅਤੇ [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) ਨੂੰ ਜੋੜਦੇ ਹੋ, ਤਾਂ ਤੁਹਾਡੇ ਐਪ ਵਿੱਚ ਕੀਤੇ ਗਏ ਕੋਈ ਵੀ ਬਦਲਾਅ ਤੁਰੰਤ ਪ੍ਰਗਟ ਹੋ ਜਾਣਗੇ ਬਿਨਾਂ ਸਰਵਰ ਨੂੰ ਫਿਰ ਨਾਲ ਚਾਲੂ ਕੀਤੇ। ਧਿਆਨ ਰੱਖੋ! ਇਸ ਮੋਡ ਨੂੰ ਪ੍ਰੋਡੱਖਸ਼ਨ ਐਪ ਵਿੱਚ ਸਮਰਥਿਤ ਨਾ ਕਰੋ।

ਜੇ ਤੁਸੀਂ `python app.py` ਜਾਂ `python3 app.py` ਚਲਾਓਗੇ - ਤੁਹਾਡਾ ਵੈੱਬ ਸਰਵਰ ਲੋਕਲ ਤੌਰ 'ਤੇ ਚਾਲੂ ਹੋਵੇਗਾ, ਅਤੇ ਤੁਸੀਂ ਇੱਕ ਛੋਟਾ ਫਾਰਮ ਭਰਕੇ UFO ਦਾ ਨਤੀਜਾ ਤੁਰੰਤ ਪ੍ਰਾਪਤ ਕਰ ਸਕਦੇ ਹੋ ਕਿ ਕਿੱਥੇ UFO ਦਿਖਾਈ ਦਿੱਤਾ!

ਇਸ ਤੋਂ ਪਹਿਲਾਂ, `app.py` ਦੇ ਹਿੱਸਿਆਂ ਨੂੰ ਦੇਖੋ:

1. ਸਭ ਤੋਂ ਪਹਿਲਾਂ ਨਿਰਭਰਤਾ ਲੋਡ ਕੀਤੀਆਂ ਜਾਂਦੀਆਂ ਹਨ ਅਤੇ ਐਪ ਸ਼ੁਰੂ ਹੁੰਦੀ ਹੈ।
1. ਫਿਰ ਮਾਡਲ ਨੂੰ ਇੰਪੋਰਟ ਕੀਤਾ ਜਾਂਦਾ ਹੈ।
1. ਫਿਰ, ਹੌਮ ਰੂਟ ਤੇ index.html ਰੇਡਰ ਹੁੰਦਾ ਹੈ।

`/predict` ਰੂਟ 'ਤੇ ਜਦੋਂ ਫਾਰਮ ਪੋਸਟ ਹੁੰਦਾ ਹੈ ਤਾਂ ਕਈ ਕਾਮ ਹੁੰਦੇ ਹਨ:

1. ਫਾਰਮ ਦੇ ਵੇਰੀਏਬਲ ਇਕੱਠੇ ਕਰਕੇ numpy ਐਰੇ ਵਿੱਚ ਬਦਲ ਦਿੱਤੇ ਜਾਂਦੇ ਹਨ। ਫਿਰ ਇਹ ਮਾਡਲ ਨੂੰ ਭੇਜੇ ਜਾਂਦੇ ਹਨ ਅਤੇ ਅਨੁਮਾਨ ਵਾਪਸ ਮਿਲਦਾ ਹੈ।
2. ਸਾਡੀ ਇੱਛਾ ਹੈ ਕਿ ਦਿਖਾਇਆ ਜਾਣ ਵਾਲੇ ਦੇਸ਼ਾਂ ਨੂੰ ਅਨੁਮਾਨੀ ਦੇਸ਼ ਕੋਡ ਤੋਂ ਪੜ੍ਹਨਯੋਗ ਪਾਠ ਰੂਪ ਵਿੱਚ ਫਿਰ ਤੋਂ ਰੇਡਰ ਕਰਕੇ ਵਾਪਸ index.html ਨੂੰ ਭੇਜਿਆ ਜਾਵੇ।

Flask ਅਤੇ ਪਿਕਲ ਕੀਤੇ ਮਾਡਲ ਦੀ ਵਰਤੋਂ ਨਾਲ ਇਸ ਤਰ੍ਹਾਂ ਮਾਡਲ ਵਰਤਣਾ ਸਾਫ਼-ਸੁਥਰਾ ਹੈ। ਸਭ ਤੋਂ ਮੁਸ਼ਕਲ ਗੱਲ ਇਹ ਸਮਝਣਾ ਹੈ ਕਿ ਡੇਟਾ ਕਿਸ ਰੂਪ ਵਿੱਚ ਮਾਡਲ ਨੂੰ ਭੇਜਣਾ ਹੈ ਤਾਂ ਜੋ ਅਨੁਮਾਨ ਪ੍ਰਾਪਤ ਹੋਵੇ। ਇਹ ਸਾਰਾ ਮਾਡਲ ਦੇ ਟ੍ਰੇਨਿੰਗ ਦੇ ਆਧਾਰ ਤੇ ਨਿਰਭਰ ਕਰਦਾ ਹੈ। ਇਸ ਵਾਰੀ ਮਾਡਲ ਨੂੰ ਅਨੁਮਾਨ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ ਤਿੰਨ ਡੇਟਾ ਪੁਆਇੰਟ ਚਾਹੀਦੇ ਹਨ।

ਪ੍ਰਫੈਸ਼ਨਲ ਸੈਟਿੰਗ ਵਿੱਚ, ਤੁਸੀਂ ਵੇਖ ਸਕਦੇ ਹੋ ਕਿ ਮਾਡਲ ਟ੍ਰੇਨ ਕਰਨ ਵਾਲਿਆਂ ਅਤੇ ਜੋ ਲੋਕ ਇਸਨੂੰ ਵੈੱਬ ਜਾਂ ਮੋਬਾਈਲ ਐਪ ਵਿੱਚ ਵਰਤਦੇ ਹਨ, ਉਨ੍ਹਾਂ ਦਾ ਚੰਗਾ ਸੰਚਾਰ ਕਿੰਨਾ ਜ਼ਰੂਰੀ ਹੁੰਦਾ ਹੈ। ਸਾਡੇ ਮਾਮਲੇ ਵਿੱਚ, ਇਹ ਸਿਰਫ ਇੱਕ ਬੰਦਾ ਹੈ, ਤੁਹਾਨੂੰ!

---

## 🚀 ਚੈਲੇਂਜ

ਨੋਟਬੁੱਕ ਵਿੱਚ ਕੰਮ ਕਰਨ ਅਤੇ ਮਾਡਲ ਨੂੰ Flask ਐਪ ਵਿੱਚ ਲਾਉਣ ਦੀ ਬਜਾਏ, ਤੁਸੀਂ ਮਾਡਲ ਨੂੰ ਸੀਧਾ Flask ਐਪ ਵਿੱਚ ਟ੍ਰੇਨ ਕਰ ਸਕਦੇ ਹੋ! ਆਪਣੇ ਨੋਟਬੁੱਕ ਦੇ Python ਕੋਡ ਨੂੰ ਸਾਫ਼-ਸੁਥਰੇ ਡੇਟਾ ਤੋਂ ਬਾਅਦ ਸ਼ਾਇਦ `train` ਨਾਮਕ ਰੂਟ 'ਤੇ ਟ੍ਰੇਨ ਕਰਨ ਵਾਸਤੇ ਐਪ ਵਿੱਚ ਬਦਲੋ। ਇਸ ਤਰੀਕੇ ਦੇ ਲਾਭ ਅਤੇ ਨੁਕਸਾਨ ਕੀ ਹਨ?

## [ਪੋਸਟ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਸਮੀਖਿਆ ਅਤੇ ਸਵੈ ਅਧਿਐਨ

ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲਾਂ ਦੀ ਖਪਤ ਲਈ ਵੈੱਬ ਐਪ ਬਣਾਉਣ ਦੇ ਕਈ ਤਰੀਕੇ ਹਨ। ਜਾਵਾਸਕ੍ਰਿਪਟ ਜਾਂ Python ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਵੈੱਬ ਐਪ ਬਣਾਉਣ ਦੇ ਤਰੀਕਿਆਂ ਦੀ ਸੂਚੀ ਬਣਾਓ। ਆਰਕੀਟੈਕਚਰ ਬਾਰੇ ਸੋਚੋ: ਕੀ ਮਾਡਲ ਨੂੰ ਐਪ ਵਿੱਚ ਹੀ ਰਹਿਣਾ ਚਾਹੀਦਾ ਹੈ ਜਾਂ ਕਲਾਊਡ ਵਿੱਚ? ਜੇ ਕਲਾਊਡ ਵਿੱਚ, ਤਾਂ ਤੁਸੀਂ ਕਿਵੇਂ ਪਹੁੰਚੋਗੇ? ਇੱਕ ਲਾਗੂ ਕੀਤੇ ਗਏ ML ਵੈੱਬ ਹੱਲ ਲਈ ਆਰਕੀਟੈਕਚਰਲ ਮਾਡਲ ਬਣਾਓ।

## ਕੰਮ

[ਕਿਸੇ ਹੋਰ ਮਾਡਲ ਦੀ ਕੋਸ਼ਿਸ ਕਰੋ](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ਅਸਵੀਕਾਰੋਪਣ**:
ਇਸ ਦਸਤਾਵੇਜ਼ ਦਾ ਅਨੁਵਾਦ ਏਆਈ ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀਤਾਵਾਂ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਰੱਖੋ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸਮੱਤਿਆਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਆਪਣੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਕ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਜਰੂਰੀ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫ਼ਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਅਸੀਂ ਇਸ ਅਨੁਵਾਦ ਦੇ ਉਪਯੋਗ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੀਆਂ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀਆਂ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆਵਾਂ ਲਈ ਜਵਾਬਦੇਹ ਨਹੀਂ ਹਾਂ।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->