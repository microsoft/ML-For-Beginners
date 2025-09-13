<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "cf8ecc83f28e5b98051d2179eca08e08",
  "translation_date": "2025-08-29T17:30:20+00:00",
  "source_file": "1-Introduction/README.md",
  "language_code": "pa"
}
-->
# ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦਾ ਪਰਿਚਯ

ਇਸ ਪਾਠਕ੍ਰਮ ਦੇ ਇਸ ਭਾਗ ਵਿੱਚ, ਤੁਹਾਨੂੰ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦੇ ਮੂਲ ਧਾਰਨਾਵਾਂ, ਇਹ ਕੀ ਹੈ, ਇਸ ਦਾ ਇਤਿਹਾਸ ਅਤੇ ਉਹ ਤਕਨੀਕਾਂ ਜਿਨ੍ਹਾਂ ਨੂੰ ਖੋਜਕਰਤਾ ਇਸ ਨਾਲ ਕੰਮ ਕਰਨ ਲਈ ਵਰਤਦੇ ਹਨ, ਨਾਲ ਜਾਣੂ ਕਰਵਾਇਆ ਜਾਵੇਗਾ। ਆਓ, ਇਸ ਨਵੇਂ ML ਦੀ ਦੁਨੀਆ ਨੂੰ ਇਕੱਠੇ ਖੋਜੀਏ!

![globe](../../../translated_images/globe.59f26379ceb40428672b4d9a568044618a2bf6292ecd53a5c481b90e3fa805eb.pa.jpg)
> ਫੋਟੋ <a href="https://unsplash.com/@bill_oxford?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਬਿਲ ਆਕਸਫੋਰਡ</a> ਦੁਆਰਾ <a href="https://unsplash.com/s/photos/globe?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਅਨਸਪਲੈਸ਼</a> 'ਤੇ
  
### ਪਾਠ

1. [ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦਾ ਪਰਿਚਯ](1-intro-to-ML/README.md)
1. [ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਅਤੇ AI ਦਾ ਇਤਿਹਾਸ](2-history-of-ML/README.md)
1. [ਨਿਆਂ ਅਤੇ ਮਸ਼ੀਨ ਲਰਨਿੰਗ](3-fairness/README.md)
1. [ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦੀਆਂ ਤਕਨੀਕਾਂ](4-techniques-of-ML/README.md)

### ਸ਼੍ਰੇਯ

"ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦਾ ਪਰਿਚਯ" ਨੂੰ [ਮੁਹੰਮਦ ਸਾਕਿਬ ਖਾਨ ਇਨਾਨ](https://twitter.com/Sakibinan), [ਓਰਨੇਲਾ ਅਲਤੁਨਯਾਨ](https://twitter.com/ornelladotcom) ਅਤੇ [ਜੈਨ ਲੂਪਰ](https://twitter.com/jenlooper) ਸਮੇਤ ਇੱਕ ਟੀਮ ਦੁਆਰਾ ♥️ ਨਾਲ ਲਿਖਿਆ ਗਿਆ।

"ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦਾ ਇਤਿਹਾਸ" ਨੂੰ ♥️ ਨਾਲ [ਜੈਨ ਲੂਪਰ](https://twitter.com/jenlooper) ਅਤੇ [ਐਮੀ ਬੋਇਡ](https://twitter.com/AmyKateNicho) ਦੁਆਰਾ ਲਿਖਿਆ ਗਿਆ।

"ਨਿਆਂ ਅਤੇ ਮਸ਼ੀਨ ਲਰਨਿੰਗ" ਨੂੰ ♥️ ਨਾਲ [ਟੋਮੋਮੀ ਇਮੁਰਾ](https://twitter.com/girliemac) ਦੁਆਰਾ ਲਿਖਿਆ ਗਿਆ।

"ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦੀਆਂ ਤਕਨੀਕਾਂ" ਨੂੰ ♥️ ਨਾਲ [ਜੈਨ ਲੂਪਰ](https://twitter.com/jenlooper) ਅਤੇ [ਕ੍ਰਿਸ ਨੋਰਿੰਗ](https://twitter.com/softchris) ਦੁਆਰਾ ਲਿਖਿਆ ਗਿਆ।

---

**ਅਸਵੀਕਰਤੀ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀ ਹੋਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਦੇ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚਤਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਇਸ ਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।