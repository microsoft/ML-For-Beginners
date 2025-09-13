<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-06T07:17:55+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "pa"
}
-->
# ਹੋਟਲ ਰਿਵਿਊਜ਼ ਨਾਲ ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ - ਡਾਟਾ ਪ੍ਰੋਸੈਸਿੰਗ

ਇਸ ਭਾਗ ਵਿੱਚ ਤੁਸੀਂ ਪਿਛਲੇ ਪਾਠਾਂ ਵਿੱਚ ਸਿੱਖੀਆਂ ਤਕਨੀਕਾਂ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਇੱਕ ਵੱਡੇ ਡਾਟਾਸੈਟ ਦਾ ਖੋਜਾਤਮਕ ਡਾਟਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋਗੇ। ਜਦੋਂ ਤੁਹਾਨੂੰ ਵੱਖ-ਵੱਖ ਕਾਲਮਾਂ ਦੀ ਉਪਯੋਗਤਾ ਦੀ ਚੰਗੀ ਸਮਝ ਹੋ ਜਾਵੇਗੀ, ਤਾਂ ਤੁਸੀਂ ਸਿੱਖੋਗੇ:

- ਕਿਵੇਂ ਗੈਰ-ਜ਼ਰੂਰੀ ਕਾਲਮਾਂ ਨੂੰ ਹਟਾਉਣਾ ਹੈ
- ਮੌਜੂਦਾ ਕਾਲਮਾਂ ਦੇ ਆਧਾਰ 'ਤੇ ਕੁਝ ਨਵਾਂ ਡਾਟਾ ਕਿਵੇਂ ਗਿਣਨਾ ਹੈ
- ਅੰਤਿਮ ਚੁਣੌਤੀ ਲਈ ਡਾਟਾਸੈਟ ਨੂੰ ਸੇਵ ਕਰਨਾ ਕਿਵੇਂ ਹੈ

## [ਪ੍ਰੀ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

### ਪਰਿਚਯ

ਅਜੇ ਤੱਕ ਤੁਸੀਂ ਸਿੱਖਿਆ ਹੈ ਕਿ ਟੈਕਸਟ ਡਾਟਾ ਅੰਕੜੇਵਾਰ ਡਾਟਾ ਦੀ ਤਰ੍ਹਾਂ ਨਹੀਂ ਹੁੰਦਾ। ਜੇ ਇਹ ਟੈਕਸਟ ਕਿਸੇ ਮਨੁੱਖ ਦੁਆਰਾ ਲਿਖਿਆ ਜਾਂ ਬੋਲਿਆ ਗਿਆ ਹੈ, ਤਾਂ ਇਸ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਕੇ ਪੈਟਰਨ ਅਤੇ ਆਵ੍ਰਿੱਤੀਆਂ, ਭਾਵਨਾ ਅਤੇ ਅਰਥ ਪਤਾ ਲਗਾਇਆ ਜਾ ਸਕਦਾ ਹੈ। ਇਹ ਪਾਠ ਤੁਹਾਨੂੰ ਇੱਕ ਅਸਲ ਡਾਟਾਸੈਟ ਅਤੇ ਅਸਲ ਚੁਣੌਤੀ ਵਿੱਚ ਲੈ ਜਾਂਦਾ ਹੈ: **[ਯੂਰਪ ਵਿੱਚ 515K ਹੋਟਲ ਰਿਵਿਊਜ਼ ਡਾਟਾ](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, ਜਿਸ ਵਿੱਚ ਇੱਕ [CC0: ਪਬਲਿਕ ਡੋਮੇਨ ਲਾਇਸੈਂਸ](https://creativecommons.org/publicdomain/zero/1.0/) ਸ਼ਾਮਲ ਹੈ। ਇਹ ਡਾਟਾ Booking.com ਤੋਂ ਪਬਲਿਕ ਸਰੋਤਾਂ ਤੋਂ ਸਕ੍ਰੈਪ ਕੀਤਾ ਗਿਆ ਸੀ। ਡਾਟਾਸੈਟ ਦੇ ਰਚਨਹਾਰ Jiashen Liu ਹਨ।

### ਤਿਆਰੀ

ਤੁਹਾਨੂੰ ਲੋੜ ਹੋਵੇਗੀ:

* Python 3 ਦੀ ਵਰਤੋਂ ਕਰਕੇ .ipynb ਨੋਟਬੁੱਕ ਚਲਾਉਣ ਦੀ ਸਮਰੱਥਾ
* pandas
* NLTK, [ਜਿਸਨੂੰ ਤੁਹਾਨੂੰ ਸਥਾਨਕ ਤੌਰ 'ਤੇ ਇੰਸਟਾਲ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ](https://www.nltk.org/install.html)
* ਡਾਟਾਸੈਟ ਜੋ Kaggle 'ਤੇ ਉਪਲਬਧ ਹੈ [ਯੂਰਪ ਵਿੱਚ 515K ਹੋਟਲ ਰਿਵਿਊਜ਼ ਡਾਟਾ](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)। ਇਹ ਅਨਜ਼ਿਪ ਕਰਨ ਤੋਂ ਬਾਅਦ ਲਗਭਗ 230 MB ਹੈ। ਇਸਨੂੰ `/data` ਫੋਲਡਰ ਵਿੱਚ ਡਾਊਨਲੋਡ ਕਰੋ ਜੋ ਕਿ ਇਹਨਾਂ NLP ਪਾਠਾਂ ਨਾਲ ਜੁੜਿਆ ਹੋਇਆ ਹੈ।

## ਖੋਜਾਤਮਕ ਡਾਟਾ ਵਿਸ਼ਲੇਸ਼ਣ

ਇਹ ਚੁਣੌਤੀ ਇਹ ਮੰਨਦੀ ਹੈ ਕਿ ਤੁਸੀਂ ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ ਅਤੇ ਮਹਿਮਾਨ ਰਿਵਿਊ ਸਕੋਰਾਂ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਇੱਕ ਹੋਟਲ ਸਿਫਾਰਸ਼ੀ ਬੋਟ ਤਿਆਰ ਕਰ ਰਹੇ ਹੋ। ਡਾਟਾਸੈਟ ਜਿਸਦੀ ਤੁਸੀਂ ਵਰਤੋਂ ਕਰ ਰਹੇ ਹੋ, ਇਸ ਵਿੱਚ 6 ਸ਼ਹਿਰਾਂ ਵਿੱਚ 1493 ਵੱਖ-ਵੱਖ ਹੋਟਲਾਂ ਦੇ ਰਿਵਿਊ ਸ਼ਾਮਲ ਹਨ।

Python, ਹੋਟਲ ਰਿਵਿਊਜ਼ ਦੇ ਡਾਟਾਸੈਟ, ਅਤੇ NLTK ਦੇ ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਤੁਸੀਂ ਪਤਾ ਲਗਾ ਸਕਦੇ ਹੋ:

* ਰਿਵਿਊਜ਼ ਵਿੱਚ ਸਭ ਤੋਂ ਵੱਧ ਵਰਤੇ ਜਾਣ ਵਾਲੇ ਸ਼ਬਦ ਅਤੇ ਵਾਕਾਂਸ਼ ਕਿਹੜੇ ਹਨ?
* ਕੀ ਹੋਟਲ ਨੂੰ ਵਰਣਨ ਕਰਨ ਵਾਲੇ ਅਧਿਕਾਰਕ *ਟੈਗ* ਰਿਵਿਊ ਸਕੋਰਾਂ ਨਾਲ ਸਬੰਧਤ ਹਨ (ਜਿਵੇਂ ਕਿ ਕੀ ਕਿਸੇ ਖਾਸ ਹੋਟਲ ਲਈ *Family with young children* ਦੇ ਰਿਵਿਊ *Solo traveller* ਦੇ ਰਿਵਿਊਜ਼ ਨਾਲੋਂ ਜ਼ਿਆਦਾ ਨਕਾਰਾਤਮਕ ਹਨ, ਜੋ ਇਹ ਦਰਸਾਉਂਦਾ ਹੈ ਕਿ ਇਹ ਹੋਟਲ *Solo travellers* ਲਈ ਬਿਹਤਰ ਹੈ)?
* ਕੀ NLTK ਭਾਵਨਾ ਸਕੋਰ ਹੋਟਲ ਰਿਵਿਊਅਰ ਦੇ ਅੰਕੜੇਵਾਰ ਸਕੋਰ ਨਾਲ 'ਸਹਿਮਤ' ਹਨ?

#### ਡਾਟਾਸੈਟ

ਆਓ ਡਾਟਾਸੈਟ ਦੀ ਜਾਂਚ ਕਰੀਏ ਜੋ ਤੁਸੀਂ ਡਾਊਨਲੋਡ ਕੀਤਾ ਹੈ ਅਤੇ ਸਥਾਨਕ ਤੌਰ 'ਤੇ ਸੇਵ ਕੀਤਾ ਹੈ। ਫਾਈਲ ਨੂੰ ਕਿਸੇ ਐਡੀਟਰ ਜਿਵੇਂ ਕਿ VS Code ਜਾਂ Excel ਵਿੱਚ ਖੋਲ੍ਹੋ।

ਡਾਟਾਸੈਟ ਵਿੱਚ ਹੇਠਾਂ ਦਿੱਤੇ ਸਿਰਲੇਖ ਹਨ:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

ਇਹਨਾਂ ਨੂੰ ਹੇਠਾਂ ਦਿੱਤੇ ਤਰੀਕੇ ਨਾਲ ਸਮੂਹਬੱਧ ਕੀਤਾ ਗਿਆ ਹੈ ਜੋ ਸ਼ਾਇਦ ਜਾਂਚਣ ਲਈ ਆਸਾਨ ਹੋਵੇ:  
##### ਹੋਟਲ ਕਾਲਮ

* `Hotel_Name`, `Hotel_Address`, `lat` (ਅਕਸ਼ਾਂਸ), `lng` (ਦੇਸ਼ਾਂਤਰ)
  * *lat* ਅਤੇ *lng* ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਤੁਸੀਂ Python ਨਾਲ ਇੱਕ ਨਕਸ਼ਾ ਬਣਾਉ ਸਕਦੇ ਹੋ ਜੋ ਹੋਟਲ ਦੇ ਸਥਾਨ ਦਿਖਾਉਂਦਾ ਹੈ (ਸ਼ਾਇਦ ਨਕਾਰਾਤਮਕ ਅਤੇ ਸਕਾਰਾਤਮਕ ਰਿਵਿਊਜ਼ ਲਈ ਰੰਗ ਕੋਡਿੰਗ ਕਰਕੇ)
  * Hotel_Address ਸਾਨੂੰ ਸਿੱਧੇ ਤੌਰ 'ਤੇ ਜ਼ਿਆਦਾ ਲਾਭਕਾਰੀ ਨਹੀਂ ਲੱਗਦਾ, ਅਤੇ ਅਸੀਂ ਇਸਨੂੰ ਆਸਾਨ ਸਾਰਟਿੰਗ ਅਤੇ ਖੋਜ ਲਈ ਦੇਸ਼ ਨਾਲ ਬਦਲ ਸਕਦੇ ਹਾਂ

**ਹੋਟਲ ਮੈਟਾ-ਰਿਵਿਊ ਕਾਲਮ**

* `Average_Score`
  * ਡਾਟਾਸੈਟ ਦੇ ਰਚਨਹਾਰ ਦੇ ਅਨੁਸਾਰ, ਇਹ ਕਾਲਮ *ਹੋਟਲ ਦਾ ਔਸਤ ਸਕੋਰ ਹੈ, ਜੋ ਪਿਛਲੇ ਸਾਲ ਦੇ ਤਾਜ਼ਾ ਟਿੱਪਣੀ ਦੇ ਆਧਾਰ 'ਤੇ ਗਿਣਿਆ ਗਿਆ ਹੈ*। ਇਹ ਸਕੋਰ ਗਿਣਨ ਦਾ ਇੱਕ ਅਜੀਬ ਤਰੀਕਾ ਲੱਗਦਾ ਹੈ, ਪਰ ਇਹ ਡਾਟਾ ਸਕ੍ਰੈਪ ਕੀਤਾ ਗਿਆ ਹੈ, ਇਸ ਲਈ ਅਸੀਂ ਇਸਨੂੰ ਫਿਲਹਾਲ ਸੱਚ ਮੰਨ ਸਕਦੇ ਹਾਂ। 
  
  ✅ ਇਸ ਡਾਟੇ ਦੇ ਹੋਰ ਕਾਲਮਾਂ ਦੇ ਆਧਾਰ 'ਤੇ, ਕੀ ਤੁਸੀਂ ਔਸਤ ਸਕੋਰ ਗਿਣਨ ਦਾ ਹੋਰ ਤਰੀਕਾ ਸੋਚ ਸਕਦੇ ਹੋ?

* `Total_Number_of_Reviews`
  * ਇਹ ਹੋਟਲ ਨੂੰ ਮਿਲੇ ਕੁੱਲ ਰਿਵਿਊਜ਼ ਦੀ ਗਿਣਤੀ ਹੈ - ਇਹ ਸਪਸ਼ਟ ਨਹੀਂ ਹੈ (ਕੋਡ ਲਿਖਣ ਤੋਂ ਬਿਨਾਂ) ਕਿ ਕੀ ਇਹ ਡਾਟਾਸੈਟ ਵਿੱਚ ਸ਼ਾਮਲ ਰਿਵਿਊਜ਼ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ।
* `Additional_Number_of_Scoring`
  * ਇਸਦਾ ਮਤਲਬ ਹੈ ਕਿ ਰਿਵਿਊ ਸਕੋਰ ਦਿੱਤਾ ਗਿਆ ਸੀ ਪਰ ਰਿਵਿਊਅਰ ਦੁਆਰਾ ਕੋਈ ਸਕਾਰਾਤਮਕ ਜਾਂ ਨਕਾਰਾਤਮਕ ਟਿੱਪਣੀ ਨਹੀਂ ਲਿਖੀ ਗਈ

**ਰਿਵਿਊ ਕਾਲਮ**

- `Reviewer_Score`
  - ਇਹ ਇੱਕ ਅੰਕੜੇਵਾਰ ਮੁੱਲ ਹੈ ਜਿਸ ਵਿੱਚ ਘੱਟੋ-ਘੱਟ 1 ਦਸ਼ਮਲਵ ਸਥਾਨ ਹੁੰਦਾ ਹੈ ਅਤੇ ਇਸਦੇ ਘੱਟੋ-ਘੱਟ ਅਤੇ ਵੱਧ ਤੋਂ ਵੱਧ ਮੁੱਲ 2.5 ਅਤੇ 10 ਹਨ
  - ਇਹ ਨਹੀਂ ਸਮਝਾਇਆ ਗਿਆ ਕਿ 2.5 ਸਭ ਤੋਂ ਘੱਟ ਸਕੋਰ ਕਿਉਂ ਹੈ
- `Negative_Review`
  - ਜੇਕਰ ਕਿਸੇ ਰਿਵਿਊਅਰ ਨੇ ਕੁਝ ਨਹੀਂ ਲਿਖਿਆ, ਤਾਂ ਇਸ ਖੇਤਰ ਵਿੱਚ "**No Negative**" ਹੋਵੇਗਾ
  - ਧਿਆਨ ਦਿਓ ਕਿ ਰਿਵਿਊਅਰ ਨਕਾਰਾਤਮਕ ਰਿਵਿਊ ਕਾਲਮ ਵਿੱਚ ਸਕਾਰਾਤਮਕ ਟਿੱਪਣੀ ਲਿਖ ਸਕਦਾ ਹੈ (ਜਿਵੇਂ ਕਿ "ਇਸ ਹੋਟਲ ਬਾਰੇ ਕੁਝ ਵੀ ਬੁਰਾ ਨਹੀਂ ਹੈ")
- `Review_Total_Negative_Word_Counts`
  - ਵੱਧ ਨਕਾਰਾਤਮਕ ਸ਼ਬਦ ਗਿਣਤੀ ਘੱਟ ਸਕੋਰ ਦਰਸਾਉਂਦੀ ਹੈ (ਭਾਵਨਾ ਦੀ ਜਾਂਚ ਕੀਤੇ ਬਿਨਾਂ)
- `Positive_Review`
  - ਜੇਕਰ ਕਿਸੇ ਰਿਵਿਊਅਰ ਨੇ ਕੁਝ ਨਹੀਂ ਲਿਖਿਆ, ਤਾਂ ਇਸ ਖੇਤਰ ਵਿੱਚ "**No Positive**" ਹੋਵੇਗਾ
  - ਧਿਆਨ ਦਿਓ ਕਿ ਰਿਵਿਊਅਰ ਸਕਾਰਾਤਮਕ ਰਿਵਿਊ ਕਾਲਮ ਵਿੱਚ ਨਕਾਰਾਤਮਕ ਟਿੱਪਣੀ ਲਿਖ ਸਕਦਾ ਹੈ (ਜਿਵੇਂ ਕਿ "ਇਸ ਹੋਟਲ ਬਾਰੇ ਕੁਝ ਵੀ ਚੰਗਾ ਨਹੀਂ ਹੈ")
- `Review_Total_Positive_Word_Counts`
  - ਵੱਧ ਸਕਾਰਾਤਮਕ ਸ਼ਬਦ ਗਿਣਤੀ ਵੱਧ ਸਕੋਰ ਦਰਸਾਉਂਦੀ ਹੈ (ਭਾਵਨਾ ਦੀ ਜਾਂਚ ਕੀਤੇ ਬਿਨਾਂ)
- `Review_Date` ਅਤੇ `days_since_review`
  - ਇੱਕ ਤਾਜ਼ਗੀ ਜਾਂ ਪੁਰਾਣੇਪਨ ਦਾ ਮਾਪ ਲਾਗੂ ਕੀਤਾ ਜਾ ਸਕਦਾ ਹੈ (ਪੁਰਾਣੇ ਰਿਵਿਊ ਸ਼ਾਇਦ ਨਵੇਂ ਰਿਵਿਊਜ਼ ਜਿੰਨੇ ਸਹੀ ਨਹੀਂ ਹੋ ਸਕਦੇ ਕਿਉਂਕਿ ਹੋਟਲ ਪ੍ਰਬੰਧਨ ਬਦਲ ਗਿਆ, ਜਾਂ ਨਵੀਨੀਕਰਨ ਹੋਇਆ, ਜਾਂ ਇੱਕ ਤਲਾਬ ਸ਼ਾਮਲ ਕੀਤਾ ਗਿਆ ਆਦਿ)
- `Tags`
  - ਇਹ ਛੋਟੇ ਵਰਣਨ ਹਨ ਜੋ ਰਿਵਿਊਅਰ ਚੁਣ ਸਕਦੇ ਹਨ ਜਿਵੇਂ ਕਿ ਉਹ ਕਿਸ ਕਿਸਮ ਦੇ ਮਹਿਮਾਨ ਸਨ (ਜਿਵੇਂ ਕਿ ਸਿੰਗਲ ਜਾਂ ਪਰਿਵਾਰ), ਉਹਨਾਂ ਕੋਲ ਕਿਹੜਾ ਕਮਰਾ ਸੀ, ਰਹਿਣ ਦੀ ਮਿਆਦ ਅਤੇ ਰਿਵਿਊ ਕਿਵੇਂ ਜਮ੍ਹਾਂ ਕੀਤਾ ਗਿਆ ਸੀ।
  - ਦੁਖਦਾਈ ਗੱਲ ਇਹ ਹੈ ਕਿ ਇਹ ਟੈਗ ਵਰਤਣਾ ਮੁਸ਼ਕਲ ਹੈ, ਹੇਠਾਂ ਦਿੱਤੇ ਭਾਗ ਵਿੱਚ ਇਸ ਦੀ ਵਰਤੋਂਯੋਗਤਾ 'ਤੇ ਚਰਚਾ ਕੀਤੀ ਗਈ ਹੈ

**ਰਿਵਿਊਅਰ ਕਾਲਮ**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - ਇਹ ਸਿਫਾਰਸ਼ੀ ਮਾਡਲ ਵਿੱਚ ਇੱਕ ਕਾਰਕ ਹੋ ਸਕਦਾ ਹੈ, ਉਦਾਹਰਣ ਲਈ, ਜੇ ਤੁਸੀਂ ਇਹ ਨਿਰਧਾਰਤ ਕਰ ਸਕਦੇ ਹੋ ਕਿ ਸੈਂਕੜੇ ਰਿਵਿਊਜ਼ ਵਾਲੇ ਵੱਧ ਉਤਪਾਦਕ ਰਿਵਿਊਅਰ ਨਕਾਰਾਤਮਕ ਹੋਣ ਦੀ ਸੰਭਾਵਨਾ ਜ਼ਿਆਦਾ ਹੈ। ਹਾਲਾਂਕਿ, ਕਿਸੇ ਖਾਸ ਰਿਵਿਊ ਦੇ ਰਿਵਿਊਅਰ ਨੂੰ ਇੱਕ ਵਿਲੱਖਣ ਕੋਡ ਨਾਲ ਪਛਾਣ ਨਹੀਂ ਦਿੱਤੀ ਗਈ, ਅਤੇ ਇਸ ਲਈ ਇਸਨੂੰ ਰਿਵਿਊਜ਼ ਦੇ ਸੈੱਟ ਨਾਲ ਜੋੜਿਆ ਨਹੀਂ ਜਾ ਸਕਦਾ। 30 ਰਿਵਿਊਅਰ ਹਨ ਜਿਨ੍ਹਾਂ ਨੇ 100 ਜਾਂ ਵੱਧ ਰਿਵਿਊ ਦਿੱਤੇ ਹਨ, ਪਰ ਇਹ ਦੇਖਣਾ ਮੁਸ਼ਕਲ ਹੈ ਕਿ ਇਹ ਸਿਫਾਰਸ਼ੀ ਮਾਡਲ ਵਿੱਚ ਕਿਵੇਂ ਸਹਾਇਕ ਹੋ ਸਕਦਾ ਹੈ।
- `Reviewer_Nationality`
  - ਕੁਝ ਲੋਕ ਸੋਚ ਸਕਦੇ ਹਨ ਕਿ ਕੁਝ ਰਾਸ਼ਟਰੀਤਾਵਾਂ ਸਕਾਰਾਤਮਕ ਜਾਂ ਨਕਾਰਾਤਮਕ ਰਿਵਿਊ ਦੇਣ ਦੀ ਵੱਧ ਸੰਭਾਵਨਾ ਰੱਖਦੀਆਂ ਹਨ। ਆਪਣੇ ਮਾਡਲਾਂ ਵਿੱਚ ਇਸ ਤਰ੍ਹਾਂ ਦੇ ਅਨੁਮਾਨਿਕ ਦ੍ਰਿਸ਼ਟੀਕੋਣ ਸ਼ਾਮਲ ਕਰਨ ਤੋਂ ਸਾਵਧਾਨ ਰਹੋ। ਇਹ ਰਾਸ਼ਟਰੀ (ਅਤੇ ਕਈ ਵਾਰ ਨਸਲੀ) ਸਟਿਰਿਓਟਾਈਪ ਹਨ, ਅਤੇ ਹਰ ਰਿਵਿਊਅਰ ਇੱਕ ਵਿਅਕਤੀਗਤ ਸੀ ਜਿਸਨੇ ਆਪਣੇ ਅਨੁਭਵ ਦੇ ਆਧਾਰ 'ਤੇ ਰਿਵਿਊ ਲਿਖਿਆ। ਇਹ ਕਈ ਲੈਂਸਾਂ ਰਾਹੀਂ ਫਿਲਟਰ ਕੀਤਾ ਗਿਆ ਹੋ ਸਕਦਾ ਹੈ ਜਿਵੇਂ ਕਿ ਉਹਨਾਂ ਦੇ ਪਿਛਲੇ ਹੋਟਲ ਵਿੱਚ ਰਹਿਣ ਦੇ ਅਨੁਭਵ, ਯਾਤਰਾ ਦੀ ਦੂਰੀ, ਅਤੇ ਉਹਨਾਂ ਦਾ ਨਿੱਜੀ ਸੁਭਾਵ। ਇਹ ਸੋਚਣਾ ਕਿ ਉਹਨਾਂ ਦੇ ਰਾਸ਼ਟਰੀਤਾਵਾਂ ਦੇ ਕਾਰਨ ਰਿਵਿਊ ਸਕੋਰ ਸੀ, ਇਸਨੂੰ ਜਾਇਜ਼ ਠਹਿਰਾਉਣਾ ਮੁਸ਼ਕਲ ਹੈ।
🚨 ਧਿਆਨ ਦੇਣ ਯੋਗ ਗੱਲ

ਜਦੋਂ ਤੁਸੀਂ ਇਸ ਡੇਟਾਸੈਟ ਨਾਲ ਕੰਮ ਕਰ ਰਹੇ ਹੋ, ਤਾਂ ਤੁਸੀਂ ਐਸਾ ਕੋਡ ਲਿਖੋਗੇ ਜੋ ਟੈਕਸਟ ਤੋਂ ਕੁਝ ਗਣਨਾ ਕਰੇਗਾ ਬਿਨਾਂ ਇਸ ਟੈਕਸਟ ਨੂੰ ਖੁਦ ਪੜ੍ਹਨ ਜਾਂ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਨ ਦੀ ਲੋੜ ਹੋਵੇ। ਇਹ NLP ਦਾ ਮੁੱਖ ਸਿਧਾਂਤ ਹੈ, ਅਰਥ ਜਾਂ ਭਾਵਨਾ ਨੂੰ ਸਮਝਣਾ ਬਿਨਾਂ ਕਿਸੇ ਮਨੁੱਖੀ ਦਖਲ ਦੇ। ਹਾਲਾਂਕਿ, ਇਹ ਸੰਭਵ ਹੈ ਕਿ ਤੁਸੀਂ ਕੁਝ ਨਕਾਰਾਤਮਕ ਸਮੀਖਿਆਵਾਂ ਪੜ੍ਹ ਸਕਦੇ ਹੋ। ਮੈਂ ਤੁਹਾਨੂੰ ਇਹ ਕਰਨ ਤੋਂ ਰੋਕਣ ਦੀ ਸਲਾਹ ਦਿੰਦਾ ਹਾਂ, ਕਿਉਂਕਿ ਤੁਹਾਨੂੰ ਇਸ ਦੀ ਲੋੜ ਨਹੀਂ ਹੈ। ਕੁਝ ਸਮੀਖਿਆਵਾਂ ਮੂਰਖਤਾਪੂਰਨ ਜਾਂ ਅਸੰਬੰਧਤ ਨਕਾਰਾਤਮਕ ਹੋਟਲ ਸਮੀਖਿਆਵਾਂ ਹੁੰਦੀਆਂ ਹਨ, ਜਿਵੇਂ ਕਿ "ਮੌਸਮ ਚੰਗਾ ਨਹੀਂ ਸੀ", ਜੋ ਹੋਟਲ ਜਾਂ ਕਿਸੇ ਦੇ ਵਸ਼ ਵਿੱਚ ਨਹੀਂ ਹੈ। ਪਰ ਕੁਝ ਸਮੀਖਿਆਵਾਂ ਦਾ ਇੱਕ ਅੰਧਕਾਰ ਪਾਸਾ ਵੀ ਹੁੰਦਾ ਹੈ। ਕਈ ਵਾਰ ਨਕਾਰਾਤਮਕ ਸਮੀਖਿਆਵਾਂ ਵਿੱਚ ਜਾਤੀਵਾਦੀ, ਲਿੰਗਵਾਦੀ, ਜਾਂ ਉਮਰਵਾਦੀ ਟਿੱਪਣੀਆਂ ਹੁੰਦੀਆਂ ਹਨ। ਇਹ ਦੁਖਦਾਈ ਹੈ ਪਰ ਇੱਕ ਜਨਤਕ ਵੈਬਸਾਈਟ ਤੋਂ ਡੇਟਾ ਸਕ੍ਰੈਪ ਕਰਨ 'ਤੇ ਉਮੀਦ ਕੀਤੀ ਜਾ ਸਕਦੀ ਹੈ। ਕੁਝ ਸਮੀਖਿਆਕਾਰ ਅਜਿਹੀਆਂ ਸਮੀਖਿਆਵਾਂ ਛੱਡਦੇ ਹਨ ਜੋ ਤੁਹਾਨੂੰ ਅਸਹਜ, ਅਪ੍ਰਤੀਕਰਮ ਜਾਂ ਦੁਖੀ ਕਰ ਸਕਦੀਆਂ ਹਨ। ਇਹ ਬਿਹਤਰ ਹੈ ਕਿ ਕੋਡ ਭਾਵਨਾ ਨੂੰ ਮਾਪੇ ਬਜਾਏ ਕਿ ਤੁਸੀਂ ਖੁਦ ਉਹਨਾਂ ਨੂੰ ਪੜ੍ਹੋ ਅਤੇ ਦੁਖੀ ਹੋਵੋ। ਇਹ ਕਹਿਣ ਦੇ ਬਾਵਜੂਦ, ਇਹ ਘੱਟ ਸੰਖਿਆ ਵਾਲੇ ਲੋਕ ਹਨ ਜੋ ਅਜਿਹੀਆਂ ਗੱਲਾਂ ਲਿਖਦੇ ਹਨ, ਪਰ ਉਹ ਫਿਰ ਵੀ ਮੌਜੂਦ ਹਨ।
## ਕਸਰਤ - ਡਾਟਾ ਦੀ ਖੋਜ
### ਡਾਟਾ ਲੋਡ ਕਰੋ

ਡਾਟਾ ਨੂੰ ਦ੍ਰਿਸ਼ੀ ਰੂਪ ਵਿੱਚ ਦੇਖਣ ਲਈ ਕਾਫ਼ੀ ਹੈ, ਹੁਣ ਤੁਸੀਂ ਕੁਝ ਕੋਡ ਲਿਖੋਗੇ ਅਤੇ ਕੁਝ ਜਵਾਬ ਪ੍ਰਾਪਤ ਕਰੋਗੇ! ਇਸ ਭਾਗ ਵਿੱਚ pandas ਲਾਇਬ੍ਰੇਰੀ ਦੀ ਵਰਤੋਂ ਕੀਤੀ ਗਈ ਹੈ। ਤੁਹਾਡਾ ਸਭ ਤੋਂ ਪਹਿਲਾ ਕੰਮ ਇਹ ਯਕੀਨੀ ਬਣਾਉਣਾ ਹੈ ਕਿ ਤੁਸੀਂ CSV ਡਾਟਾ ਲੋਡ ਅਤੇ ਪੜ੍ਹ ਸਕਦੇ ਹੋ। pandas ਲਾਇਬ੍ਰੇਰੀ ਵਿੱਚ ਇੱਕ ਤੇਜ਼ CSV ਲੋਡਰ ਹੈ, ਅਤੇ ਨਤੀਜਾ ਇੱਕ ਡਾਟਾ ਫਰੇਮ ਵਿੱਚ ਰੱਖਿਆ ਜਾਂਦਾ ਹੈ, ਜਿਵੇਂ ਪਿਛਲੇ ਪਾਠਾਂ ਵਿੱਚ। ਜੋ CSV ਅਸੀਂ ਲੋਡ ਕਰ ਰਹੇ ਹਾਂ, ਉਸ ਵਿੱਚ ਅੱਧੇ ਮਿਲੀਅਨ ਤੋਂ ਵੱਧ ਪੰਕਤੀਆਂ ਹਨ, ਪਰ ਸਿਰਫ 17 ਕਾਲਮ ਹਨ। pandas ਤੁਹਾਨੂੰ ਡਾਟਾ ਫਰੇਮ ਨਾਲ ਸੰਚਾਰ ਕਰਨ ਦੇ ਬਹੁਤ ਸਾਰੇ ਸ਼ਕਤੀਸ਼ਾਲੀ ਤਰੀਕੇ ਦਿੰਦਾ ਹੈ, ਜਿਸ ਵਿੱਚ ਹਰ ਪੰਕਤੀ 'ਤੇ ਕਾਰਵਾਈ ਕਰਨ ਦੀ ਯੋਗਤਾ ਸ਼ਾਮਲ ਹੈ।

ਇਸ ਪਾਠ ਦੇ ਬਾਕੀ ਹਿੱਸੇ ਵਿੱਚ, ਕੋਡ ਸਨਿੱਪਟਸ ਹੋਣਗੇ ਅਤੇ ਕੋਡ ਦੇ ਕੁਝ ਵਿਆਖਿਆਨ ਅਤੇ ਨਤੀਜਿਆਂ ਦੇ ਅਰਥਾਂ ਬਾਰੇ ਕੁਝ ਚਰਚਾ ਹੋਵੇਗੀ। ਆਪਣੇ ਕੋਡ ਲਈ ਸ਼ਾਮਲ ਕੀਤੇ _notebook.ipynb_ ਦੀ ਵਰਤੋਂ ਕਰੋ।

ਆਓ ਉਸ ਡਾਟਾ ਫਾਈਲ ਨੂੰ ਲੋਡ ਕਰਨ ਨਾਲ ਸ਼ੁਰੂ ਕਰੀਏ ਜਿਸਦਾ ਤੁਸੀਂ ਇਸਤੇਮਾਲ ਕਰਨ ਜਾ ਰਹੇ ਹੋ:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

ਹੁਣ ਜਦੋਂ ਡਾਟਾ ਲੋਡ ਹੋ ਗਿਆ ਹੈ, ਅਸੀਂ ਇਸ 'ਤੇ ਕੁਝ ਕਾਰਵਾਈ ਕਰ ਸਕਦੇ ਹਾਂ। ਇਸ ਕੋਡ ਨੂੰ ਆਪਣੇ ਪ੍ਰੋਗਰਾਮ ਦੇ ਅਗਲੇ ਹਿੱਸੇ ਲਈ ਸਿਖਰ 'ਤੇ ਰੱਖੋ।

## ਡਾਟਾ ਦੀ ਖੋਜ

ਇਸ ਮਾਮਲੇ ਵਿੱਚ, ਡਾਟਾ ਪਹਿਲਾਂ ਹੀ *ਸਾਫ* ਹੈ, ਇਸਦਾ ਮਤਲਬ ਹੈ ਕਿ ਇਹ ਕੰਮ ਕਰਨ ਲਈ ਤਿਆਰ ਹੈ ਅਤੇ ਇਸ ਵਿੱਚ ਹੋਰ ਭਾਸ਼ਾਵਾਂ ਦੇ ਅੱਖਰ ਨਹੀਂ ਹਨ ਜੋ ਸਿਰਫ ਅੰਗਰੇਜ਼ੀ ਅੱਖਰਾਂ ਦੀ ਉਮੀਦ ਕਰਨ ਵਾਲੇ ਐਲਗੋਰਿਥਮ ਨੂੰ ਰੋਕ ਸਕਦੇ ਹਨ।

✅ ਤੁਹਾਨੂੰ ਸ਼ਾਇਦ ਅਜਿਹੇ ਡਾਟਾ ਨਾਲ ਕੰਮ ਕਰਨਾ ਪਵੇ ਜੋ NLP ਤਕਨੀਕਾਂ ਨੂੰ ਲਾਗੂ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ ਕੁਝ ਸ਼ੁਰੂਆਤੀ ਪ੍ਰੋਸੈਸਿੰਗ ਦੀ ਲੋੜ ਰੱਖਦਾ ਹੋਵੇ, ਪਰ ਇਸ ਵਾਰ ਨਹੀਂ। ਜੇ ਤੁਹਾਨੂੰ ਕਰਨਾ ਪਵੇ, ਤਾਂ ਤੁਸੀਂ ਗੈਰ-ਅੰਗਰੇਜ਼ੀ ਅੱਖਰਾਂ ਨੂੰ ਕਿਵੇਂ ਸੰਭਾਲੋਗੇ?

ਇੱਕ ਪਲ ਲਓ ਇਹ ਯਕੀਨੀ ਬਣਾਉਣ ਲਈ ਕਿ ਜਦੋਂ ਡਾਟਾ ਲੋਡ ਹੋ ਜਾਂਦਾ ਹੈ, ਤੁਸੀਂ ਇਸਨੂੰ ਕੋਡ ਨਾਲ ਖੋਜ ਸਕਦੇ ਹੋ। `Negative_Review` ਅਤੇ `Positive_Review` ਕਾਲਮਾਂ 'ਤੇ ਧਿਆਨ ਕੇਂਦਰਿਤ ਕਰਨਾ ਬਹੁਤ ਆਸਾਨ ਹੈ। ਇਹ ਕੁਦਰਤੀ ਟੈਕਸਟ ਨਾਲ ਭਰੇ ਹੋਏ ਹਨ ਜੋ ਤੁਹਾਡੇ NLP ਐਲਗੋਰਿਥਮ ਨੂੰ ਪ੍ਰਕਿਰਿਆ ਕਰਨ ਲਈ ਹਨ। ਪਰ ਰੁਕੋ! NLP ਅਤੇ ਭਾਵਨਾ ਵਿੱਚ ਛਾਲ ਮਾਰਨ ਤੋਂ ਪਹਿਲਾਂ, ਤੁਹਾਨੂੰ ਹੇਠਾਂ ਦਿੱਤੇ ਕੋਡ ਦੀ ਪਾਲਣਾ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ ਤਾਂ ਜੋ ਇਹ ਪਤਾ ਲਗਾਇਆ ਜਾ ਸਕੇ ਕਿ ਡਾਟਾਸੈਟ ਵਿੱਚ ਦਿੱਤੇ ਗਏ ਮੁੱਲ pandas ਨਾਲ ਗਣਨਾ ਕੀਤੇ ਮੁੱਲਾਂ ਨਾਲ ਮੇਲ ਖਾਂਦੇ ਹਨ ਜਾਂ ਨਹੀਂ।

## ਡਾਟਾ ਫਰੇਮ ਕਾਰਵਾਈਆਂ

ਇਸ ਪਾਠ ਵਿੱਚ ਪਹਿਲਾ ਕੰਮ ਇਹ ਜਾਂਚਣਾ ਹੈ ਕਿ ਹੇਠਾਂ ਦਿੱਤੇ ਦਾਅਵੇ ਸਹੀ ਹਨ ਜਾਂ ਨਹੀਂ, ਕੁਝ ਕੋਡ ਲਿਖ ਕੇ ਜੋ ਡਾਟਾ ਫਰੇਮ ਦੀ ਜਾਂਚ ਕਰਦਾ ਹੈ (ਇਸਨੂੰ ਬਦਲਣ ਤੋਂ ਬਿਨਾਂ)।

> ਬਹੁਤ ਸਾਰੇ ਪ੍ਰੋਗਰਾਮਿੰਗ ਕੰਮਾਂ ਦੀ ਤਰ੍ਹਾਂ, ਇਸਨੂੰ ਪੂਰਾ ਕਰਨ ਦੇ ਕਈ ਤਰੀਕੇ ਹਨ, ਪਰ ਚੰਗੀ ਸਲਾਹ ਇਹ ਹੈ ਕਿ ਇਸਨੂੰ ਸਭ ਤੋਂ ਸਧਾਰਨ, ਆਸਾਨ ਤਰੀਕੇ ਨਾਲ ਕਰੋ, ਖਾਸ ਕਰਕੇ ਜੇ ਇਹ ਭਵਿੱਖ ਵਿੱਚ ਇਸ ਕੋਡ 'ਤੇ ਵਾਪਸ ਆਉਣ 'ਤੇ ਸਮਝਣ ਲਈ ਆਸਾਨ ਹੋਵੇ। ਡਾਟਾ ਫਰੇਮਾਂ ਨਾਲ, ਇੱਕ ਵਿਸਤ੍ਰਿਤ API ਹੈ ਜੋ ਅਕਸਰ ਤੁਹਾਡੇ ਲਈ ਜੋ ਤੁਸੀਂ ਚਾਹੁੰਦੇ ਹੋ, ਕੁਸ਼ਲਤਾਪੂਰਵਕ ਕਰਨ ਦਾ ਤਰੀਕਾ ਹੋਵੇਗਾ।

ਹੇਠਾਂ ਦਿੱਤੇ ਪ੍ਰਸ਼ਨਾਂ ਨੂੰ ਕੋਡਿੰਗ ਟਾਸਕ ਵਜੋਂ ਮਾਨੋ ਅਤੇ ਹੱਲ ਦੇਖਣ ਤੋਂ ਬਿਨਾਂ ਉਨ੍ਹਾਂ ਦਾ ਜਵਾਬ ਦੇਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ।

1. ਤੁਸੀਂ ਹੁਣੇ ਹੀ ਲੋਡ ਕੀਤੇ ਡਾਟਾ ਫਰੇਮ ਦਾ *shape* ਪ੍ਰਿੰਟ ਕਰੋ (shape ਪੰਕਤੀਆਂ ਅਤੇ ਕਾਲਮਾਂ ਦੀ ਗਿਣਤੀ ਹੈ)
2. ਸਮੀਖਕ ਦੇਸ਼ਾਂ ਦੀ ਆਵ੍ਰਿਤੀ ਗਿਣਤੀ ਦੀ ਗਣਨਾ ਕਰੋ:
   1. `Reviewer_Nationality` ਕਾਲਮ ਲਈ ਕਿੰਨੇ ਵਿਲੱਖਣ ਮੁੱਲ ਹਨ ਅਤੇ ਉਹ ਕੀ ਹਨ?
   2. ਡਾਟਾਸੈਟ ਵਿੱਚ ਸਭ ਤੋਂ ਆਮ ਸਮੀਖਕ ਦੇਸ਼ ਕਿਹੜਾ ਹੈ (ਦੇਸ਼ ਅਤੇ ਸਮੀਖਾਵਾਂ ਦੀ ਗਿਣਤੀ ਪ੍ਰਿੰਟ ਕਰੋ)?
   3. ਅਗਲੇ 10 ਸਭ ਤੋਂ ਵੱਧ ਆਵ੍ਰਿਤੀ ਵਾਲੇ ਦੇਸ਼ ਕਿਹੜੇ ਹਨ, ਅਤੇ ਉਨ੍ਹਾਂ ਦੀ ਆਵ੍ਰਿਤੀ ਗਿਣਤੀ ਕੀ ਹੈ?
3. ਹਰ ਇੱਕ ਦੇਸ਼ ਲਈ ਸਭ ਤੋਂ ਵੱਧ ਸਮੀਖਾ ਕੀਤੇ ਗਏ ਹੋਟਲ ਕਿਹੜੇ ਹਨ ਜੋ 10 ਸਭ ਤੋਂ ਵੱਧ ਸਮੀਖਕ ਦੇਸ਼ਾਂ ਵਿੱਚ ਹਨ?
4. ਡਾਟਾਸੈਟ ਵਿੱਚ ਹਰ ਹੋਟਲ ਲਈ ਕਿੰਨੀ ਸਮੀਖਾਵਾਂ ਹਨ (ਹੋਟਲ ਦੀ ਆਵ੍ਰਿਤੀ ਗਿਣਤੀ)?
5. ਜਦੋਂ ਕਿ ਡਾਟਾਸੈਟ ਵਿੱਚ ਹਰ ਹੋਟਲ ਲਈ `Average_Score` ਕਾਲਮ ਹੈ, ਤੁਸੀਂ ਇੱਕ ਔਸਤ ਸਕੋਰ ਦੀ ਗਣਨਾ ਵੀ ਕਰ ਸਕਦੇ ਹੋ (ਡਾਟਾਸੈਟ ਵਿੱਚ ਹਰ ਹੋਟਲ ਲਈ ਸਾਰੇ ਸਮੀਖਕ ਸਕੋਰਾਂ ਦਾ ਔਸਤ ਪ੍ਰਾਪਤ ਕਰਨਾ)। ਆਪਣੇ ਡਾਟਾ ਫਰੇਮ ਵਿੱਚ `Calc_Average_Score` ਕਾਲਮ ਹੈਡਰ ਨਾਲ ਇੱਕ ਨਵਾਂ ਕਾਲਮ ਸ਼ਾਮਲ ਕਰੋ ਜਿਸ ਵਿੱਚ ਉਹ ਗਣਨਾ ਕੀਤੀ ਗਈ ਔਸਤ ਸ਼ਾਮਲ ਹੈ।
6. ਕੀ ਕੋਈ ਹੋਟਲ ਹਨ ਜਿਨ੍ਹਾਂ ਦੇ `Average_Score` ਅਤੇ `Calc_Average_Score` (1 ਦਸ਼ਮਲਵ ਸਥਾਨ 'ਤੇ ਗੋਲ ਕੀਤੇ) ਇੱਕੋ ਜਿਹੇ ਹਨ?
   1. ਇੱਕ Python ਫੰਕਸ਼ਨ ਲਿਖਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ ਜੋ ਇੱਕ Series (row) ਨੂੰ ਦਲੀਲ ਵਜੋਂ ਲੈਂਦਾ ਹੈ ਅਤੇ ਮੁੱਲਾਂ ਦੀ ਤੁਲਨਾ ਕਰਦਾ ਹੈ, ਜਦੋਂ ਮੁੱਲ ਇੱਕੋ ਨਹੀਂ ਹੁੰਦੇ ਤਾਂ ਸੁਨੇਹਾ ਪ੍ਰਿੰਟ ਕਰਦਾ ਹੈ। ਫਿਰ `.apply()` ਵਿਧੀ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਹਰ ਪੰਕਤੀ ਨੂੰ ਫੰਕਸ਼ਨ ਨਾਲ ਪ੍ਰਕਿਰਿਆ ਕਰੋ।
7. `Negative_Review` ਕਾਲਮ ਦੇ "No Negative" ਮੁੱਲਾਂ ਵਾਲੀਆਂ ਕਿੰਨੀ ਪੰਕਤੀਆਂ ਹਨ, ਇਹ ਗਿਣਨਾ ਕਰੋ ਅਤੇ ਪ੍ਰਿੰਟ ਕਰੋ।
8. `Positive_Review` ਕਾਲਮ ਦੇ "No Positive" ਮੁੱਲਾਂ ਵਾਲੀਆਂ ਕਿੰਨੀ ਪੰਕਤੀਆਂ ਹਨ, ਇਹ ਗਿਣਨਾ ਕਰੋ ਅਤੇ ਪ੍ਰਿੰਟ ਕਰੋ।
9. `Positive_Review` ਕਾਲਮ ਦੇ "No Positive" ਮੁੱਲਾਂ **ਅਤੇ** `Negative_Review` ਕਾਲਮ ਦੇ "No Negative" ਮੁੱਲਾਂ ਵਾਲੀਆਂ ਕਿੰਨੀ ਪੰਕਤੀਆਂ ਹਨ, ਇਹ ਗਿਣਨਾ ਕਰੋ ਅਤੇ ਪ੍ਰਿੰਟ ਕਰੋ।

### ਕੋਡ ਜਵਾਬ

1. ਤੁਸੀਂ ਹੁਣੇ ਹੀ ਲੋਡ ਕੀਤੇ ਡਾਟਾ ਫਰੇਮ ਦਾ *shape* ਪ੍ਰਿੰਟ ਕਰੋ (shape ਪੰਕਤੀਆਂ ਅਤੇ ਕਾਲਮਾਂ ਦੀ ਗਿਣਤੀ ਹੈ)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. ਸਮੀਖਕ ਦੇਸ਼ਾਂ ਦੀ ਆਵ੍ਰਿਤੀ ਗਿਣਤੀ ਦੀ ਗਣਨਾ ਕਰੋ:

   1. `Reviewer_Nationality` ਕਾਲਮ ਲਈ ਕਿੰਨੇ ਵਿਲੱਖਣ ਮੁੱਲ ਹਨ ਅਤੇ ਉਹ ਕੀ ਹਨ?
   2. ਡਾਟਾਸੈਟ ਵਿੱਚ ਸਭ ਤੋਂ ਆਮ ਸਮੀਖਕ ਦੇਸ਼ ਕਿਹੜਾ ਹੈ (ਦੇਸ਼ ਅਤੇ ਸਮੀਖਾਵਾਂ ਦੀ ਗਿਣਤੀ ਪ੍ਰਿੰਟ ਕਰੋ)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. ਅਗਲੇ 10 ਸਭ ਤੋਂ ਵੱਧ ਆਵ੍ਰਿਤੀ ਵਾਲੇ ਦੇਸ਼ ਕਿਹੜੇ ਹਨ, ਅਤੇ ਉਨ੍ਹਾਂ ਦੀ ਆਵ੍ਰਿਤੀ ਗਿਣਤੀ ਕੀ ਹੈ?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. ਹਰ ਇੱਕ ਦੇਸ਼ ਲਈ ਸਭ ਤੋਂ ਵੱਧ ਸਮੀਖਾ ਕੀਤੇ ਗਏ ਹੋਟਲ ਕਿਹੜੇ ਹਨ ਜੋ 10 ਸਭ ਤੋਂ ਵੱਧ ਸਮੀਖਕ ਦੇਸ਼ਾਂ ਵਿੱਚ ਹਨ?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. ਡਾਟਾਸੈਟ ਵਿੱਚ ਹਰ ਹੋਟਲ ਲਈ ਕਿੰਨੀ ਸਮੀਖਾਵਾਂ ਹਨ (ਹੋਟਲ ਦੀ ਆਵ੍ਰਿਤੀ ਗਿਣਤੀ)?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   ਤੁਸੀਂ ਧਿਆਨ ਦੇ ਸਕਦੇ ਹੋ ਕਿ *counted in the dataset* ਨਤੀਜੇ `Total_Number_of_Reviews` ਵਿੱਚ ਮੁੱਲਾਂ ਨਾਲ ਮੇਲ ਨਹੀਂ ਖਾਂਦੇ। ਇਹ ਸਪਸ਼ਟ ਨਹੀਂ ਹੈ ਕਿ ਕੀ ਡਾਟਾਸੈਟ ਵਿੱਚ ਇਹ ਮੁੱਲ ਹੋਟਲ ਦੇ ਕੁੱਲ ਸਮੀਖਾਵਾਂ ਦੀ ਨੁਮਾਇੰਦਗੀ ਕਰਦਾ ਹੈ, ਪਰ ਸਾਰੀਆਂ scraped ਨਹੀਂ ਕੀਤੀਆਂ ਗਈਆਂ, ਜਾਂ ਕੁਝ ਹੋਰ ਗਣਨਾ। `Total_Number_of_Reviews` ਨੂੰ ਮਾਡਲ ਵਿੱਚ ਇਸ ਅਸਪਸ਼ਟਤਾ ਦੇ ਕਾਰਨ ਵਰਤਿਆ ਨਹੀਂ ਜਾਂਦਾ।

5. ਜਦੋਂ ਕਿ ਡਾਟਾਸੈਟ ਵਿੱਚ ਹਰ ਹੋਟਲ ਲਈ `Average_Score` ਕਾਲਮ ਹੈ, ਤੁਸੀਂ ਇੱਕ ਔਸਤ ਸਕੋਰ ਦੀ ਗਣਨਾ ਵੀ ਕਰ ਸਕਦੇ ਹੋ (ਡਾਟਾਸੈਟ ਵਿੱਚ ਹਰ ਹੋਟਲ ਲਈ ਸਾਰੇ ਸਮੀਖਕ ਸਕੋਰਾਂ ਦਾ ਔਸਤ ਪ੍ਰਾਪਤ ਕਰਨਾ)। ਆਪਣੇ ਡਾਟਾ ਫਰੇਮ ਵਿੱਚ `Calc_Average_Score` ਕਾਲਮ ਹੈਡਰ ਨਾਲ ਇੱਕ ਨਵਾਂ ਕਾਲਮ ਸ਼ਾਮਲ ਕਰੋ ਜਿਸ ਵਿੱਚ ਉਹ ਗਣਨਾ ਕੀਤੀ ਗਈ ਔਸਤ ਸ਼ਾਮਲ ਹੈ। ਕਾਲਮ `Hotel_Name`, `Average_Score`, ਅਤੇ `Calc_Average_Score` ਪ੍ਰਿੰਟ ਕਰੋ।

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   ਤੁਸੀਂ `Average_Score` ਮੁੱਲ ਅਤੇ ਗਣਨਾ ਕੀਤੇ ਔਸਤ ਸਕੋਰ ਬਾਰੇ ਵੀ ਸੋਚ ਸਕਦੇ ਹੋ ਕਿ ਕਿਉਂਕਿ ਇਹ ਕਈ ਵਾਰ ਗਣਨਾ ਕੀਤੇ ਔਸਤ ਸਕੋਰ ਤੋਂ ਵੱਖਰੇ ਹੁੰਦੇ ਹਨ। ਜਿਵੇਂ ਕਿ ਅਸੀਂ ਨਹੀਂ ਜਾਣ ਸਕਦੇ ਕਿ ਕਿਉਂਕਿ ਕੁਝ ਮੁੱਲ ਮੇਲ ਖਾਂਦੇ ਹਨ, ਪਰ ਹੋਰਾਂ ਵਿੱਚ ਅੰਤਰ ਹੈ, ਇਸ ਮਾਮਲੇ ਵਿੱਚ ਸੁਰੱਖਿਅਤ ਹੈ ਕਿ ਅਸੀਂ ਆਪਣੇ ਆਪ ਔਸਤ ਦੀ ਗਣਨਾ ਕਰਨ ਲਈ ਸਮੀਖਾ ਸਕੋਰਾਂ ਦੀ ਵਰਤੋਂ ਕਰੀਏ। ਇਹ ਕਿਹਾ, ਅੰਤਰ ਆਮ ਤੌਰ 'ਤੇ ਬਹੁਤ ਛੋਟੇ ਹੁੰਦੇ ਹਨ, ਇੱਥੇ ਉਹ ਹੋਟਲ ਹਨ ਜਿਨ੍ਹਾਂ ਦੇ ਡਾਟਾਸੈਟ ਔਸਤ ਅਤੇ ਗਣਨਾ ਕੀਤੇ ਔਸਤ ਵਿੱਚ ਸਭ ਤੋਂ ਵੱਧ ਅੰਤਰ ਹੈ:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   ਸਿਰਫ 1 ਹੋਟਲ ਜਿਸਦਾ ਸਕੋਰ ਵਿੱਚ 1 ਤੋਂ ਵੱਧ ਅੰਤਰ ਹੈ, ਇਸਦਾ ਮਤਲਬ ਹੈ ਕਿ ਅਸੀਂ ਸ਼ਾਇਦ ਅੰਤਰ ਨੂੰ ਨਜ਼ਰਅੰਦਾਜ਼ ਕਰ ਸਕਦੇ ਹਾਂ ਅਤੇ ਗਣਨਾ ਕੀਤੇ ਔਸਤ ਸਕੋਰ ਦੀ ਵਰਤੋਂ ਕਰ ਸਕਦੇ ਹਾਂ।

6. `Negative_Review` ਕਾਲਮ ਦੇ "No Negative" ਮੁੱਲਾਂ ਵਾਲੀਆਂ ਕਿੰਨੀ ਪੰਕਤੀਆਂ ਹਨ, ਇਹ ਗਿਣਨਾ ਕਰੋ ਅਤੇ ਪ੍ਰਿੰਟ ਕਰੋ।

7. `Positive_Review` ਕਾਲਮ ਦੇ "No Positive" ਮੁੱਲਾਂ ਵਾਲੀਆਂ ਕਿੰਨੀ ਪੰਕਤੀਆਂ ਹਨ, ਇਹ ਗਿਣਨਾ ਕਰੋ ਅਤੇ ਪ੍ਰਿੰਟ ਕਰੋ।

8. `Positive_Review` ਕਾਲਮ ਦੇ "No Positive" ਮੁੱਲਾਂ **ਅਤੇ** `Negative_Review` ਕਾਲਮ ਦੇ "No Negative" ਮੁੱਲਾਂ ਵਾਲੀਆਂ ਕਿੰਨੀ ਪੰਕਤੀਆਂ ਹਨ, ਇਹ ਗਿਣਨਾ ਕਰੋ ਅਤੇ ਪ੍ਰਿੰਟ ਕਰੋ।

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## ਇੱਕ ਹੋਰ ਤਰੀਕਾ

ਲੈਂਬਡਾ ਤੋਂ ਬਿਨਾਂ ਆਈਟਮਾਂ ਦੀ ਗਿਣਤੀ ਕਰਨ ਦਾ ਇੱਕ ਹੋਰ ਤਰੀਕਾ, ਅਤੇ ਪੰਕਤੀਆਂ ਦੀ ਗਿਣਤੀ ਕਰਨ ਲਈ sum ਦੀ ਵਰਤੋਂ ਕਰੋ:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   ਤੁਸੀਂ ਧਿਆਨ ਦਿੱਤਾ ਹੋਵੇਗਾ ਕਿ 127 ਪੰਕਤੀਆਂ ਹਨ ਜਿਨ੍ਹਾਂ ਵਿੱਚ `Negative_Review` ਅਤੇ `Positive_Review` ਕਾਲਮਾਂ ਲਈ "No Negative" ਅਤੇ "No Positive" ਮੁੱਲ ਹਨ। ਇਸਦਾ ਮਤਲਬ ਹੈ ਕਿ ਸਮੀਖਕ ਨੇ ਹੋਟਲ ਨੂੰ ਇੱਕ ਅੰਕਤਮਕ ਸਕੋਰ ਦਿੱਤਾ, ਪਰ ਨਾ ਤਾਂ ਇੱਕ ਸਕਾਰਾਤਮਕ ਅਤੇ ਨਾ ਹੀ ਨਕਾਰਾਤਮਕ ਸਮੀਖਾ ਲਿਖਣ ਤੋਂ ਇਨਕਾਰ ਕੀਤਾ। ਖੁਸ਼ਕਿਸਮਤੀ ਨਾਲ ਇਹ ਪੰਕਤੀਆਂ ਦੀ ਛੋਟੀ ਗਿਣਤੀ ਹੈ (127 ਵਿੱਚੋਂ 515738, ਜਾਂ 0.02%), ਇਸ ਲਈ ਇਹ ਸ਼ਾਇਦ ਸਾਡੇ ਮਾਡਲ ਜਾਂ ਨਤੀਜਿਆਂ ਨੂੰ ਕਿਸੇ ਵਿਸ਼ੇਸ਼ ਦਿਸ਼ਾ ਵਿੱਚ ਝੁਕਾਉਣ ਵਾਲਾ ਨਹੀਂ ਹੋਵੇਗਾ, ਪਰ ਤੁਸੀਂ ਸ਼ਾਇਦ ਇੱਕ ਸਮੀਖਾਵਾਂ ਦੇ ਡਾਟਾ ਸੈੱਟ ਦੀ ਉਮੀਦ ਨਹੀਂ ਕੀਤੀ ਹੋਵੇਗੀ ਜਿਸ ਵਿੱਚ ਕੋਈ ਸਮੀਖਾਵਾਂ ਨਹੀਂ ਹਨ, ਇਸ ਲਈ ਇਹ ਡਾਟਾ ਦੀ ਖੋਜ ਕਰਨ ਲਈ ਮੁੱਲਾਂ ਦੀ ਖੋਜ ਕਰਨਾ ਯੋਗ ਹੈ।

ਹੁਣ ਜਦੋਂ ਤੁਸੀਂ ਡਾਟਾਸੈਟ ਦੀ ਖੋਜ ਕਰ ਲਈ ਹੈ, ਅਗਲੇ ਪਾਠ ਵਿੱਚ ਤੁਸੀਂ ਡਾਟਾ ਨੂੰ ਫਿਲਟਰ ਕਰੋਗੇ ਅਤੇ ਕੁਝ ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ ਸ਼ਾਮਲ ਕਰੋਗੇ।

---
## 🚀ਚੁਣੌਤੀ

ਜਿਵੇਂ ਕਿ ਅਸੀਂ ਪਿਛਲੇ ਪਾਠਾਂ ਵਿੱਚ ਵੇਖਿਆ, ਇਹ ਪਾਠ ਦਰਸਾਉਂਦਾ ਹੈ ਕਿ ਆਪਣੇ ਡਾਟਾ ਅਤੇ ਇਸਦੇ ਖਾਮੀਆਂ ਨੂੰ ਸਮਝਣਾ ਕਿੰਨਾ ਮਹੱਤਵਪੂਰਨ ਹੈ। ਖਾਸ ਕਰਕੇ ਟੈਕਸਟ-ਅਧਾਰਿਤ ਡਾਟਾ ਨੂੰ ਧਿਆਨ ਨਾਲ ਜਾਂਚਣ ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ। ਵੱਖ-ਵੱਖ ਟੈਕਸਟ-ਹੈਵੀ ਡਾਟਾਸੈਟਾਂ ਵਿੱਚ ਖੋਜ ਕਰੋ ਅਤੇ ਵੇਖੋ ਕਿ ਕੀ ਤੁਸੀਂ ਅਜਿਹੇ ਖੇਤਰਾਂ ਦੀ ਪਛਾਣ ਕਰ ਸਕਦੇ ਹੋ ਜੋ ਮਾਡਲ ਵਿੱਚ ਪੱਖਪਾਤ ਜਾਂ ਝੁਕਾਅ ਭਾਵਨਾ ਪੈਦਾ ਕਰ ਸਕਦੇ ਹਨ।

## [ਪੋਸਟ-ਪਾਠ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਸਮੀਖਾ ਅਤੇ ਸਵੈ-ਅਧਿਐਨ

[ਇਹ NLP ਲਰਨਿੰਗ ਪਾਥ](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952

---

**ਅਸਵੀਕਾਰਨਾ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀ ਹੋਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਦੇ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚੱਜੇਪਣ ਹੋ ਸਕਦੇ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।