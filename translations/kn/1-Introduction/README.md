<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "cf8ecc83f28e5b98051d2179eca08e08",
  "translation_date": "2025-12-19T12:57:04+00:00",
  "source_file": "1-Introduction/README.md",
  "language_code": "kn"
}
-->
# ಯಂತ್ರ ಅಧ್ಯಯನಕ್ಕೆ ಪರಿಚಯ

ಪಠ್ಯಕ್ರಮದ ಈ ವಿಭಾಗದಲ್ಲಿ, ನೀವು ಯಂತ್ರ ಅಧ್ಯಯನ ಕ್ಷೇತ್ರದ ಮೂಲ ತತ್ವಗಳನ್ನು ಪರಿಚಯಿಸಿಕೊಳ್ಳುತ್ತೀರಿ, ಅದು ಏನು ಮತ್ತು ಅದರ ಇತಿಹಾಸ ಮತ್ತು ಸಂಶೋಧಕರು ಅದನ್ನು ಬಳಸುವ ತಂತ್ರಗಳನ್ನು ತಿಳಿಯುತ್ತೀರಿ. ಬನ್ನಿ, ಈ ಹೊಸ ಯಂತ್ರ ಅಧ್ಯಯನ ಲೋಕವನ್ನು ಒಟ್ಟಿಗೆ ಅನ್ವೇಷಿಸೋಣ!

![globe](../../../translated_images/kn/globe.59f26379ceb40428.webp)
> ಫೋಟೋ <a href="https://unsplash.com/@bill_oxford?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ಬಿಲ್ ಆಕ್ಸ್ಫರ್ಡ್</a> ಅವರಿಂದ <a href="https://unsplash.com/s/photos/globe?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ಅನ್ಸ್ಪ್ಲ್ಯಾಶ್</a> ನಲ್ಲಿ
  
### ಪಾಠಗಳು

1. [ಯಂತ್ರ ಅಧ್ಯಯನಕ್ಕೆ ಪರಿಚಯ](1-intro-to-ML/README.md)
1. [ಯಂತ್ರ ಅಧ್ಯಯನ ಮತ್ತು ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯ ಇತಿಹಾಸ](2-history-of-ML/README.md)
1. [ನ್ಯಾಯ ಮತ್ತು ಯಂತ್ರ ಅಧ್ಯಯನ](3-fairness/README.md)
1. [ಯಂತ್ರ ಅಧ್ಯಯನದ ತಂತ್ರಗಳು](4-techniques-of-ML/README.md)
### ಕ್ರೆಡಿಟ್ಸ್

"ಯಂತ್ರ ಅಧ್ಯಯನಕ್ಕೆ ಪರಿಚಯ" ಅನ್ನು ♥️ ಸಹಿತ [ಮುಹಮ್ಮದ್ ಸಕಿಬ್ ಖಾನ್ ಇನಾನ್](https://twitter.com/Sakibinan), [ಓರ್ನೆಲ್ಲಾ ಅಲ್ಟುನ್ಯಾನ್](https://twitter.com/ornelladotcom) ಮತ್ತು [ಜೆನ್ ಲೂಪರ್](https://twitter.com/jenlooper) ಸೇರಿದಂತೆ ತಂಡದವರು ಬರೆದಿದ್ದಾರೆ

"ಯಂತ್ರ ಅಧ್ಯಯನದ ಇತಿಹಾಸ" ಅನ್ನು ♥️ ಸಹಿತ [ಜೆನ್ ಲೂಪರ್](https://twitter.com/jenlooper) ಮತ್ತು [ಏಮಿ ಬಾಯ್ಡ್](https://twitter.com/AmyKateNicho) ಬರೆದಿದ್ದಾರೆ

"ನ್ಯಾಯ ಮತ್ತು ಯಂತ್ರ ಅಧ್ಯಯನ" ಅನ್ನು ♥️ ಸಹಿತ [ಟೊಮೊಮಿ ಇಮುರಾ](https://twitter.com/girliemac) ಬರೆದಿದ್ದಾರೆ

"ಯಂತ್ರ ಅಧ್ಯಯನದ ತಂತ್ರಗಳು" ಅನ್ನು ♥️ ಸಹಿತ [ಜೆನ್ ಲೂಪರ್](https://twitter.com/jenlooper) ಮತ್ತು [ಕ್ರಿಸ್ ನೋರಿಂಗ್](https://twitter.com/softchris) ಬರೆದಿದ್ದಾರೆ

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ಅಸ್ವೀಕರಣ**:  
ಈ ದಸ್ತಾವೇಜು [Co-op Translator](https://github.com/Azure/co-op-translator) ಎಂಬ AI ಅನುವಾದ ಸೇವೆಯನ್ನು ಬಳಸಿ ಅನುವಾದಿಸಲಾಗಿದೆ. ನಾವು ಶುದ್ಧತೆಯತ್ತ ಪ್ರಯತ್ನಿಸುತ್ತಿದ್ದರೂ, ಸ್ವಯಂಚಾಲಿತ ಅನುವಾದಗಳಲ್ಲಿ ತಪ್ಪುಗಳು ಅಥವಾ ಅಸತ್ಯತೆಗಳು ಇರಬಹುದು ಎಂಬುದನ್ನು ದಯವಿಟ್ಟು ಗಮನಿಸಿ. ಮೂಲ ಭಾಷೆಯಲ್ಲಿರುವ ಮೂಲ ದಸ್ತಾವೇಜನ್ನು ಅಧಿಕೃತ ಮೂಲವೆಂದು ಪರಿಗಣಿಸಬೇಕು. ಮಹತ್ವದ ಮಾಹಿತಿಗಾಗಿ, ವೃತ್ತಿಪರ ಮಾನವ ಅನುವಾದವನ್ನು ಶಿಫಾರಸು ಮಾಡಲಾಗುತ್ತದೆ. ಈ ಅನುವಾದ ಬಳಕೆಯಿಂದ ಉಂಟಾಗುವ ಯಾವುದೇ ತಪ್ಪು ಅರ್ಥಮಾಡಿಕೊಳ್ಳುವಿಕೆ ಅಥವಾ ತಪ್ಪು ವಿವರಣೆಗಳಿಗೆ ನಾವು ಹೊಣೆಗಾರರಾಗುವುದಿಲ್ಲ.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->