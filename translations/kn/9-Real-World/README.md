<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5e069a0ac02a9606a69946c2b3c574a9",
  "translation_date": "2025-12-19T13:06:11+00:00",
  "source_file": "9-Real-World/README.md",
  "language_code": "kn"
}
-->
# ಪೋಸ್ಟ್‌ಸ್ಕ್ರಿಪ್ಟ್: ಕ್ಲಾಸಿಕ್ ಮೆಷಿನ್ ಲರ್ನಿಂಗ್‌ನ ನೈಜ ಜಗತ್ತಿನ ಅನ್ವಯಿಕೆಗಳು

ಪಠ್ಯಕ್ರಮದ ಈ ವಿಭಾಗದಲ್ಲಿ, ನೀವು ಶ್ರೇಷ್ಟ ML ನ ಕೆಲವು ನೈಜ ಜಗತ್ತಿನ ಅನ್ವಯಿಕೆಗಳನ್ನು ಪರಿಚಯಿಸಿಕೊಳ್ಳುತ್ತೀರಿ. ನಾವು ಇಂಟರ್ನೆಟ್ ಅನ್ನು ಹುಡುಕಿ ಈ ತಂತ್ರಗಳನ್ನು ಬಳಸಿದ ಅನ್ವಯಿಕೆಗಳ ಬಗ್ಗೆ ಶ್ವೇತಪತ್ರಗಳು ಮತ್ತು ಲೇಖನಗಳನ್ನು ಕಂಡುಹಿಡಿದಿದ್ದೇವೆ, ನ್ಯೂರಲ್ ನೆಟ್‌ವರ್ಕ್‌ಗಳು, ಡೀಪ್ ಲರ್ನಿಂಗ್ ಮತ್ತು AI ಅನ್ನು ಸಾಧ್ಯವಾದಷ್ಟು ತಪ್ಪಿಸಿ. ವ್ಯವಹಾರ ವ್ಯವಸ್ಥೆಗಳು, ಪರಿಸರ ಅನ್ವಯಿಕೆಗಳು, ಹಣಕಾಸು, ಕಲೆ ಮತ್ತು ಸಂಸ್ಕೃತಿ ಮತ್ತು ಇನ್ನಷ್ಟು ಕ್ಷೇತ್ರಗಳಲ್ಲಿ ML ಹೇಗೆ ಬಳಸಲಾಗುತ್ತದೆ ಎಂಬುದನ್ನು ತಿಳಿಯಿರಿ.

![chess](../../../translated_images/kn/chess.e704a268781bdad8.jpg)

> ಫೋಟೋ <a href="https://unsplash.com/@childeye?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Alexis Fauvet</a> ಅವರಿಂದ <a href="https://unsplash.com/s/photos/artificial-intelligence?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> ನಲ್ಲಿ
  
## ಪಾಠ

1. [ML ಗಾಗಿ ನೈಜ ಜಗತ್ತಿನ ಅನ್ವಯಿಕೆಗಳು](1-Applications/README.md)
2. [ಜವಾಬ್ದಾರಿಯುತ AI ಡ್ಯಾಶ್‌ಬೋರ್ಡ್ ಘಟಕಗಳನ್ನು ಬಳಸಿಕೊಂಡು ಮೆಷಿನ್ ಲರ್ನಿಂಗ್‌ನಲ್ಲಿ ಮಾದರಿ ಡಿಬಗಿಂಗ್](2-Debugging-ML-Models/README.md)

## ಕ್ರೆಡಿಟ್‌ಗಳು

"ನೈಜ ಜಗತ್ತಿನ ಅನ್ವಯಿಕೆಗಳು" ಅನ್ನು [ಜೆನ್ ಲೂಪರ್](https://twitter.com/jenlooper) ಮತ್ತು [ಒರ್ನೆಲ್ಲಾ ಅಲ್ಟುನ್ಯಾನ್](https://twitter.com/ornelladotcom) ಸೇರಿದಂತೆ ತಂಡದವರು ಬರೆಯಲಾಗಿದೆ.

"ಜವಾಬ್ದಾರಿಯುತ AI ಡ್ಯಾಶ್‌ಬೋರ್ಡ್ ಘಟಕಗಳನ್ನು ಬಳಸಿಕೊಂಡು ಮೆಷಿನ್ ಲರ್ನಿಂಗ್‌ನಲ್ಲಿ ಮಾದರಿ ಡಿಬಗಿಂಗ್" ಅನ್ನು [ರೂತ್ ಯಾಕುಬು](https://twitter.com/ruthieyakubu) ಬರೆಯಲಾಗಿದೆ.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ಅಸ್ವೀಕರಣ**:  
ಈ ದಸ್ತಾವೇಜು [Co-op Translator](https://github.com/Azure/co-op-translator) ಎಂಬ AI ಅನುವಾದ ಸೇವೆಯನ್ನು ಬಳಸಿ ಅನುವಾದಿಸಲಾಗಿದೆ. ನಾವು ಶುದ್ಧತೆಯತ್ತ ಪ್ರಯತ್ನಿಸುತ್ತಿದ್ದರೂ, ಸ್ವಯಂಚಾಲಿತ ಅನುವಾದಗಳಲ್ಲಿ ತಪ್ಪುಗಳು ಅಥವಾ ಅಸತ್ಯತೆಗಳು ಇರಬಹುದು ಎಂದು ದಯವಿಟ್ಟು ಗಮನಿಸಿ. ಮೂಲ ಭಾಷೆಯಲ್ಲಿರುವ ಮೂಲ ದಸ್ತಾವೇಜನ್ನು ಅಧಿಕೃತ ಮೂಲವೆಂದು ಪರಿಗಣಿಸಬೇಕು. ಮಹತ್ವದ ಮಾಹಿತಿಗಾಗಿ, ವೃತ್ತಿಪರ ಮಾನವ ಅನುವಾದವನ್ನು ಶಿಫಾರಸು ಮಾಡಲಾಗುತ್ತದೆ. ಈ ಅನುವಾದ ಬಳಕೆಯಿಂದ ಉಂಟಾಗುವ ಯಾವುದೇ ತಪ್ಪು ಅರ್ಥಮಾಡಿಕೊಳ್ಳುವಿಕೆ ಅಥವಾ ತಪ್ಪು ವಿವರಣೆಗಳಿಗೆ ನಾವು ಹೊಣೆಗಾರರಾಗುವುದಿಲ್ಲ.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->