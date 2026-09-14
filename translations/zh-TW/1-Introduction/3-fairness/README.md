# 建立負責任 AI 的機器學習解決方案
 
![機器學習中負責任 AI 的摘要手繪筆記](../../../../translated_images/zh-TW/ml-fairness.ef296ebec6afc98a.webp)
> 手繪筆記作者：[Tomomi Imura](https://www.twitter.com/girlie_mac)

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)
 
## 介紹

在本課程中，您將開始了解機器學習如何並且正在影響我們的日常生活。即使現在，系統和模型也參與日常決策任務，如醫療診斷、貸款核准或詐欺偵測。因此，這些模型有效運作以提供可被信賴的結果是很重要的。就如任何軟體應用，AI 系統也可能未達預期或產生不理想的結果。這就是為什麼了解並解釋 AI 模型的行為是必要的原因。

想像當您用來建立這些模型的數據缺少某些族群，如種族、性別、政治觀點、宗教，或是不成比例地代表這些族群，會發生什麼？如果模型的輸出被解讀為偏袒某些族群會怎麼樣？應用的後果是什麼？另外，當模型產生不良結果並對人們有害時會怎麼樣？誰應對 AI 系統的行為負責？這些是我們在本課程中將探索的一些問題。

在本課程中，您將：

- 提升您對機器學習中公平性重要性及相關傷害的意識。
- 熟悉探索異常值和不尋常情況以確保可靠性與安全性的做法。
- 理解設計包容系統以賦能所有人的必要性。
- 探索保護數據及個人隱私安全的重要性。
- 了解採用透明（玻璃盒）方法解釋 AI 模型行為的重要性。
- 注意問責制對於建立對 AI 系統信任的重要性。

## 預備知識

作為先修，請完成《負責任 AI 原則》學習路徑，並觀看以下相關主題影片：

透過此[學習路徑](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)深入了解負責任 AI

[![微軟負責任 AI 方法](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "微軟負責任 AI 方法")

> 🎥 點擊上方圖片觀看影片：微軟負責任 AI 方法

## 公平性

AI 系統應公平對待每個人，避免以不同方式影響相似群體。例如，當 AI 系統提供醫療治療建議、貸款申請或就業指引時，應針對具有相似症狀、財務狀況或專業資格的每個人給予相同的建議。我們每個人都攜帶著影響決策與行動的固有偏見。這些偏見可能明顯存在於用於訓練 AI 系統的數據中。有時這種操縱是無意的，而且通常難以自覺意識到何時在數據中引入了偏見。

**「不公平」** 指對某群體造成負面影響或「傷害」，例如按種族、性別、年齡或殘障狀況等分類。主要的公平性相關傷害可分為：

- <strong>分配不均</strong>，如性別或種族偏袒其中一方。
- <strong>服務品質</strong>。若只針對特定情境訓練數據，而現實更為複雜，將導致服務表現低劣。比如，一款洗手液分配器無法識別深膚色人員。[參考](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- <strong>貶損</strong>。不公平地批評和標籤某人或某事。例如，有影像標記技術曾臭名昭著錯誤將深色皮膚者標記為猩猩。
- <strong>過度或不足代表</strong>。某群體在某職業中看似不存在，而任何維持這種現象的服務或功能都會造成傷害。
- <strong>刻板印象</strong>。將既定屬性與特定群體連結。例如，英土語言翻譯系統可能因具有性別刻板關聯詞而出錯。

![翻譯成土耳其語](../../../../translated_images/zh-TW/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> 翻譯成土耳其語

![再翻譯回英文](../../../../translated_images/zh-TW/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> 再翻譯回英文

在設計和測試 AI 系統時，我們需確保 AI 公平，不受偏見或歧視決策控制，而人類亦被禁止做此類決策。要保證 AI 和機器學習的公平仍是複雜的社會技術挑戰。

### 可靠性與安全性

為建立信任，AI 系統需在正常及意外情況下保持可靠、安全和一致。了解 AI 在各種情境下（尤其是異常狀況）的行為很重要。建構 AI 解決方案時，必須大量聚焦如何處理 AI 可能遇到的各種狀況。例如，自駕車必須將人身安全列為最高優先。結果，自駕車所用 AI 需考慮所有可能面臨的情況，如夜晚、雷暴或暴風雪、小孩突然跑過馬路、寵物、道路施工等。AI 系統在寬廣情況下的可靠與安全處理能力，反映了資料科學家或 AI 開發者在系統設計與測試階段的預見程度。

> [🎥 點擊這裡觀看影片：](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### 包容性

AI 系統設計應涵蓋並賦能每個人。設計和實施 AI 系統時，資料科學家和 AI 開發者需發掘並消除可能無意排斥某些人的障礙。例如，全球約有 10 億身心障礙者。隨著 AI 進步，他們能更輕鬆地獲得資訊和機會。藉由消除障礙，促成創新，開發體驗更佳、惠及所有人的 AI 產品。

> [🎥 點擊這裡觀看影片：AI 的包容性](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### 安全與隱私

AI 系統需要安全且尊重個人隱私。民眾較不信任可能危害隱私、資訊或生命安全的系統。訓練機器學習模型時，我們依賴數據得出最佳結果，因此必須考慮數據來源與完整性。例如，數據是由用戶提交還是公開可用？在使用數據時，建立能保護機密信息且抵抗攻擊的 AI 系統至關重要。隨著 AI 越來越普及，保護隱私與重要個人及企業資訊變得愈加關鍵且複雜。AI 特別需要密切關注隱私和資料安全議題，因為 AI 系統獲取數據是做出準確且有根據判斷的基礎。

> [🎥 點擊這裡觀看影片：AI 的安全性](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- 整個產業在隱私與安全方面已有顯著進展，很大程度受 GDPR（一般資料保護規範）等法規推動。
- 但對 AI 系統而言，必須承認在提升系統個人化與效能與保障隱私間的緊張關係。
- 正如連網電腦誕生初期，我們也觀察到 AI 相關安全問題大幅增加。
- 同時，AI 也被用來強化安全。例如，大多數現代防毒掃描器如今以 AI 啟發式法則驅動。
- 我們必須確保數據科學流程與最新隱私及安全實踐和諧合作。


### 透明度
AI 系統應該是可理解的。透明度的關鍵在於解釋 AI 系統及其組件的行為。要改善對 AI 系統的理解，持份者需明白系統如何及為何運作，以便識別潛在的性能問題、安全與隱私疑慮、偏見、排斥行為或非預期結果。我們也認為 AI 使用者應坦誠說明何時、為何以及如何部署 AI 系統，以及他們所用系統的限制。例如，若銀行使用 AI 支援其消費者貸款決策，重要的是檢視結果並了解哪些數據影響系統推薦。政府開始針對各行業管控 AI，故資料科學家和組織需說明 AI 系統是否符合法規，特別是當出現不良結果時。

> [🎥 點擊這裡觀看影片：AI 的透明度](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- 由於 AI 系統十分複雜，理解其工作原理及解釋結果均不易。
- 這種理解不足影響系統的管理、運作和文件編制。
- 更重要的是，影響依據系統產出結果做決策的品質。

### 問責制
 
設計與部署 AI 系統的人必須對系統運作負責。問責制在使用敏感技術如臉部辨識時尤其重要。近期，臉部辨識技術需求成長，尤其來自執法機構，他們看到此技術在尋找失蹤兒童等用途的潛力。然而，這些技術也可能被政府用於威脅公民基本自由，例如持續監視特定個人。因此，資料科學家與組織需對其 AI 系統對個人或社會的影響負責。

[![領先 AI 研究者警告臉部辨識引發大規模監控](../../../../translated_images/zh-TW/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "微軟負責任 AI 方法")

> 🎥 點擊上方圖片觀看影片：警告臉部辨識引起大規模監控

最終，對於我們這一代—第一代將 AI 引入社會的人來說，最大的問題之一是如何確保電腦向人類負責，以及設計電腦的人對其他人負責。

## 影響評估

在訓練機器學習模型之前，進行影響評估很重要，以了解 AI 系統的目的、預期用途、部署地點及使用者。這有助於評估者或測試者知道評估系統時需考慮的因素，從而識別潛在風險與預期結果。

進行影響評估時的重點領域包括：

* <strong>對個人的不良影響</strong>。了解任何限制或要求、不被支持的使用方式或已知性能限制，對避免系統被用於可能傷害個人的方式至關重要。
* <strong>數據需求</strong>。理解系統如何及在哪裡使用數據，讓評審能探討需顧慮的數據要求（如 GDPR 或 HIPAA 規定），並檢視數據來源與數量是否充足以訓練模型。
* <strong>影響摘要</strong>。收集可能因使用系統而產生的傷害清單。在 ML 生命週期中，不斷檢視是否已緩解或處理所識別問題。
* <strong>適用目標</strong>，對六大核心原則的評估。檢查是否達成各原則目標，並找出不足之處。


## 負責任 AI 的除錯

類似除錯軟體應用，除錯 AI 系統是找出並解決系統問題的必要過程。多種因素可能導致模型表現不如預期或不負責任。大多數傳統模型性能指標是模型表現的量化彙整，無法充分分析模型如何違反負責任 AI 原則。此外，機器學習模型是黑盒，難以理解其決策依據或在出錯時提供解釋。本課程後面將學習如何使用負責任 AI 儀表板來協助除錯 AI 系統。該儀表板為資料科學家與 AI 開發者提供整體工具以執行：

* <strong>錯誤分析</strong>。識別影響系統公平性或可靠性的模型誤差分布。
* <strong>模型概覽</strong>。發現模型於不同資料群組間的性能差異。
* <strong>數據分析</strong>。了解數據分布，識別可能導致公平性、包容性與可靠性問題的偏誤。
* <strong>模型可解釋性</strong>。理解影響模型預測的因素，有助解釋模型行為，對透明度與問責性至關重要。


## 🚀 挑戰
 
為防止初期即引入傷害，我們應該：

- 在系統開發團隊中具備多元背景與觀點
- 投資反映社會多元性的數據集
- 在機器學習生命週期中發展更佳方法以偵測並修正負責任 AI 問題

思考實際情況中模型不可信的例子，從模型建立與使用中可看出哪些問題？我們還應考慮什麼？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學
 

在本課程中，您已經學習了機器學習中公平與不公平概念的一些基礎知識。  
 
觀看此工作坊以深入探討相關主題： 

- 追求負責任的 AI：將原則付諸實踐，由 Besmira Nushi、Mehrnoosh Sameki 和 Amit Sharma 主講

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 點擊上方圖片觀看影片：RAI Toolbox：由 Besmira Nushi、Mehrnoosh Sameki 和 Amit Sharma 主講的負責任 AI 建構開源框架

另外，閱讀： 

- 微軟的 RAI 資源中心：[Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- 微軟的 FATE 研究團隊：[FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox： 

- [Responsible AI Toolbox GitHub 儲存庫](https://github.com/microsoft/responsible-ai-toolbox)

閱讀關於 Azure Machine Learning 確保公平性的工具：

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## 作業

[探索 RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
此文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們努力追求準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於關鍵資訊，建議採用專業人工翻譯。我們不對因使用此翻譯所產生的任何誤解或誤譯承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->