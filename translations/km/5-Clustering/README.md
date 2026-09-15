# គំរូក្លាស្តឺរីសម្រាប់ការរៀនម៉ាស៊ីន

ក្លាស្តឺរីគឺជាការប្រព្រឹត្តការងាររៀនម៉ាស៊ីនមួយដែលស្វែងរកវត្ថុដែលមានការខុសគ្នា និងបែងចែកវាទៅជាក្រុមដែលហៅថា cluster។ អ្វីដែលបញ្ជាក់ភាពខុសគ្នារវាងក្លាស្តឺរីជាមួយវិធីសាស្រ្តផ្សេងទៀតក្នុងការរៀនម៉ាស៊ីន គឺវាកើតឡើងដោយស្វ័យប្រវត្តិ ជាក់លាក់ជាអ្វីដែលអាចនិយាយថា វាជា វិធីដែលផ្ទុយពីការរៀនដែលមានការត្រួតពិនិត្យ។

## ប្រធានបទតំបន់៖ គំរូក្លាស្តឺរីសម្រាប់អារម្មណ៍តន្ត្រីរបស់សាធារណជននីហ្សេរី 🎧

សាធារណជននីហ្សេរីមានរសជាតិតន្ត្រីជាច្រើន។ ដោយប្រើ​ទិន្នន័យដែលបានប្រមូលពីSpotify (ដោយយកចិត្តទុកដាក់លើ [អត្ថបទនេះ](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)) អ្នកនឹងមើលឃើញតន្ត្រីដែលពេញនិយមនៅក្នុងនីហ្សេរី។ ឌាតាសែលនេះរួមបញ្ចូលទិន្នន័យអំពីពិន្ទុ 'danceability' របស់បទចម្រៀង ពិន្ទុ 'acousticness' ភាពខ្លាំង សំឡេងនិយាយ ភាពពេញនិយម និងថាមពល។ វានឹងគួរឲ្យចាប់អារម្មណ៍ក្នុងការរកឃើញគំរូនៅក្នុងទិន្នន័យនេះ!

![A turntable](../../../translated_images/km/turntable.f2b86b13c53302dc.webp)

> រូបថតដោយ <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> នៅលើ <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
ក្នុងស៊េរីមេរៀននេះ អ្នកនឹងបានស្វែងយល់ពីវិធីថ្មីៗក្នុងការវិភាគទិន្នន័យដោយប្រើបច្ចេកវិធីក្លាស្តឺរី។ ក្លាស្តឺរីមានប្រយោជន៍ពិសេសនៅពេលដែលឌាតាតាមអ្នកគ្មានស្លាកទិន្នន័យ ប្រសិនបើវាមានស្លាកទិន្នន័យវិធីសាស្រ្តចំរុះដូចនេះដែលអ្នកបានរៀននៅមេរៀនមុនអាចមានប្រយោជន៍ជាង។ ប៉ុន្តែនៅក្នុងករណីដែលអ្នកស្វែងរកការបែងចែកឌាតាដែលគ្មានស្លាកទិន្នន័យ ក្លាស្តឺរីគឺជាវិធីល្អក្នុងការស្វែងរកគំរូ។

> មានឧបករណ៍តិចកូដដែលមានប្រយោជន៍ជួយឲ្យអ្នករៀនពីការងារជាមួយគំរូក្លាស្តឺរី។ សាកល្បង [Azure ML សម្រាប់ភារកិច្ចនេះ](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## មេរៀន

1. [ការណែនាំអំពីក្លាស្តឺរី](1-Visualize/README.md)
2. [ក្លាស្តឺរី K-Means](2-K-Means/README.md)

## ឯកឧត្តម

មេរៀនទាំងនេះត្រូវបានសរសេរទៅជាមួយនឹង​សំឡេងតន្ត្រី 🎶 ដោយ [Jen Looper](https://www.twitter.com/jenlooper) និងមានការពិនិត្យយ៉ាងល្អពី [Rishit Dagli](https://rishit_dagli/) និង [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan)។

ឌាតាសែល [បទចម្រៀងនីហ្សេរី](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ត្រូវបានយកចេញពី Kaggle ដោយប្រមូលពី Spotify។

ឧទាហរណ៍ K-Means ដែលមានប្រយោជន៍ ដែលបានជួយក្នុងការបង្កើតមេរៀននេះរួមមាន [ការស្រាវជ្រាវលើផ្កា iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [សៀវភៅកំណត់ហេតុចាប់ផ្តើម](https://www.kaggle.com/prashant111/k-means-clustering-with-python) និង [ឧទាហរណ៍ NGO ស្មាន](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)។

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ការបដិសេធ**:
ឯកសារនេះត្រូវបានបម្លែងភាសា ដោយប្រើសេវាបម្លែងភាសា AI [Co-op Translator](https://github.com/Azure/co-op-translator)។ ទោះយើងខ្ញុំមានក្តីប្រាថ្នាឱ្យបានច្បាស់លាស់ តែសូមយល់ដឹងថាការបម្លែងដោយស្វ័យប្រវត្តិក៏អាចមានកំហុសឬភាពមិនត្រឹមត្រូវ។ ឯកសារដើមជាភាសាទីតាំងគួរត្រូវបានគេប្រើជាប្រភពច្បាស់លាស់។ សម្រាប់ព័ត៌មានសំខាន់ៗ សូមណែនាំឱ្យប្រើប្រាស់ការប្រែដោយមនុស្សជំនាញ។ យើងខ្ញុំមិនទទួលខុសត្រូវចំពោះការយល់ច្រឡំ ឬការបកស្រាយខុសបន្ទាប់ពីការប្រើប្រាស់ការបម្លែងនេះនោះទេ។
<!-- CO-OP TRANSLATOR DISCLAIMER END -->