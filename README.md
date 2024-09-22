# MediaEvalML
The modern world blurs the lines between what is real and what is misinformation. The increasing usage of bots and A.I to spread false narratives is a problem with high stakes. A large majority of people use websites such as X.com (Twitter) for news and for public discussion. How do we ensure, with the unfathomable amount of data uploaded onto Twitter each day, that misinformation isn't being maliciously spread onto the feeds of the users? One answer is via the use of fake detectors.

In this project, two ML pipelines are created for a dataset containing real and fake media. These algorithms aim to classify and distinguish the real/fake media in the test set. 

The MediaEval 2015 Dataset was used, consisting of imageless tweets. Fake and Real tweets are labelled in a binary fashion. 
Fake tweets consist of misinformation, recalling untrue events. 

In total, the dataset is composed of 18032 entries. The training to testing data is split 80/20.

The dataset also consists of 'humor' labels for tweets that are joking or unserious, these are also labelled fake.

The Jupyter Notebook shows:
- A statistical exploration of the dataset
- Preprocessing 
- Feature extraction
- Model training
- Performance analysis via. confusion matrix and F1 scores

Two algorithms were trained:
- Multinomial Naive Bayes Model 
- linearSVC

Both pipelines use the BOW (Bag-of-Words) and TF-IDF (Term Frequency - Inverse Document Frequency) extraction methods.

The MultinomialNB w/ TF-IDF pipeline was the most successful with 0.87 Accuracy score.
This is in line with values in similar research papers.

Source:
P. Jain, S. Sharma, Monica, and P. K. Aggarwal, “Classifying fake news detection
using svm, naive bayes and lstm,” in 2022 12th International Conference on Cloud
Computing, Data Science Engineering (Confluence), 2022, pp. 460–464

*Full Technical Report can be found in this repo.

