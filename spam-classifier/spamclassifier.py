#
# SPAM Classifier
#
# Author: Nasheb Ismaily
#

import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import collections

df = pd.read_csv('smsspamcollection.txt',
                 delimiter='\t', header=None)

print(df.head())

# Create wordcloud for SPAM words
spam_words = df.loc[df[0] == 'spam'][1]
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                        background_color='black',
                        stopwords=stopwords,
                        max_words=200,
                        max_font_size=40,
                        random_state=42
                    ).generate(str(spam_words))

plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Train / Test Model
x_train, x_test, y_train, y_test = train_test_split(df[1],df[0])

stopwords = set(STOPWORDS)
vectorizer = TfidfVectorizer(stop_words=stopwords)
x_train_v = vectorizer.fit_transform(x_train)
x_test_v = vectorizer.transform(x_test)
classifier = LogisticRegression()
classifier.fit(x_train_v, y_train)

predictions = classifier.predict(x_test_v)

results = pd.DataFrame(
    {'Prediction': pd.Series(predictions).tolist(),
     'Message': x_test.tolist()
    })

for index, row in results.head(n=20).iterrows():
     print('Prediction: %s. Message: %s' % (row['Prediction'], row['Message']))

scores = cross_val_score(classifier, x_train_v, y_train, cv=5)
print(np.mean(scores), scores)

target_names = ['ham', 'spam']
print(classification_report(y_test, predictions, target_names=target_names))

# Calculate the fpr and tpr for all thresholds of the classification
preds = [1 if x == "spam" else 0 for x in predictions]

false_positive_rate, recall, thresholds = roc_curve(y_test,preds,pos_label="spam")
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()

print("TEST DATA")
print(y_test.value_counts())
print("Predictions")

