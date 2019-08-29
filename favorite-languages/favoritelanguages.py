#
# K-Nearest Neighbor - Favorite Programming Languages
#
# Author: Nasheb Ismaily
#

import pandas as pd
import matplotlib.pyplot as plt
import plot_state_borders as psb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('cities.csv', delimiter=',', header=0)
print(df.head())

# Data Exploration
plots = { "javascript" : ([], []),
          "python" : ([], []),
          "java" : ([], []),
          "c++" : ([], []),
          "swift" : ([], []),
          "sql" : ([], []),
          "ruby" : ([], []),
          "r" : ([], []),
          "php" : ([], []),
          "perl" : ([], []),
          "c#" : ([], []),
          "scala" : ([], []) }

colors = { "javascript" : "red",
          "python" : "green",
          "java" : "blue",
          "c++" : "orange",
          "swift" : "yellow",
          "sql" : "purple",
          "ruby" : "pink",
          "r" : "violet",
          "php" : "grey",
          "perl" : "brown",
          "c#" : "indigo",
          "scala" : "lime" }

for index, row in df.iterrows():
    language = row["language"]
    plots[language][0].append(row["longitude"])
    plots[language][1].append(row["latitude"])

f = plt.figure(1)
for lang, (x,y) in plots.items():
    plt.scatter(x, y, color=colors[lang],label=lang, zorder=10)

psb.plot_state_borders(plt)
plt.legend()
plt.axis([-130,-60,20,55]) # set the axes
plt.title("Favorite Programming Languages")

# Data Modeling
x_train, x_test, y_train, y_test = train_test_split(df[['longitude', 'latitude']].copy(),df["language"], test_size=0.3)

k = 6
model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

plots = { "javascript" : ([], []),
          "python" : ([], []),
          "java" : ([], []),
          "c++" : ([], []),
          "swift" : ([], []),
          "sql" : ([], []),
          "ruby" : ([], []),
          "r" : ([], []),
          "php" : ([], []),
          "perl" : ([], []),
          "c#" : ([], []),
          "scala" : ([], []) }

for longitude in range(-130, -60):
    for latitude in range(20, 55):
        predicted_language = model.predict([[longitude,latitude]])
        predicted_language=predicted_language.flat[0]

        plots[predicted_language][0].append(longitude)
        plots[predicted_language][1].append(latitude)

g = plt.figure(2)

for lang, (x,y) in plots.items():
    plt.scatter(x, y, color=colors[lang],label=lang, zorder=10)

psb.plot_state_borders(plt, color='black')
plt.legend(bbox_to_anchor=(1, 1), loc=0, borderaxespad=0.)
plt.axis([-130,-60,20,55]) # set the axes
plt.title("%s-Nearest Neighbor Programming Languages" % k)

plt.show()
