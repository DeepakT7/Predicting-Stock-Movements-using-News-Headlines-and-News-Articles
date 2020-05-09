import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from pandas_datareader import data
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from yellowbrick.model_selection import CVScores


df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")

##features = list(df.columns[2:3])
###print(features)
##x = df[features]
##y = df["Label"]

##print(x,y)
 
##num_of_rows = int(4100 * 0.8)
##train = df[:num_of_rows] 
##test = df[num_of_rows:]

##train_x, test_x, train_y, test_y = train_test_split(x,y,stratify = y, random_state =7)

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

#kf = KFold(len(test), 5, True, 42)
#clf = make_pipeline(SVC())
##cv = StratifiedKFold(n_splits=5, random_state=42)

##train = df[df['Date'] < '20110101']
##test = df[df['Date'] > '20101231']
#print(test)
##    # Removing punctuations
data=train.iloc[:,2:3]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(1)]
new_Index=[str(i) for i in list1]
data.columns= new_Index

# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()

##    ' '.join(str(x) for x in data.iloc[1,0:1])

headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:1]))

countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)
##model=RandomForestClassifier(n_estimators=200,criterion='entropy')
##model.fit(traindataset,train['Label'])
##model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
##model.fit(traindataset, train['Label'])
##model = LogisticRegression(random_state = 0)
##model.fit(traindataset, train['Label'])
##model = svm.SVC(random_state = 0)
##model.fit(traindataset, train['Label'])
model = GaussianNB()
model.fit(traindataset.toarray(), train['Label'])

## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:3]))
test_dataset = countvector.transform(test_transform)
predictions = model.predict(test_dataset)

##score = cross_val_score(model,train_x,train_y,cv = 5)
score = accuracy_score(test['Label'],predictions)

print(score)
