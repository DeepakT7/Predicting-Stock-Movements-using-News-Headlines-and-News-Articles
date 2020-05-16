# Predicting Stock Movements using News Headlines and News Articles

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

 The main objective of the system is to analyse the future value of a certain stock of a particular company using the sentiment analysis and to predict whether a particular stock will go up that is whether it will increase or it will go down which means it will decrease on the basis of certain news headline, also detection of fake news and OCR was implemented for providing the user as an option for entering the news headline or a news article, the data we used was DJIA news headlines dataset and five different machine learning algorithms were used – Random Forest classifier, Naïve Bayes, Decision Tree, Logistic Regression and Support Vector Machine(SVM).
 
 
 Predicting stock prices and the status of stock market is a quite strenuous task in itself. Today stock prices of any company so not depend upon the financial factors of the company but also on the various other factors such as socio-economic factors and especially in this century the movement of stock prices are no more only linked with the current economic situation of the country rather the stock prices of the particular day are also directly or indirectly depends on the company related news, natural calamities as well as the political events. The motive of the research is to build a machine learning model which will predict whether the stock price of a company will go up or will go down and the model also predicts the exact stock prices for the next day and the day after based on the today’s news headlines of the company. We have taken Dow historical stock dataset of the years 2008-2016 which consists of the date, news headline, stock movement labelled as ‘1’ for increment and ‘0’ for decrement. The OCR model was also integrated with this model to make sure if the user is reading a headline on a newspaper or a different language newspaper, he/she should be able to know the price or movement of a stock just by clicking the picture of the headline on a newspaper of any language and upload that picture on the portal. 


Also the user can have a quick view on the real time stock history or stock prices jut by selecting the ticker and the no of days/months the user wants to see, the model will return real time graph of the selected day/month for stock price of the particular company of which the ticker has been selected. Also the system 
provides an option to upload the news headlines as well as the whole web-app in three different languages which are English, Hindi and Marathi.    

## Quickstart

Clone the git repository:
```console
$ git clone https://github.com/DeepakT7/Predicting-Stock-Movements-using-News-Headlines-and-News-Articles.git && cd Predicting-Stock-Movements-using-News-Headlines-and-News-Articles
```

Install necessary dependencies
```console
$ pip install -r requirements.txt
```

Start the developement server
```console
$ python app.py
```

Open the server running on

### [http://localhost:5000/](http://localhost:5000/)


## Screenshots

#### 1. Select a language

![image](https://github.com/DeepakT7/Predicting-Stock-Movements-using-News-Headlines-and-News-Articles/blob/master/Screenshots/Capture1.PNG)

#### 2. Prediction through OCR (Optical Character Recognition)

![image](https://github.com/DeepakT7/Predicting-Stock-Movements-using-News-Headlines-and-News-Articles/blob/master/Screenshots/Capture3.PNG) 
![image](https://github.com/DeepakT7/Predicting-Stock-Movements-using-News-Headlines-and-News-Articles/blob/master/Screenshots/Capture4.PNG)

#### 3. Prediction through Headline

![image](https://github.com/DeepakT7/Predicting-Stock-Movements-using-News-Headlines-and-News-Articles/blob/master/Screenshots/Capture6.PNG)

#### 4. Prediction through URL

![image](https://github.com/DeepakT7/Predicting-Stock-Movements-using-News-Headlines-and-News-Articles/blob/master/Screenshots/Capture10.PNG)

#### 5. Real Time Analysis

![image](https://github.com/DeepakT7/Predicting-Stock-Movements-using-News-Headlines-and-News-Articles/blob/master/Screenshots/Capture13.PNG)

#### 6. Accuracy Measure

![image](https://github.com/DeepakT7/Predicting-Stock-Movements-using-News-Headlines-and-News-Articles/blob/master/Screenshots/Capture15.PNG)
