from flask import Flask, request, redirect, url_for, flash, jsonify, render_template, Blueprint, session
from flask_mysqldb import MySQL
import numpy as np
import pickle as p
import json
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from newspaper import Article
from yahoo_fin import stock_info as si
from yahoo_fin.stock_info import get_analysts_info
from yahoo_fin.stock_info import *
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import itertools
import io
import base64
from flask_babel import Babel, _, gettext
from language_translate import translate
from ocr import ocr
from werkzeug.utils import secure_filename

##LANGUAGES = ['en', 'sv']

app = Flask(__name__)
# app = Flask(__name__) = pickle.load(open('model.pkl','rb'))
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'many random bytes'

##home_blueprint = Blueprint('home', __name__)

##app.config["BABEL_DEFAULT_LOCALE"] = LANGUAGES[0]
##babel = Babel(app)

lang = 'en'

cors = CORS(app)

modelfile = 'model.pkl'
model = p.load(open(modelfile, 'rb'))

##app.secret_key = 'some secret key'

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

##@babel.localeselector
##def get_locale():
##    if request.args.get("lang"):
##        session["lang"] = request.args.get("lang")
##    return session.get("lang", LANGUAGES[0])

@app.route('/')
def index():
    global lang
    lang = 'en'
    return render_template('base.html', what = translate("Select a language:", lang))

@app.route('/', methods=['POST'])
def main():
    #return translate("क्या", 'en')
    global lang
    lang = request.form.get('lang')
    #return lang
##    if(lang != 1):
##        session['lang'] = 'en'
##    else:
##        session['lang'] = 'sv'
    return render_template('base.html', what = translate("Select a language:", lang))

@app.route('/train')
def train():
    if(lang == 'en'):
        train = 'Train'
    else:
        train = 'Training'
    return render_template('train.html', Movements = translate('Predict Stock Movements', lang),Prediction = translate("Stock Market Prediction",lang),model = translate("Choose a model",lang), RF = translate("Random Forest Classifier",lang), DT = translate("Decision Tree Classifier",lang), LR = translate("Logistic Regression",lang), SVM = translate("Support Vector Machine",lang),Train = translate(train,lang), Accuracy = translate("Training Accuracy",lang),Matrix = translate("Confusion Matrix",lang))

@app.route('/accuracy', methods = ['POST'])
def accuracy():
    if(lang == 'en'):
        Train = 'Train'
    else:
        Train = 'Training'
    algorithm = request.form.get('algo')
    df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")
    train = df[df['Date'] < '20150101']
    test = df[df['Date'] > '20141231']
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

    global countvector
    countvector=CountVectorizer(ngram_range=(2,2))
    traindataset=countvector.fit_transform(headlines)

    if(algorithm == 'randomforest'):
        # implement RandomForest Classifier
        model=RandomForestClassifier(n_estimators=200,criterion='entropy')
        model.fit(traindataset,train['Label'])
    elif(algorithm == 'decisiontree'):
        model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        model.fit(traindataset, train['Label'])
    elif(algorithm == 'logistic'):
        model = LogisticRegression(random_state = 0)
        model.fit(traindataset, train['Label'])
    elif(algorithm == 'svm'):
        model = svm.SVC(random_state = 0)
        model.fit(traindataset, train['Label'])
        
    ## Predict for the Test Dataset
    test_transform= []
    for row in range(0,len(test.index)):
        test_transform.append(' '.join(str(x) for x in test.iloc[row,2:3]))
    test_dataset = countvector.transform(test_transform)
    predictions = model.predict(test_dataset)

    ## Import library to check accuracy
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

    matrix=confusion_matrix(test['Label'],predictions)
    score=accuracy_score(test['Label'],predictions)
    report=classification_report(test['Label'],predictions)

    score = str(round(score * 100,2))+'%'

    return render_template('train.html', accuracy=score, tables = [pd.DataFrame(matrix).to_html(classes='data')], titles=df.columns.values, Movements = translate('Predict Stock Movements', lang),Prediction = translate("Stock Market Prediction",lang),model = translate("Choose a model",lang), RF = translate("Random Forest Classifier",lang), DT = translate("Decision Tree Classifier",lang), LR = translate("Logistic Regression",lang), SVM = translate("Support Vector Machine",lang),Train = translate(Train,lang), Accuracy = translate("Training Accuracy",lang),Matrix = translate("Confusion Matrix",lang))

##def plot_confusion_matrix(matrix, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
##    """
##    This function prints and plots the confusion matrix.
##    Normalization can be applied by setting `normalize=True`.
##    """
##
##    img = io.BytesIO()
##    
##    if normalize:
##        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
##        print("Normalized confusion matrix")
##    else:
##        print('Confusion matrix, without normalization')
##
##    print(matrix)
##
##    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
##    plt.title(title)
##    plt.colorbar()
##    tick_marks = np.arange(len(classes))
##    plt.xticks(tick_marks, classes, rotation=45)
##    plt.yticks(tick_marks, classes)
##
##    fmt = '.2f' if normalize else 'd'
##    thresh = matrix.max() / 2.
##    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
##        plt.text(j, i, format(matrix[i, j], fmt),horizontalalignment="center",color="white" if matrix[i, j] > thresh else "black")
##
##    plt.tight_layout()
##    plt.ylabel('True label')
##    plt.xlabel('Predicted label')
##    plt.savefig(img, format='png')
##    img.seek(0)
##
##    plot_url = base64.b64encode(img.getvalue()).decode()
##
##    return plot_url
    
@app.route('/home')
def home():
    return render_template('index.html', Prediction = translate('Stock Market Prediction', lang), Predict = translate('Predict', lang), Plot = translate('plot', lang))

@app.route('/predict', methods = ['POST'])
def make_predict():
    #return jsonify(request.get_json())
    #data = request.get_json(force=True)
    #return '123'
    #predict_request = [str(x) for x in request.form.values()]
    #predict_request = np.array(predict_request)
    #delay_predicted = model.predict(predict_request)
    #output = predict(data['string'])
    #predict_request = [str(x) for x in request.form.values()]
    #predict_request = predict_request[0]
    #output = predict(translate(predict_request[0],'en'))
    output = predict(translate(str(request.form.get('headline')),'en'))
    #return '123'
    #return str(output)
    #return jsonify(str(output))
    #url = predict_request[1]
    url = str(request.form.get('url'))
    if(request.form.get('headline')!=''):
        if(output):
            result = "decrease"
        else:
            result = "increase"
##        return render_template('index.html', prediction_text='Stock Price will {}'.format(result))
    elif(request.form.get("url") != ''):
        toi_article = Article(url, language="en") # en for English
        toi_article.download()
        #toi_article.parse()
        output_2 = predict(toi_article.title)
        if(output_2):
            result = "decrease"
        else:
            result = "increase"
    else:
        file = request.files['file']
        #filename = secure_filename(file.filename)
        text = ocr(file.filename)
        output_3 = predict(text)
        if(output_3):
            result = "decrease"
        else:
            result = "increase"
    return render_template('index.html', prediction_text=translate('Stock Price will {}'.format(result),lang),Prediction = translate('Stock Market Prediction', lang), Predict = translate('Predict', lang), Plot = translate('plot', lang))
    
def predict(string):
    return model.predict(countvector.transform([string]))[0]
##    return model.predict(string)[0]
##    return model.predict_proba(string)[0][0]

@app.route('/plot', methods = ['POST'])
def graph1():
    from pandas_datareader import data
    days = [str(x) for x in request.form.values()]
    days_final=int(days[0])
    ticker=days[1]
    present=datetime.today().date()
    d =  datetime.today()- timedelta(days=days_final)
    start=d.date()
    start_date=start
    end_date=present
    data1 = data.get_data_yahoo(ticker, start_date, end_date)
    data1['Adj Close'].plot(figsize=(10, 7))
    plt.title("Adjusted Open Price of %s" % ticker, fontsize=16)
    plt.ylabel('Price', fontsize=14)
    plt.xlabel('date', fontsize=14)
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.savefig('C:/Users/rohit/Documents/Stock Market Prediction/static/plot1.png')
    return render_template('index.html', url = '/static/plot1.png',Prediction = translate('Stock Market Prediction', lang), Predict = translate('Predict', lang), Plot = translate('plot', lang))
    



if __name__ == '__main__':
    
    app.run(debug=True, port = 5000)
