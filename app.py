from flask import Flask,request,render_template
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pickle,string,unicodedata,re

def process_data(txt):
    text_emoji = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    text_schar = re.compile('[^A-Za-z0-9 ]+')
    text_nums = re.compile('[0-9]+')    
    temp=text_emoji.sub(r'',txt)  
    temp=temp.translate(temp.maketrans('','',string.punctuation))          
    temp=text_schar.sub(r'',temp)
    temp=text_nums.sub(r'',temp)
    temp = unicodedata.normalize('NFKD',temp).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    temp = temp.lower()

    token = word_tokenize(temp)
    token_no_sw = [word for word in token if word not in stopwords.words()]
    lemmatizer = WordNetLemmatizer()
    lemm_token = [lemmatizer.lemmatize(word) for word in token_no_sw]
    temp = " ".join(lemm_token)
    return temp

def predict(txt):
    txt = process_data(txt)    
    vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
    x = vectorizer.transform([txt])
    pred = model.predict(x)
    pred_prob=model.predict_proba(x)
    if(pred):
        r="Positive"
    else:
        r="Negative"    
    return r, pred, pred_prob

app= Flask(__name__)
@app.route('/')
def home():
    return render_template("frontend.html")

@app.route('/result',methods=['POST'])
def result():
    x = request.form['feelings']
    predict_result,pred,pred_prob= predict(x)
    return render_template("frontend.html",r=predict_result,pred=pred,pred_prob=pred_prob)

if __name__=='__main__':
    app.run(debug=True)