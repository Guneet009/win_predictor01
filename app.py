from flask import Flask,render_template,request 
import pandas as pd


app = Flask(__name__)
features=[]
df = pd.read_csv("ipl_Data//ipl_data.csv")

all_features = [col for col in df.columns]

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html',features = all_features)
