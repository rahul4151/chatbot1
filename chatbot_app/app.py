# usr name = rahulghoshchatbot
# pass = Smart@123
# pip install flask
# pip install scikit-learn
# mkvirtualenv --python=/usr/bin/python3.7 neha
# wornon neha
# 



from flask import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/",methods = ["GET","POST"])
def home():
    if request.method == "POST":    
        data = pd.read_csv("conv.csv")

        cv = CountVectorizer()
        vector = cv.fit_transform(data["qts"])
        qts = request.form["qts"]
        vqts = cv.transform([qts])
        cs = cosine_similarity(vqts,vector)
        fcs = cs.flatten()
        indices = np.argpartition(fcs,-1)[-1:]
        res = data.iloc[indices]
        msg = " ".join(res["ans"])
        tc = request.form["tc"]
        tc = tc +"qts = "+str(qts) + "\nchitty = "+str(msg) + "\n"
        return render_template("home.html",msg = msg,tc= tc)


    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True,use_reloader = True)