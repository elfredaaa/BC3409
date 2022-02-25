#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
from joblib import load


# In[2]:


app=Flask(__name__)


# In[3]:


from flask import request, render_template 
import joblib 
 
@app.route("/", methods = ["GET", "POST"]) 
def index(): 
    if request.method == "POST": 
        income = request.form.get("income") 
        age = request.form.get("age") 
        loan = request.form.get("loan") 
        print(income, age, loan) 
         
        model_log = joblib.load("LR") 
        model_cart = joblib.load("CART") 
        model_rf = joblib.load("RF") 
        model_xgb = joblib.load("XGB") 
        model_nn = joblib.load("NNMLP") 
         
        pred_log = model_log.predict([[float(income), float(age), float(loan)]]) 
        pred_cart = model_cart.predict([[float(income), float(age), float(loan)]]) 
        pred_rf = model_rf.predict([[float(income), float(age), float(loan)]]) 
        pred_xgb = model_xgb.predict([[float(income), float(age), float(loan)]]) 
        pred_nn = model_nn.predict([[float(income), float(age), float(loan)]]) 
         
        print(pred_log) 
        print(pred_cart) 
        print(pred_rf) 
        print(pred_xgb) 
        print(pred_nn) 
         
        pred_log = pred_log[0] 
        pred_cart = pred_cart[0] 
        pred_rf = pred_rf[0] 
        pred_xgb = pred_xgb[0] 
        pred_nn = pred_nn[0] 
         
        log = "The predicted default on credit card is " + str(pred_log) 
        cart = "The predicted default on credit card is " + str(pred_cart) 
        rf = "The predicted default on credit card is " + str(pred_rf) 
        xgb = "The predicted default on credit card is " + str(pred_xgb) 
        nn = "The predicted default on credit card is " + str(pred_nn) 
         
        return(render_template("index.html", result1 = log, result2 = cart, result3 = rf, result4 = xgb, result5 = nn)) 
    else:  
        return(render_template("index.html", result1 = "", result2 = "", result3 = "", result4 = "", result5 = ""))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




