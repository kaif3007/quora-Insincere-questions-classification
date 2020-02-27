import secrets,os
from flask import Flask,render_template,current_app,redirect,url_for
from forms import PostForm
from nlp import predict_sentiment


app=Flask(__name__)
app.config['SECRET_KEY']='5791628bb0b13ce0c676dfde280ba245'


@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
	form=PostForm()
	ans=""
	pred_prob=0
	if form.validate_on_submit():
		ques=form.Question.data
		ans,pred_prob=predict_sentiment(ques)
	return render_template('compressed.html',title='result',ans=ans,pred_prob=pred_prob,form=form)


if __name__=='__main__':
	app.run(debug=True)
