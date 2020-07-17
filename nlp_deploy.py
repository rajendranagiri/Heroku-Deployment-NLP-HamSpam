from flask import Flask, render_template, request, url_for
import pickle
import os

os.chdir('D:/DATA_SCIENCE/DEPLOYMENT/Heroku-Deployment-NLP-HamSpam')
# load the model from disk
nlp_file = 'nlp_model.pkl'
model = pickle.load(open(nlp_file, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app_nlp = Flask(__name__)

#First/Starting Web Page
@app_nlp.route('/')
def home():
	return render_template('home.html')
#Web page after clicking on POST button
@app_nlp.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
#    elif request.method == 'RESET':
#        message = request.form['message']
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app_nlp.run(debug=True)
