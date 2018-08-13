import dill as pickle
from flask import Flask, request, jsonify
import numpy as np

with open("rand_for.pkl", "rb") as f:
	model = pickle.load(f)

with open("heat_clf.pkl", "rb") as h:
	heatmap = pickle.load(h)

def labeler(inp):
	if inp == -1:
		return 'A'
	elif inp == 0:
		return 'C'
	elif inp == 1:
		return 'I'

def request_handler(request):
	if request["method"] == "POST":
		if not request["form"]["batch"]:
			return model.predict([request["form"]["comment"]])
		else:
			return model.predict(request["form"]["comment"])


app = Flask(__name__)

@app.route("/predict",methods=['GET', 'POST'])
def predict():
	'''
	Handles: Predicting individual comment labels as well as heat. Limitations: Currently needs all 133 features to be calculated first. 
	Can be changed to use automatic data processing & LIWC analysis when needed.
	'''
	try:
		if request.method == "POST":
			#return request.get_data()
			if not request.form["batch"]:
				if not request.form["heat"]:
					#return str(model.predict([request.form["comment"]]))
					return str(model.predict([[float(i) for i in request.form["comment"][1:-1].split(',')]]))
				else:
					return str(heatmap.predict([[float(i) for i in request.form["comment"][1:-1].split(',')]]))
			else:
				results =[]
				if not request.form["heat"]:
					for comment in request.form["comment"].split('/'):
						try:
							results.append(model.predict([[float(i) for i in comment.split(',')]])[0])
						except:
							results.append('error for:' + str([[float(i) for i in comment.split(',')]]))
				else:
					for comment in request.form["comment"].split('/'):
						try:
							results.append(heatmap.predict([[float(i) for i in comment.split(',')]])[0])
						except:
							results.append('error for:' + str([[float(i) for i in comment.split(',')]]))
				return str(results)
		if request.method == 'GET':
			return 'Requests should be an array in the form: ["WC", "is_comment",  "parent_label", "post_count", "replies_count","mt_avg_sim","mp_avg_sim","mt_max_sim","mp_max_sim", "difficult_words", "data_type","num_sents"]'
	except:
		return "Your request was incorrectly formatted. Make sure it's an array of 27 arguments, which map to the 27 features in the model. Vectors should be separated by slashes and components separated by commas."
@app.route("/color",methods=['GET', 'POST'])
	

app.run(debug=True)