import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, cohen_kappa_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import sparse
import itertools
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import FastText
import datetime

dataframe = pd.read_csv('C:/Users/ktwic/Desktop/with_predictions.csv')
# twentyeighteen = pd.read_csv('C:/Users/ktwic/Desktop/lol.csv')
# twentyeighteen = twentyeighteen.drop(columns=['Unnamed: 0', 'comment_id', 'parent_id', 'author_id', 'location_id','body', 'marked_par', 'ctime',  'marked_text', 'type', 'signed', 'deleted', 'moderated', 'path1', 'offset1', 'path2', 'offset2', 'url' ,'id.1' ,'name' ,'hashtag_idea' ,'hashtag_question', 'hashtag_help' ,'hashtag_useful' ,'hashtag_confused', 'hashtag_curious' ,'hashtag_interested' ,'hashtag_frustrated' ,'body.1'])
# print(twentyeighteen.columns.values)
#print(twentyeighteen.columns.values)

#items = np.concatenate([twentyeighteen.columns.values,['word_tag']])
#dataframe= dataframe[items]
#print(set(dataframe.columns.values).difference(set(twentyeighteen.columns.values)))

#dataframe = pd.read_csv('C:/Users/ktwic/Desktop/merged_3.csv')

def bool_to_int(inp): 
    return 0 if inp == 'FALSE' else 1

def label_to_int(inp):
    if not inp:
        print("no inp?")
    if inp == 'A1':
        return 2
    elif inp ==  'A2':
        return 1
    elif inp == 'C1':
        return 4
    elif inp == 'C2':
        return 3
    elif inp == 'I':
        return 5

def simplified_label(inp):
    if pd.isnull(inp):
        print("no input")
    if inp[0] == 'A':
        return -1
    elif inp[0] == 'C':
        return 0
    elif inp[0] == 'I':
        return 1
    else:
        return 99

def parent_label(inp):
    if inp == -1:
        return -1
    else:
        return 0

def parent_label_to_int(inp):
    if pd.isnull(inp) or not inp or inp == -1 or inp == '-1':
        return -1
    elif inp == 'A1':
        return 2
    elif inp ==  'A2':
        return 1
    elif inp == 'C1':
        return 3
    elif inp == 'C2':
        return 4
    elif inp == 'I':
        return 5

def get_color(inp):
    if inp>=1 and inp <4:
        inp =inp - 0.6
    hue = (inp*24)
    return hue
dataframe["is_comment"] = dataframe.is_comment.apply(lambda x : bool_to_int(x))
dataframe["is_question"] = dataframe.is_question.apply(lambda x : bool_to_int(x))
dataframe["is_elaboration"] = dataframe.is_elaboration.apply(lambda x : bool_to_int(x))
dataframe["is_comparative"] = dataframe.is_comparative.apply(lambda x : bool_to_int(x))
dataframe["label"] = dataframe.word_tag.apply(lambda x : simplified_label(x))
# twentyeighteen["is_comment"]=dataframe.is_comment.apply(lambda x : bool_to_int(x))
# twentyeighteen["is_question"]=dataframe.is_question.apply(lambda x : bool_to_int(x))
# twentyeighteen["is_elaboration"]=dataframe.is_elaboration.apply(lambda x : bool_to_int(x))
# twentyeighteen["is_comparative"]=dataframe.is_comparative.apply(lambda x : bool_to_int(x))
#extras["label"] = extras.word_tag.apply(lambda x : simplified_label(x))
#extras["parent_label"] = extras.parent_id.apply(lambda x: parent_label_to_int(x))
#extras["predicted_label"] =0
#extras["is_comment"] = extras.is_comment.apply(lambda x : bool_to_int(x))
dataframe["predicted_label"] = 0
# dataframe["parent_label"] = dataframe.parent_label.apply(lambda x: parent_label_to_int(x))
dataframe.fillna(0)
#extras.fillna(0)

k_fold = KFold(n_splits=10)


rand_for = RandomForestClassifier(n_estimators=300, n_jobs=-1)
#lasso = Lasso(alpha = 0.25)
#svr = svm.SVR(C=0.7, epsilon = 0.3, kernel = "rbf")
#gbrt = GradientBoostingRegressor(n_estimators = 300, max_depth =3)

x = dataframe.drop(columns = ["label","TRUE", "paragraph_entropy","paragraph_heat","word_tag","predicted_label", 'comment_id','text', 'parent_id', 'author_id', 'location_id', 'marked_par', 'ctime',  'marked_text'])
print(x.columns.shape)
#extras_x = extras.drop(columns = ["comment_id", "word_tag","parent_id","source_id","location_id", "text","marked_text", "marked_par","ctime","label"])
#print(extras_x.columns.values)

#extras_y = extras["label"]
#x = dataframe[["WC", "is_comment", "parent_label","replies_count", "para_sim", "text_sim", "fast_text", "fast_par", "num_sents"]]

paras = dataframe["marked_par"].unique()
#comment_corr = pd.DataFrame({"paragraph":paras, "num_comments": np.zeros([paras.shape[0]]),"truth_diff":np.zeros([paras.shape[0]])})

y = dataframe["label"]
#y = dataframe["true_heat"]
tau_avg = 0
acc_avg = 0
kappa_avg = 0
score =0
p_color = []
t_color = []
correct, count = 0,0
difference = []
g_t_one =0
confusion = np.zeros([5,5])
x = x.fillna(0)
#extras_x = extras_x.fillna(0)
y = y.fillna(0)
#extras_y = extras_y.fillna(0)
for train, test in k_fold.split(x):
    train_x, test_x = x.iloc[train], x.iloc[test]
    train_y, test_y = y.iloc[train] , y.iloc[test]
    #rand_for.fit(pd.concat([train_x, extras_x]), pd.concat([train_y, extras_y]))
    rand_for.fit(train_x, train_y)
    #svr.fit(train_x, train_y)
    predictions = rand_for.predict(test_x)
    # i=0
    # for row in test:
    #     dataframe["predicted_label"].iat[row] = predictions[i]
    #     i += 1
    kappa_avg += cohen_kappa_score(test_y, predictions, weights="linear")
    tau_avg += stats.kendalltau(test_y, predictions)[0]
    acc_avg += accuracy_score(test_y, predictions)
    #score += rand_for.score(test_x, test_y)
    # cm = confusion_matrix(test_y, predictions)
    # for i in range(5):
    #   for j in range(5):
    #         try:
    #             confusion[i][j] += cm[i][j]
    #         except:
    #             pass
    for item in range(len(paras)):
        i = paras[item]
        d_x = test_x[dataframe["marked_par"] == i]
        if len(d_x) > 0:
            d_y= test_y[d_x.index.values]
            p =rand_for.predict(d_x)
            #print(p)
            avg_predicted = p.mean()
            avg_truth = d_y.mean()
            #t_color = get_color(avg_truth)
            #p_color = get_color(avg_predicted)
            if len(p) > 0:
                difference.append(abs(avg_truth-avg_predicted))
            # comment_corr.at[item, 'num_comments'] += len(d_x)
            # comment_corr.at[item, 'truth_diff'] += abs(t_color-p_color)
print ("Features sorted by their score:", sorted(zip(map(lambda x: round(x, 4), rand_for.feature_importances_), x.columns), 
             reverse=True))                      
print("Kappa: ", kappa_avg/10)
print("Tau: ", tau_avg/10)
print("Accuracy: ",acc_avg/10)

# print("Heatmap Average Difference: ", sum(difference)/len(difference), "std. dv.",np.std(difference) )
# print(score/10)
# twentyeighteen["predicted_label"] = 0

# rand_for.fit(x,y)
# twentyeighteen = twentyeighteen.fillna(-1)
# predictions = rand_for.predict(twentyeighteen)
# print(predictions.shape)
# print(predictions)
# print(twentyeighteen.shape)
# i =0
# for row in twentyeighteen:
#     twentyeighteen["predicted_label"].iat[i] = predictions[i]
#     i+=1
# twentyeighteen["predicted_label"] = predictions
# twentyeighteen.to_csv("parlol.csv")
#comment_corr.to_csv("comments_vs_correct.csv")
# fig = plt.figure()
# plt.clf()
# ax = fig.add_subplot(111)
# ax.set_aspect(1)
# plt.xticks(range(5), ['A2','A1','C1','C2','I'])
# plt.yticks(range(5), ['A2','A1','C1','C2', 'I'])
# # plt.xticks(range(3), ['A','C','I'])
# # plt.yticks(range(3), ['A','C','I'])
# res = ax.imshow(np.array(confusion), cmap=plt.cm.Blues, 
#                 interpolation='nearest')
# thresh = confusion.max() / 2
# for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
#   plt.text(j, i, "{:,}".format(confusion[i, j]),
#                        horizontalalignment="center",
#                        color="white" if confusion[i, j] > thresh else "black")
# plt.show()
#dataframe.to_csv("with_predictions.csv")
#extras.to_csv("merged_2.csv")