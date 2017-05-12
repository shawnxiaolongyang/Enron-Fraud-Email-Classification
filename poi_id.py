
# coding: utf-8

# In[1]:

get_ipython().system(u'jupyter nbconvert --to script poi_id.ipynb')


# In[1]:

import sys
import pickle
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from time import time
from sklearn.model_selection import GridSearchCV
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# ### Task 1: Select what features you'll use.
# 

# In[2]:

### features_list is a list of strings, each of which is a feature name.
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[3]:

Name = [x for x in data_dict]


# In[4]:

for i in Name:
    print i


# In[5]:

feature = [x for x in data_dict[Name[0]]]


# In[6]:

feature


# In[7]:

feature = [x for x in feature if x!= 'poi' and x!='email_address'and x!='from_this_person_to_poi' and x!='from_poi_to_this_person'and x!='shared_receipt_with_poi'and x!='total_stock_value' and x!= 'total_payments']
features_list = ['poi']+feature


# In[8]:

print "Numbers of features are",len(features_list)


# In[9]:

print "Numbers of data points are",len(data_dict)


# In[10]:

print "Percentage of POI in the dataset", float(len([x for x in data_dict if data_dict[x]['poi']==1 ]))/len(Name)


# In[11]:

d = {}
for i in feature:
    d[i] = 0
    for ii in Name:
        if data_dict[ii][i] == 'NaN':
            d[i] += 1


# In[12]:

for i in d:
    print "Feature",i,"contains",float(d[i])/len(Name), "Percent Missing values"


# In[13]:

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[14]:

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=42)


# In[15]:

def generate_feature_list():
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    importance = {}
    if accuracy_score(pred, labels_test) > 0.8:
        sort_features = sorted(range(len(clf.feature_importances_)), key=lambda i: clf.feature_importances_[i])
        sort_features = [x for x in sort_features if clf.feature_importances_[x]>0.1]
        for x in sort_features:
            importance[x] = clf.feature_importances_[x]
        return sort_features,importance
    else:
        return [],[]


# In[16]:

def vari_class_test(clf):
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    test_classifier(clf, my_dataset, features_list_2)


# In[17]:

feature_list = []
importance_dict = {}

for i in xrange(10000):
    sort_features,importance = generate_feature_list()
    feature_list.append(sort_features)
    for ii in importance:
        try:
            if importance_dict[ii] < importance[ii]:
                importance_dict[ii] = importance[ii]
        except:
            importance_dict[ii] = importance[ii]
    
feature_list = set.union(*[set(list) for list in feature_list])


# In[18]:

feature_list_2 = {}
for idx in feature_list:
    feature_list_2[importance_dict[idx]] = feature[idx]
    print "The feature",feature[idx],"has max importance",importance_dict[idx]


# #### let's try how many feature used would optimize the performace

# In[19]:

f_list_2 = []
for i in feature_list_2:
    f_list_2.append(feature_list_2[i])
for i in xrange(len(f_list_2)):
    clf = tree.DecisionTreeClassifier()
    features_list_2 = ['poi']+f_list_2[i:]
    print "use",len(f_list_2)-i,"features"
    vari_class_test(clf)


# We can find use 6 features would get better result
# Accuracy: 0.81360	Precision: 0.31297	Recall: 0.33300	F1: 0.32267	F2: 0.32879

# In[20]:

features_list_2 = ['poi']+f_list_2


# In[21]:

features_list_2


# ### Task 2: Remove outliers

# In[21]:

### Find outliers from plot
def plot3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for point in data:
        salary = point[1]
        exercised_stock_options = point[3]
        bonus = point[4]
        ax.scatter(salary, exercised_stock_options,bonus)
    ax.set_xlabel('salary Label')
    ax.set_ylabel('exercised_stock_options Label')
    ax.set_zlabel('bonus Label')
    plt.show()


# In[22]:

plot3d(data)


# In[23]:

for i in data_dict:
    if data_dict[i]['salary'] > 2e+07 and data_dict[i]['salary']!="NaN":
        print i


# In[24]:

### plot after pop out Total
data_dict.pop('TOTAL')


# We would also exclude those has almost all features 'NaN' point

# In[25]:

d = {}
for i in data_dict:
    d[i] = 0
    for ii in feature:
        if data_dict[i][ii] == 'NaN':
            d[i] += 1
miss_value =  [x for x in d if d[x] > 10]
print miss_value


# In[26]:

for i in miss_value:
    data_dict.pop(i)


# If we want to analyze based on features list, we should not include one has all these as "NaN"

# In[27]:

pop_list = []
for i in data_dict:
    count = 0
    for feature in features_list_2:
        if data_dict[i][feature]== "NaN":
            count += 1
    if count > len(features_list_2)-2:
        pop_list.append(i)
print pop_list


# In[28]:

### plot after pop out Total
for x in pop_list:
    data_dict.pop(x)


# In[29]:

data = featureFormat(data_dict, features_list)
plot3d(data)


# ### Task 3: Create new feature(s) using TF-IDF

# In[30]:

import os
import pickle
import re
import sys
from nltk.stem.snowball import SnowballStemmer 
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


# In[31]:

a = os.listdir("./emails_by_address")
print len(a)


# #### get the email from POI 

# In[32]:

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
Name = [x for x in data_dict]
feature = [x for x in data_dict[Name[0]]]
first_name = []
last_name = []
for i in Name:
    if data_dict[i]['poi'] == True:
        for idx,ii in enumerate(i.split(' ')):
            if idx == 0 :
                last_name.append(ii)
            else:
                if len(ii) >= 3:
                    first_name.append(ii)
print "POI's first name and last name:"


# In[33]:

d = {}
remove_list = {}
for i in a:
    stats = i.split('_')[0]
    name = i.split('_')[1].split('@')[0].strip('0123456789').replace('enron','').upper()
    f_name = ''
    l_name = ''

    for idx, ii in enumerate(name.split('.')):
        if idx == 0:
            f_name = ii       
        else:
            if len(ii) >= 3:
                l_name = ii
        if f_name != '' or l_name != '':
            remove_list[f_name] = 1
            remove_list[l_name] = 1
        if f_name != '' and l_name != '':
            
            for iii in xrange(len(first_name)):
                if f_name in first_name[iii] and l_name in last_name[iii]:
                    d[(first_name[iii],last_name[iii],stats)] = i
print "POI's first, last name and email txt:"
for i in d:
    print i,d[i]
from_POI = []
to_POI = []
for i in d:
    if "from" in i:
        from_POI.append(d[i])
    if "to" in i:
        to_POI.append(d[i])


# #### get the email from non-POI

# In[34]:

first_name = []
last_name = []

for i in Name:
    if data_dict[i]['poi'] == False:
        for idx,ii in enumerate(i.split(' ')):
            if idx == 0 :
                last_name.append(ii)
            else:
                if len(ii) >= 3:
                    first_name.append(ii)
                else:
                    first_name.append('')
                

fir_name = []
las_name = []

for i in xrange(len(last_name)):
    if first_name[i]!="" and last_name[i]!="":
        fir_name.append(first_name[i])
        las_name.append(last_name[i])

d = {}

for i in a:
    stats = i.split('_')[0]
    name = i.split('_')[1].split('@')[0].strip('0123456789').replace('enron','').upper()
    f_name = ''
    l_name = ''

    for idx, ii in enumerate(name.split('.')):
        if idx == 0:
            f_name = ii        
        else:
            if len(ii) >= 3:
                l_name = ii
                
                for iii in xrange(len(fir_name)):
                    if las_name[iii].startswith(l_name) and fir_name[iii].startswith(f_name):
                        d[(fir_name[iii],las_name[iii],stats)] = i
print "None-POI's first, last name and email txt:"
for i in d:
    print i,d[i]
No_from_POI = []
No_to_POI = []
for i in d:
    if "from" in i:
        No_from_POI.append(d[i])
    if "to" in i:
        No_to_POI.append(d[i])


# #### get the corpus

# In[35]:

def parseOutText(f):
    stemmm = SnowballStemmer("english")
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        for i in text_string.split():
                i_stem = stemmm.stem(i)
                words += ' ' + stemmm.stem(i)

    return words


# In[36]:

from_data = []
word_data = []
a = {}
for i in from_POI:
    a[i] = "POI"
for i in No_from_POI:
    a[i] = "NoPOI"
    
for i in a:
    from_POI_address = open("./emails_by_address/"+i,"r")
    for path in from_POI_address:
        try:
            path = path.replace("enron_mail_20110402/","")
            path = os.path.join(os.pardir, path[:-1])
            ### n represent how many levels to climb
            email = open(path, "r")
            ### use parseOutText to extract the text from the opened email
            text = parseOutText(email)
            ### use str.replace() to remove any instances of the words
            for item in remove_list:
                 text = text.replace(item, "")
            ### append the text to word_data
            word_data.append(text)
            ### append a 1 to from_data if email is from POI, and 0 if email is not from POI
            if a[i] == "POI":
                from_data.append(0)
            if a[i] == "NoPOI":
                from_data.append(1)
            
        except:
            skip = True
        email.close()
    print i,"process finished" 
    from_POI_address.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w"))


# In[37]:

tfidf_vectorizer1 = TfidfVectorizer(stop_words='english')
tfidf1 = tfidf_vectorizer1.fit_transform(word_data)
feature_names = tfidf_vectorizer1.get_feature_names()
print len(feature_names)


# In[38]:

labels = from_data
features = tfidf1
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.4, random_state=42)


# In[39]:

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
sort_features = sorted(range(len(clf.feature_importances_)), key=lambda i: clf.feature_importances_[i])
sort_features = [x for x in sort_features if clf.feature_importances_[x]>0.01]
for x in sort_features:
    print "important key words", feature_names[x], "has importance", clf.feature_importances_[x]


# In[40]:

true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for prediction, truth in zip(pred, labels_test):
    if prediction == 0 and truth == 0:
        true_negatives += 1
    elif prediction == 0 and truth == 1:
        false_negatives += 1
    elif prediction == 1 and truth == 0:
        false_positives += 1
    elif prediction == 1 and truth == 1:
        true_positives += 1
total_predictions = true_negatives + false_negatives + false_positives + true_positives
accuracy = 1.0*(true_positives + true_negatives)/total_predictions
precision = 1.0*true_positives/(true_positives+false_positives)
recall = 1.0*true_positives/(true_positives+false_negatives)        
print "accuracy",accuracy
print "precision",precision
print "recall",recall


# Although the data looks great, the tfidf model conclude only 14 POI and 2 Non-poi (since we did not have others actual email address based on their name), the prediction is focus on the exist data.
# 
# In this case, I won't use this data in further analysis

# ### Task 4: Try a varity of classifiers

# In[41]:

def vari_class_test(clf):
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    test_classifier(clf, my_dataset, features_list_2)


# ### GaussianNB

# In[42]:

clf = GaussianNB()
vari_class_test(clf)


# ### Adaboost

# In[43]:

clf =  AdaBoostClassifier()
vari_class_test(clf)


# ### K_Nearest_Neighbour

# In[44]:


clf = KNeighborsClassifier(2)
vari_class_test(clf)


# ### Random_Forest

# In[45]:


clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
vari_class_test(clf)


# #### Now wew add PCA to each of them to find better results:

# ### PCA + Adaboost

# In[46]:

pipe = Pipeline(steps=[('pca', PCA()), ('classify', AdaBoostClassifier())])
vari_class_test(pipe)


# ### Task 5: Tune your classifier to achieve better than .3 precision and recall 

# ### PCA + K_Nearest_Neighbour

# In[47]:

pipe = Pipeline(steps=[('pca', PCA()), ('classify', KNeighborsClassifier(2))])
vari_class_test(pipe)


# ### PCA + Random_Forest

# In[48]:

pipe = Pipeline(steps=[('pca', PCA()), ('classify', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))])
vari_class_test(pipe)


# #### After adding PCA, we can find all model increase precision and recall
# #### Then we can test different parameters effect

# In[49]:

import optunity
import optunity.metrics


# Every learning algorithm has its own hyperparameters:
# 
# k-NN: 1<n_neighbors<5 the number of neighbours to use
# 
# naive Bayes: no hyperparameters
# 
# random forest:
# 10<n_estimators<30: number of trees in the forest
# 5<max_features<20: number of features to consider for each split

# In[50]:

search = {'algorithm': {'k-nn': {'n_neighbors': [1, 5]},
                        'naive-bayes': None,
                        'random-forest': {'n_estimators': [10, 30],
                                          'max_features': [1, 5]}
                        }
         }


# In[51]:

def performance(algorithm, n_neighbors=None, n_estimators=None, max_features=None):
    # split the data
    folds = 1000
    data = featureFormat(my_dataset, features_list_2, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        # fit the model
        if algorithm == 'k-nn':
            model = Pipeline(steps=[('pca', PCA()), ('classify', KNeighborsClassifier(n_neighbors=int(n_neighbors)))])       
            model.fit(features_train, labels_train)

        elif algorithm == 'naive-bayes':
            model = Pipeline(steps=[('pca', PCA()), ('classify', GaussianNB())])
            model.fit(features_train, labels_train)
        elif algorithm == 'random-forest':
            model = Pipeline(steps=[('pca', PCA()), ('classify', RandomForestClassifier(n_estimators=int(n_estimators),
                                           max_features=int(max_features)))])
            model.fit(features_train, labels_train)
        else:
            raise ArgumentError('Unknown algorithm: %s' % algorithm)

        # predict the test set
        predictions = model.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        return (accuracy*precision*recall)
    except:
        print "Got a divide by zero when trying out:", clf


# In[52]:

optimal_configuration, info, _ = optunity.maximize_structured(performance,
                                                              search_space=search,
                                                              num_evals=300)
solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
print('Solution\n========')
print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))


# #### We can find the best result is KNN with N = 2

# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# 

# In[53]:

pipe = Pipeline(steps=[('pca', PCA()), ('classify', KNeighborsClassifier(2))])
vari_class_test(pipe)
dump_classifier_and_data(pipe, my_dataset, features_list_2)





