import sklearn.datasets as skd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# import data (after moving cleaned files into a new directory. ex: Cleaned files from train\faculty goes into train\cleaned\faculty)
categories=["student","course","faculty"]
web_train = skd.load_files(r"C:\Users\Richa\Downloads\A1\dataset\train\cleaned",categories=categories, encoding="ISO-8859-1")
web_test = skd.load_files(r"C:\Users\Richa\Downloads\A1\dataset\test\cleaned",categories=categories, encoding="ISO-8859-1")

#converting text data into numerical representations using count vectorizer
count_vect = CountVectorizer()
X_train_tf = count_vect.fit_transform(web_train.data)
X_test_tf = count_vect.transform(web_test.data)

#training the model
tree = DecisionTreeClassifier().fit(X_train_tf, web_train.target)
bayes = MultinomialNB().fit(X_train_tf, web_train.target)

#setup for knn with grid search #
# setting a range for the number of neighbors to consider (this will be the k-value)
k_range = range(1, 31)

# Define the parameter grid to search
param_grid = dict(n_neighbors=k_range)

# Initialize the grid search
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy')

# Fit the grid search to the training data
grid.fit(X_train_tf, web_train.target)

# find and use the best performing k-value and model
best_k = grid.best_params_['n_neighbors']
best_model = grid.best_estimator_

predictedTree = tree.predict(X_test_tf)
predictedBayes = bayes.predict(X_test_tf)
predictedKNN = best_model.predict(X_test_tf)

#Performing the validation
print("Decision Tree Accuracy: ", accuracy_score(web_test.target,predictedTree))
print(metrics.classification_report(web_test.target, predictedTree, target_names=web_test.target_names))

print("Naive Bayes Accuracy: ", accuracy_score(web_test.target,predictedBayes))
print(metrics.classification_report(web_test.target, predictedBayes, target_names=web_test.target_names))

print("KNN Accuracy: ", accuracy_score(web_test.target,predictedKNN))
print(metrics.classification_report(web_test.target, predictedKNN, target_names=web_test.target_names))