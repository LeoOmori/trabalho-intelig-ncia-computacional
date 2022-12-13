from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# Import training data
data = pd.read_csv('lbp_features2.csv')
print('lbp_features2.csv')
## combine data with newData with pandas concat
y = data.iloc[:, -1]


# train split with train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y,test_size=0.4, random_state=1, stratify=y)

## standardize data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


## remove label from last column


# Create the three classifiers
#euclidean distance
knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
svm_clf = SVC(kernel='linear', gamma='auto')
rf_clf = RandomForestClassifier()



#Combine the classifiers in a voting classifier
voting_clf = VotingClassifier(estimators=[('knn', knn_clf), ('svm', svm_clf), ('rf', rf_clf)],
                              voting='hard',)

 
# Evaluate the voting_clf
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))

## make a confusion matrix in mat plot lib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True)
# ## save image as confusion_matrix.jpg
# plt.savefig('confusion_matrix.jpg')





# # Save the trained model to a file
# with open('voting_clf_lbp.pkl', 'wb') as f:
#   pickle.dump(voting_clf, f)


