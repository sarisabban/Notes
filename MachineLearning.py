import os , numpy , pandas , matplotlib.pyplot , matplotlib.colors					#pip3 install numpy pandas matplotlib scikit-learn
#--------------------------------------------------
''' Representation '''
from sklearn import model_selection
data = pandas.read_csv('fruits.csv')
#print(data)												#To see dataset table
#print(data.head())											#To see only the top of the table
#print(data.shape)											#Shows dimentions of the table (Rows , Columns)
X = data[['mass', 'width', 'height' , 'color_score']]									#Identify features
Y = data['fruit_label']											#Identify target lable(s)/value(s)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X , Y , random_state = 0)		#Split dataset into a train set (75%) and a test set (25%) [random_state: if use None then the split will change everytime the program is run. On the other hand if you use random_state=some_number, then you can guarantee that the output of Run 1 will be equal to the output of Run 2, i.e. your split will be always the same. It doesn't matter what the actual random_state number is 42, 0, 21, ... The important thing is that everytime you use 42, you will always get the same output the first time you make the split]
prediction = [[20 , 4.3 , 5.5]]
#--------------------------------------------------
''' Feature Engineering '''
#Polynomial:
#from sklearn import preprocessing
#preprocessing.PolynomialFeatures(degree = 2).fit_transform(X_train)
#preprocessing.PolynomialFeatures(degree = 2).fit_transform(X_test)

#MinMax Scaling:
#from sklearn import preprocessing
#preprocessing.MinMaxScaler().fit_transform(X_train)
#preprocessing.MinMaxScaler().fit_transform(X_test)
#--------------------------------------------------
''' Models '''
#1 - KNN:
#from sklearn import neighbors
#ML = neighbors.KNeighborsClassifier(n_neighbors = 5).fit(X_train , Y_train)
#ML = neighbors.KNeighborsRegressor(n_neighbors = 5).fit(X_train , Y_train)

#2 - Linear Regression:
#from sklearn import linear_model
#ML = linear_model.LinearRegression().fit(X_train , Y_train)

#3 - Ridge:
#from sklearn import linear_model
#ML = linear_model.Ridge(alpha = 2).fit(X_train , Y_train)

#4 - Lasso:
#from sklearn import linear_model
#ML = linear_model.Lasso(alpha = 2).fit(X_train , Y_train)

#5 - Logistic Regression:
#from sklearn import linear_model
#ML = linear_model.LogisticRegression(C = 10).fit(X_train , Y_train)

#6.1 - SVM:
#from sklearn import svm
#ML = svm.LinearSVC(C = 1, random_state = 0).fit(X_train, Y_train)

#6.2 - Kernalised SVM: 'rbf' , 'linear' , 'poly'
#from sklearn import svm
#ML = svm.SVC(kernel = 'rbf' , C = 1, gamma = 1 , random_state = 0).fit(X_train, Y_train)

#7 - Dicision Trees:
from sklearn import tree
ML = tree.DecisionTreeClassifier(max_depth = 3 , random_state=0).fit(X_train, Y_train)
#ML = tree.DecisionTreeRegressor(max_depth = 3 , random_state=0).fit(X_train, Y_train)


#--------------------------------------------------
''' Evaluate '''
#Default Score:
#print(ML.score(X_train , Y_train))
#print(ML.score(X_test , Y_test))

#Feature Importance:
#print(ML.feature_importances_)

#Class Imballance:
#from sklearn import dummy
#dummy_frequent = dummy.DummyClassifier(strategy = 'most_frequent').fit(X_train, Y_train)
#print(dummy_frequent.score(X_test , Y_test))
#dummy_stratified = dummy.DummyClassifier(strategy = 'stratified').fit(X_train, Y_train)
#print(dummy_stratified.score(X_test , Y_test))
#dummy_uniform = dummy.DummyClassifier(strategy = 'uniform').fit(X_train, Y_train)
#print(dummy_uniform.score(X_test , Y_test))
#dummy_mean = dummy.DummyRegressor(strategy = 'mean').fit(X_train, Y_train)
#print(dummy_mean.score(X_test , Y_test))
#dummy_median = dummy.DummyRegressor(strategy = 'median').fit(X_train, Y_train)
#print(dummy_median.score(X_test , Y_test))

#Confusion Matrix:
#from sklearn import metrics
#prediction = ML.predict(X_test)
#print(metrics.confusion_matrix(Y_test , prediction))
#print(metrics.classification_report(Y_test , prediction , target_names = X.columns.values))
#--------------------------------------------------
''' Optimisation '''
#Cross-Validation:
#from sklearn import model_selection
#score = model_selection.cross_val_score(ML , X , Y , cv = 3)
#print('Test set score:\t\t' , score)
#print('Test set mean score:\t' , numpy.mean(score))

#Validation Curve:
#param = 'max_depth'																		#Identify model parameter
#therange = [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10]														#Identify model parameter range
#score = model_selection.validation_curve(ML , X , Y , param_name = param , param_range = therange , cv = 5)							#Preform Validation Curve
#print('Train Score\n' , score[0])																	#Show plot
#print('Test Score\n' , score[1])

#Grid Search:

#--------------------------------------------------
''' Prediction '''
#print(ML.predict(prediction))
#--------------------------------------------------
''' Plots '''
#Features Plot:
#scatter = pandas.plotting.scatter_matrix(X_train, c = Y_train , hist_kwds = {'bins' : 15} , cmap = matplotlib.cm.get_cmap('gnuplot')).all()
#matplotlib.pyplot.show(scatter)




#KNN Plot Clasdification:



#KNN Plot Regression:



#Regression Plot:
#matplotlib.pyplot.scatter(data[['mass']] , data[['width']])
#matplotlib.pyplot.show()

#Classification/Cluster Plot:
#matplotlib.pyplot.scatter(data[['mass']] , data[['width']] , c = data[['fruit_label']] , cmap = matplotlib.colors.ListedColormap(['#FFFF00' , '#00FF00' , '#0000FF' , '#000000']))
#matplotlib.pyplot.show()

#Validation Curve:
#param = 'max_depth'																		#Identify model parameter
#therange = [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10]														#Identify model parameter range
#score = model_selection.validation_curve(ML , X , Y , param_name = param , param_range = therange , cv = 5)							#Preform Validation Curve
#Train_Score_mean = 		numpy.mean(score[0] , axis = 1)													#Calculate mean of Train Score values
#Test_Score_mean =		numpy.mean(score[1] , axis = 1)													#Calculate mean of Test Score values
#Train_Score_std = 		numpy.std(score[0] , axis = 1)													#Calculate standard deviation of Train Score values
#Test_Score_std	= 		numpy.std(score[1] , axis = 1)													#Calculate standard deviation of Train Score values
#matplotlib.pyplot.plot(	therange , Train_Score_mean , label = 'Train Score' , color = 'darkorange')							#Plot X , Y values for the mean of the Train Scores
#matplotlib.pyplot.plot(	therange , Test_Score_mean , label = 'Test Score' , color = 'navy')								#Plot X , Y values for the mean of the Test Scores
#matplotlib.pyplot.fill_between(therange , Train_Score_mean - Train_Score_std , Train_Score_mean + Train_Score_std , alpha = 0.2 , color = 'darkorange')	#Fill area of standard deviation for Train Score
#matplotlib.pyplot.fill_between(therange , Test_Score_mean - Test_Score_std , Test_Score_mean + Test_Score_std , alpha = 0.2 , color = 'navy')			#Fill area of standard deviation for Test Score
#matplotlib.pyplot.ylim(	0.0 , 1.0)															#Y axis range from 0 to 1
#matplotlib.pyplot.legend(	loc = 'best')															#Display legend
#matplotlib.pyplot.xlabel(	param)																#X-axis label
#matplotlib.pyplot.ylabel(	'Score')															#Y-axis label
#matplotlib.pyplot.show()

#Decision Tree:
#TheTree = tree.export_graphviz(ML , out_file = 'tree.dot')
#os.system('dot -Tpng tree.dot -o tree.png')			#sudo apt install graphviz
#os.remove('tree.dot')

#Feature Importance:
#importances = ML.feature_importances_
#indices = numpy.argsort(importances)[::-1]
#matplotlib.pyplot.title('Feature importances')
#matplotlib.pyplot.bar(range(X.shape[1]) , importances[indices])
#matplotlib.pyplot.xticks(range(X.shape[1]) , indices)
#matplotlib.pyplot.xlim([-1 , X.shape[1]])
#matplotlib.pyplot.show()
