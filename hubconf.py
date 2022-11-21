###### PART 2 ######
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

def build_lr_model(X=None, y=None):
  pass
  # lr_model = None
  # write your code...
  # Build logistic regression, refer to sklearn
  lr_model = LogisticRegression(random_state=0).fit(X, y)
  return lr_model

def build_rf_model(X=None, y=None):
  pass
  rf_model = None
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  rf_model = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y)
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  pass
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  acc, prec, rec, f1, auc = 0,0,0,0,0
  # write your code here...
  y_pred = model1.predict(X)
  acc = accuracy_score(y, y_pred)
  prec = precision_score(y, y_pred, average='weighted')
  rec = recall_score(y, y_pred, average='weighted')
  f1 = f1_score(y, y_pred, average='weighted')
  fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=2)
  auc = metrics.auc(fpr, tpr)
  # auc = 0
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  # lr_param_grid = None
  lr_param_grid = {'C':[1, 10, 100,1000], 'penalty': ('l1', 'l2')}
  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  # rf_param_grid = None
  rf_param_grid = {'C':[1, 10, 100, 1000], 'n_estimators': [1, 10, 100], 'criterion': ('gini', 'entropy'), 'maximum depth': [1, 10, None]}
  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  
  grid_search_cv = GridSearchCV(model1, param_grid, cv=cv, scoring = metrics, refit = False)
  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...

  grid_search_cv.fit(X, y)
  
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  cv_results = cross_validate(model1, X, y, cv=cv)
  top1_scores = []
  # for i in metrics:
  #   top1_scores[i] = cv_results[i]

  top_scores = cv_results['test_score']
  
  
  return top1_scores
