import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
class EstimatorSelectionHelper:

    """
        Class to train models passed in a dict (models) with gridsearch testind parameters passed by another dict (params).
        
        The keys of the models dict should be the same as the keys of the dict params.
        
        methods:
            fit - train each model and store it
            score_summary - build a table with training information for each model
            get_best_model - return the best model according to best_score_ from gridsearchcv
            
       This class is an improvement from http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
    """
    def __init__(self, models=None,params=None,cv=3, n_jobs=1,scoring = "f1"):
        
        assert isinstance(models,dict), "Models should be a dict"
        assert isinstance(params,dict), "Params should be a dict"
        assert isinstance(cv,int), "cv should be an Integer"
        assert isinstance(n_jobs,int), "n_jobs should be an Integer"
        
        if not set(models.keys()).issubset(set(params.keys())): #check if all models have parameters on dict and raise error
            missing_params = list(set(models.keys()) - set(params.keys())) 
            raise ValueError("Some estimators are missing parameters: %s" % missing_params) 
            
        # TODO same for params in models:
        if not set(params.keys()).issubset(set(models.keys())): #check if all parameters have models on dict and raise error
            missing_models = list(set(params.keys()) - set(models.keys())) 
            raise ValueError("Some parameters are missing estimators: %s" % missing_models) 
            
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring

    def fit(self, X, y):
        """
            Fit all models using gridsearchcv.
            
            X - Training data without dependent variable
            y - Target or dependent variable
        
        """
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=self.cv, n_jobs=self.n_jobs,scoring=self.scoring)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        """
            Build a table ordered by sort_by with estimator name, score statistics and hyperparameters.
            
            sort_by - String with values min_score,max_score,mean_score
        """
        
        assert sort_by in ["min_score","max_score","mean_score"], "sort_by should be min_score,max_score or mean_score"
        
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
    
    def get_best_model(self):
        best_model = None
        best_score = 0
        for key in self.keys:
               if self.grid_searches[key].best_score_ > best_score:
                    best_score = self.grid_searches[key].best_score_
                    best_model = self.grid_searches[key].best_estimator_
        return best_model