import numpy as np
import pandas as pd

# model 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# metrics
from sklearn.metrics import mean_squared_error

## validation data 간단하게 돌려볼 moelling 코드

class Val_Modeling() :
    
    def __init__(self, train_X, train_y, test_X, test_y) :
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        
    def lr(self) :  # linear regression
        model = LinearRegression()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
    
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
        y_pred = y_pred.clip(0, 20)
        
        rmse = np.sqrt(mean_squared_error(self.test_y, y_pred)) # rmse = sqrt(mse)
        
        return y_pred, rmse
    
    def dtr(self) :  # decision tree regressor
        model = DecisionTreeRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
    
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
        y_pred = y_pred.clip(0, 20)
        
        rmse = np.sqrt(mean_squared_error(self.test_y, y_pred)) # rmse = sqrt(mse)
        
        return y_pred, rmse
        
    def rfr(self) :  # random forest regressor
        model = RandomForestRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
    
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
        y_pred = y_pred.clip(0, 20)
        
        rmse = np.sqrt(mean_squared_error(self.test_y, y_pred)) # rmse = sqrt(mse)
        
        return y_pred, rmse
    
    def abr(self) :  # adaboost regressor
        model = AdaBoostRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
    
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
        y_pred = y_pred.clip(0, 20)
        
        rmse = np.sqrt(mean_squared_error(self.test_y, y_pred)) # rmse = sqrt(mse)
        
        return y_pred, rmse
    
    def gbr(self) :  # gradient boosting regressor
        model = GradientBoostingRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
    
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
        y_pred = y_pred.clip(0, 20)
        
        rmse = np.sqrt(mean_squared_error(self.test_y, y_pred)) # rmse = sqrt(mse)
        
        return y_pred, rmse
    
    def xgbr(self) :  # xgboost regressor
        model = XGBRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
    
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
        y_pred = y_pred.clip(0, 20)
            
        rmse = np.sqrt(mean_squared_error(self.test_y, y_pred)) # rmse = sqrt(mse)
        
        return y_pred, rmse
    
    def lgbmr(self) :  # lightgbm regressor
        model = LGBMRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
    
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
        y_pred = y_pred.clip(0, 20)
            
        rmse = np.sqrt(mean_squared_error(self.test_y, y_pred)) # rmse = sqrt(mse)
        
        return y_pred, rmse
    
    def cbr(self) :  # catboost regressor
        model = CatBoostRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
    
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
        y_pred = y_pred.clip(0, 20)
            
        rmse = np.sqrt(mean_squared_error(self.test_y, y_pred)) # rmse = sqrt(mse)
        
        return y_pred, rmse
    

## test 데이터로 돌려볼 모델

class Modeling() :
    
    def __init__(self, train_X, train_y, test_X) :
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        
    def lr(self) :  # linear regression
        model = LinearRegression()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
            
        return y_pred
    
    def dtr(self) :  # decision tree regressor
        model = DecisionTreeRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
            
        return y_pred
        
    def rfr(self) :  # random forest regressor
        model = RandomForestRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
            
        return y_pred
    
    def abr(self) :  # adaboost regressor
        model = AdaBoostRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
            
        return y_pred
    
    def gbr(self) :  # gradient boosting regressor
        model = GradientBoostingRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
            
        return y_pred
    
    def xgbr(self) :  # xgboost regressor
        model = XGBRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
            
        return y_pred
    
    def lgbmr(self) :  # lightgbm regressor
        model = LGBMRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
    
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
            
        return y_pred
    
    def cbr(self) :  # catboost regressor
        model = CatBoostRegressor()  
        model.fit(self.train_X, self.train_y) # train 데이터를 fit해줌
        
        y_pred = model.predict(self.test_X) # test 데이터를 넣어서 y의 예측값을 y_pred에 저장
            
        return y_pred