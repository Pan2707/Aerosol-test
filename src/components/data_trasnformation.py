import os
import sys
from dataclasses import dataclass
#The dataclass decorator is used to create classes that primarily store data, and it automatically generates special methods like __init__, __repr__, and others based on the class attributes.
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer  # basically used to create a pipeline , for eg first one hot encoding, then second standard scaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException #allows you to raise and handle custom exceptions with specific messages or error codes. It's a way to provide more context when something goes wrong in your code
from src.logger import logging   #Logging is used to record messages that provide insights into what your code is doing, which can be extremely helpful for debugging and monitoring.


from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl") #giving input to data transformation component

class DataTransformation:
    def __init__(self):  # it will create all the pickle files responsible for converting the categorical features into numerical and performing stanadrd scaler
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self): # this is all to create pickle files responsible for converting the categorical features into numerical and performing standard scaler
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
              #this numerical pipeline does two tasks: handling missing values and performing standard scaler (numerical into standard scale)
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),  #resposible for handling missing values  , strategy="median is for outlier as seen in EDA
                ("scaler",StandardScaler())   #This step is used for feature scaling. It standardizes your data by subtracting the mean and dividing by the standard deviation for each featur

                ]
            )
           #this categorical pipeline does three tasks: handling missing values and performing one hot encoding and standard scaler
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), 
                ("one_hot_encoder",OneHotEncoder()), #It converts categorical variables into a binary (0/1) format
                ("scaler",StandardScaler(with_mean=False)) 
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()  # preprocessing_obj are obj created above in preprocessing class, num&cat piepline, this  needs to be converted into a pickle file

            target_column_name="math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                          # allowing you to save pickle file to harddisk
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)