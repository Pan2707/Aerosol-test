# The main aim is to read the dataset from some specific data source ( which can be from big data team or cloud team or live stream data)
#Our aim is to read the data and split it into training and test data

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass ##The dataclass decorator is used to create classes that primarily store data, 

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

#why data ingestion config called : these are the input whic we give to data ingestion component and now data igestion component shows where to train path and filepath
@dataclass  ##The dataclass decorator is used to create classes that primarily store data, 
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") # data ingestion will save train.csv in this particular path
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")   #readig the dataset in very easy way
        try:
            df=pd.read_csv('/Users/00_Projects/00_mlproject/notebook/data/data/StudentsPerformance.csv') # Read the dataset , here from C drive, but it cannbe from UI
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,   # this infor is required for data transformation. the data transformation will grab this two points and process it 
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation() #we are startting data transformation inside this function
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))      # this will print the r2 score 