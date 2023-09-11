##responsible for building ML appilcation as a package 

from setuptools import find_packages,setup # find_packages automatically find all the packages available in the entire ML application
from typing import List # funcion will retunr a list 

HYPEN_E_DOT='-e .' 
def get_requirements(file_path:str)->List[str]: # a function called get_requirements will install all the libraries mentioned in the requirements.txt file, file_path will be in str and it will return a list in stri
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:                                         # it will open the file and the file object will be returned
        requirements=file_obj.readlines()                                      # once we read the lines inside the requirements.text, iut will read one by one elementand a /n will get added
        requirements=[req.replace("\n","") for req in requirements]            # here /n will b replaced by a blank

        if HYPEN_E_DOT in requirements:                                         # this -e . should nt be available inn  requirements   
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

#data information about entire ML project
setup(
name='00_mlproject',
version='0.0.1',
author='Prashant',
author_email='pandeyprashant895@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)