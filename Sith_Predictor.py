import requests
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup #to read the web page
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score,f1_score,roc_auc_score


#Import the model and the GridSearch
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")

df = [] 
def collumn_filler(): #Collum filler will fill the different columns with the data it finds as h3 and divs
    for line in class_content_divs:
        
        #find the names of characteristics for the character
        val_a = soup.find_all('h3', class_='pi-data-label pi-secondary-font')
        for val in val_a:
            characteristic = val.text if val else None
            col_char.append(characteristic)
         
        #find the value of the characteristics the character has
        val_b = soup.find_all('div',class_='pi-data-value pi-font')
        for val in val_b:
            variable_1 = val.text if val else None
            variable_2 = val.a.text if val.a else ''
            variable = variable_1 if variable_1 else variable_2
            col_var.append(variable)

# if statement for each card within all the divs
if __name__ == "__main__":
    r = requests.get("https://starwars.fandom.com/wiki/Databank_(original)#app_characters") #this is the main source of characters from starwars
    html = r.text
    soup = BeautifulSoup(html)


    class_content_divs = soup.find('table', class_='appearances') 
    links_anchors = class_content_divs.find_all('a') 
    
    link_list = [] #the links of the characters which will be looped
    for link_anchor in links_anchors:
        link = link_anchor.get('href')
        link_list.append(link)
    
    dfs = []
    var_counter = 1
    for link in link_list:    
        
        r = requests.get("https://starwars.fandom.com{}".format(link)) #this loop will request each complete link for the characters
        html = r.text
        soup = BeautifulSoup(html)
        
        class_content_divs = soup.find_all('div',class_='mw-parser-output')
        
        
        col_var = []
        col_char = []
    
        collumn_filler() 
        
        df_tojoin = pd.DataFrame([col_var, col_char]).T
        df_tojoin.columns = ('Character_{}'.format(var_counter),'Characteristics')
        var_counter = var_counter+1
        df_tojoin.set_index('Characteristics', inplace=True)
        
        dfs.append(df_tojoin) #get all of the lists in one place
    base = pd.DataFrame(dfs[1]) #set the base for the DF

dfs = dfs[2:] #get rid of the first columns
df = base.join(dfs, how='outer') #join all the lists to the  base DF
df = df.T #transpose for a better view

for column in df: #
    columns_without_numbers = ['Affiliation(s)','Apprentices','Caste','Clan(s)', 'Cybernetics','Designation','Distinctions','Domain','Duties','Established by','Eye color','Government','Hair color','Homeworld','Kajidic','Language','Masters','Organization','Skin color','Species']
    for column in columns_without_numbers:
        i=0
        while i < len(columns_without_numbers):
            df[column] = df[column].replace('{}'.format(i),'', regex=True) #remove the numbers from the columns in the DF
            i=i+1

for column in df:
    col = []
    if 'date'in column.lower():
        df.drop(column, inplace=True, axis=1)  #drop the columns which have dates, we don't need them 

cols = df.columns

cols_with_numbers = [] #a list of columns which dont have numbers
for col in cols:
    if col not in columns_without_numbers:
        cols_with_numbers.append(col)

#this loop will clean each row for each number column leaving only the numbers or None        
for column in cols_with_numbers: 
    new_column = []
    for value in df[column]:
        value = str(value)
        values = re.findall(r'\d+', value)
        target_val = values[0:2] if values else None
        res = []
        if target_val != None:
            if len(target_val) > 1:
                res.append(target_val[0]+'.'+target_val[1]) #if the value is currently 2 different values it will capture both
            elif len(target_val) == 1:
                 res.append(target_val[0]) #if the value is a single value, it will just copy that value
        else:
            res.append(None)
        new_column.append(res)
    df[column]=new_column

#This loop will remove the initial brackets and make sure the words are not joint and instead use a coma
for column in columns_without_numbers:
    col_clean = []
    for value in df[column]:
        value =str(value).replace('[','').replace(']',',')
        value = value.split(',') #the comas will separate the different values
        col_clean.append(value)
        
    df[column]=col_clean

#this loop will create dummies on different variables
for column in columns_without_numbers:
    affs = []
    for row in df[column]:
        for aff in row:
            if aff not in affs:
                affs.append(aff)

    for aff in affs:
        variable_count = []
        for val in df[column]:
            if aff in val:
                variable_count.append(1) #input in the column a 1 if true
            else:
                variable_count.append(None) #input in the column a None if false

        if (variable_count.count(1))>10: #this decides that if the ammount of defaults is more than 10 it'll add value to the decision, change different parameters
            df[aff]=variable_count
        }

df.drop(columns_without_numbers, axis=1, inplace=True) #drop the columns which have been set to dummies
df.drop(['nan','','None','Gender','Born','Died','Term length'], axis=1, inplace=True) #drop these other not usefull columns
df = df.iloc[:,4:] #drop more useless columns

height_and_mass = df.iloc[:,:2]

#this loop will change the geights and masses of the characters to numbers only
for column in height_and_mass:
    new_column=[]
    for bracket in df[column]:
        for character in bracket:
            new_column.append(float(character) if character else None)
    df[column]=new_column
    df[column].replace(np.nan, df[column].mean(), inplace=True)

df.replace(np.nan, 0, inplace=True) #replace all the nan with a 0 for the models

x = df.drop('Galactic Empire', axis=1)
y = df['Galactic Empire'] #change this if you want to predict their alligeance to another alliance

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30 , random_state=1)

xgb_tuned = XGBClassifier(random_state=1, eval_metric='logloss')

# Grid of parameters to choose from
parameters = {
    "n_estimators": [10,30,50],
    "scale_pos_weight":[1,2,5],
    "subsample":[0.7,0.9,1],
    "learning_rate":[0.05, 0.1,0.2],
    "colsample_bytree":[0.7,0.9,1],
    "colsample_bylevel":[0.5,0.7,1]
    }

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
rand_obj = RandomizedSearchCV(xgb_tuned, parameters,scoring=scorer, cv=5)
rand_obj = rand_obj.fit(x_train, y_train)

xgb_tuned = rand_obj.best_estimator_

# Fit the best algorithm to the data.
xgb_tuned.fit(x_train, y_train)


#The following commands will help set up the display of importances.
feature_names = x_train.columns
importances = xgb_tuned.feature_importances_
indices = np.argsort(importances)

#plot to see the importances per characteristic
plt.figure(figsize=(10,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

#The following formula will create the confusion matrix
def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    
    #Print the scores above the matrix
    print(metrics.recall_score(target, y_pred))
    print(metrics.precision_score(target, y_pred))
    print(metrics.accuracy_score(target, y_pred))
    
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    #Plotting the matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")