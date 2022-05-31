# Disaster Response Pipeline Project

### Introduction
In this project, a web app where an emergency worker can input a new message and get classification results in several categories was developed. In detail, a machine learning model was triained with real data set containing real messages that were sent during disaster events by using Multi-output classification.

### Project structure 
                app   
                | - template 
                | |- master.html # main page of web app
                | |- go.html # classification result page of web app
                |- run.py # Flask file that runs app
                data 
                |- disaster_categories.csv # data to process
                |- disaster_messages.csv # data to process
                |- process_data.py
                |- InsertDatabaseName.db # database to save clean data to
                models
                |- train_classifier.py
                |- classifier.pkl # saved model
                README.md

    
### Training data
Figure 8

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/YourDatabasename.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/YourDatabaseName.db models/classifier.pkl

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
