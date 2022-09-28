# Disaster Response Pipeline Project

### Table of Contents

1. [Instructions](#instructions)
2. [Project Description](#description)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

### Instructions:<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


## Project Description<a name="description"></a>

This project showcases data engineering skills by analysing data from [Appen](https://appen.com/) (formally Figure 8) to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events which are categorised across 36 different categories and can be used to send the messages to an appropriate disaster relief agency.


## File Descriptions <a name="files"></a>
There are three directories included in this project:
1. app/ directory containing files related to the flask web app
2. data/ directory containig underlying dataset, python script for ETL Pipeline and a database with data ready for ML Pipeline
3. models/ directory containing python script for ML Pipeline and .pkl file with trained model

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Appen for the data. You can find more information about Appen, their mission and other descriptive information [here](https://appen.com/).

This project is inspired and submitted for the purposes of Udacity Data Science Nanodegree. You can find additional information about the course [here](https://udacity.com/course/data-scientist-nanodegree--nd025).