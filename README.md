DISASTER RESPONSE PIPELINE PROJECT
------------------------------------
The three major aspects of this project is as follows:

* ETL Pipeline
* ML Pipeline
* Flask web-app displaying analysis from data

Whilst building the app, the RandomForestClassifier produced results:
* **Average precision:** 0.9161571841990476
* **Average recall:** 0.9312031222395322
* **Average f_score:** 0.9105049019931754

Whilst building the app, the AdaBoostClassifier produced results:
* **Average recall:** 0.9282398856463709
* **Average f_score:** 0.9096621267659827

Related Blog
----------------------
https://medium.com/@deogakofiofuafor/airbnb-seattle-for-better-understanding-21b1132ee69f

File Structure
----------------------
* `../data/run.py` is the executable for the app

* `requirements.txt`: contains the environment requirements to run program
* `app` folder contains the following:
  * `static`: Folder containing all image files
  * `templates`: Folder containing
    * `index.html`: Renders homepage
    * `go.html`: Renders the message classifier
  * `run.py`: Defines the app routes

* `data` folder contains the following:
    *  `disaster_categories.csv`: contains the disaster categories csv file
    * `disaster_messages.csv`: contains the disaster messages csv file
    * `emergency.db`: contains the emergency db which is a merge of categories and messages by ID
    * `etl_pipeline.py`: contains the scripts to transform data
    * `figures.db`: contains the figures db to plot graphs
    * `proccess_data.py`: contains the scripts to run etl pipeline for cleaning data

* `models` folder contains the following:
    * `ml_pipeline.py`: contains scripts that create ml pipeline
    * `model_ada_fit.pickle`: contains the AdaBoostClassifier pickle fit file
    * `model_ada.pickle`: contains the AdaBoostClassifier pickle file
    * `model_rf_fit.pickle`: contains the RandomForestClassifier fit pickle file
    * `model_rf.pickle`: contains the RandomForestClassifier pickle file
    * `results_ada.pickle`: contains the AdaBoostClassifier prediction results pickle file
    * `train_ml_pipeline.py`: script to train ml_ipeline.py


* It is recommended you run the solution in a virtual environment. Please see https://docs.python.org/3/library/venv.html


INSTALLATION
----------------------
### Clone Repo

* Clone this repo to your computer.
* For mac please ensure you have xcode or download it from the app store (probably not needed)
* From your CLI install homebrew using `/usr/bin/ruby -e "$(curl -fsSL https:/raw.githubusercontent.com/Homebrew/install/master/install)"`
* After installing homebrew successfully, install python3 using `brew install python3`
* Check python3 installed correctly using `python3 --version` and this should return python3 version
* Install the requirements using `pip3 install -r requirements.txt`.
    * Make sure you use Python 3
* `cd` to the location "../app/"
* Execute `python3 run.py`
* Follow the information printed in your environment to the site. Usually 0.0.0.0:3001 or localhost:3001


### Rerun Scripts
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
      -  `python3 data/process_data.py`
      -  `python3 data/etl_pipeline.py`
      -  `python3 data/figures.py`


    - To run ML pipeline that trains classifier and saves
      - `python3 model/ml_pipeline.py`
      - `python3 model/train_ml_pipeline.py`
2. Run the following command in the app's directory to run your web app.
    `python3 app/run.py`

3. Go to http://0.0.0.0:3001/



Extending this
-------------------------

If you want to extend this work, here are a few places to start:

* Rewriting code with dask so gridsearch debugging and compiling can be quicker
* Include more graphs
* Explore the effect of stopwords on precision
* Use dummies data for related field instead of replacing 2 with 1



## Credits

Lead Developer - Deoga Kofi


## License

The MIT License (MIT)

Copyright (c) 2020 Deoga Kofi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
