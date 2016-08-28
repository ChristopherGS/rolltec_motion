#RollTec Motion
=====================

This is an open source project for motion classification of grappling martial arts using wearable sensors. 
The repo contains a self-contained sandbox, with ready-to-run algorithm and complete training data.

This code was created as part of a prototype wearable for grappling martial artists. Unfortunately the project
crowdfunding campaign was unsuccessful: [RollTec Kickstarter](https://www.kickstarter.com/projects/1489178162/rolltec-grappling-the-first-wearable-made-for-grap)

As a result, the project is no longer being actively developed.

A detailed post on the feature engineering and algorithm approach is coming soon.

### Setup
---------------------
Code is in python 2.7. 
Install the dependencies listed in the _requirements.txt_ file

The `main.py` file contains the starter code, and is heavily commented. 

The `main` file is also available as a jupyter notebook.

The majority of the machine learning tasks are accomplished using scikit-learn, with the exception of the Hidden Markov Model,
which leverages [hmmlearn](https://github.com/hmmlearn/hmmlearn)


### About the Data
------------------

The accelerometer and gyroscope data were streamed from a MetaWear C sensor, made by the guys at [mbientlabs](https://mbientlab.com/). Highly recommended for wearable development.


### Full server and Android app
--------------------

I have a flask server which runs this classification algorithm on data sent from an Android app. This code, however, was 
written to get the job done, not be shared. If you're ready to wade through some very ropey code, here are the links:

[Flask server](https://github.com/ChristopherGS/sensor_readings)
[Android app](https://github.com/ChristopherGS/mbient_sensor_pigeon/tree/master/MbientBasic)