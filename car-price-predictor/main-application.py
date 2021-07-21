#flask library
from flask import Flask, render_template, request, url_for

#pickle library
import pickle

#pandas and numpy (dataframe) library
import pandas as pd
import numpy as np

#CORS Policy library
from flask_cors import CORS, cross_origin

#Flask main application
app = Flask(__name__,template_folder='view')

#open and read the LR-model
lrModel = pickle.load(open('F:\Python-ML-Project\car-price-predictor\LR-Model.pkl','rb'))

#open and read the RF-model
rfModel = pickle.load(open('F:\Python-ML-Project\car-price-predictor\RF-Model.pkl','rb'))

#open and read data in csv
car = pd.read_csv('F:\Python-ML-Project\car-price-predictor\data-set.csv')

@app.route("/", methods = ['GET', 'POST'])
def home():

    #store respective fields for the user to select the car details
    carCompany = sorted(car['model'].unique())

    carModelNames = sorted(car['name'].unique())

    manufacturedYear = sorted(car['year'].unique(),reverse=True)

    fuelType = car['fuel'].unique()

    return render_template( 'index.html', companies = carCompany, models = carModelNames, years = manufacturedYear, fuelTypes = fuelType )

@app.route('/predict-car-price',methods=['POST'])
@cross_origin()
def predictPrice():

    carCompany = request.form.get('company')

    carBrand = request.form.get('car_models')

    year = request.form.get('year')

    fuelType = request.form.get('fuel_type')

    kmDriven = request.form.get('kilo_driven')

    mileage = request.form.get('mileage')

    engine = request.form.get('engine')

    seats = request.form.get('seats')

    predictionOfLrModel = lrModel.predict(pd.DataFrame(columns=['name', 'model', 'year', 'km_driven', 'fuel', 'mileage', 'engine', 'seats'],
                 data = np.array([carBrand, carCompany, year, kmDriven, fuelType, mileage, engine, seats]).reshape(1,8)))
    
    predictionOfRfModel = rfModel.predict(pd.DataFrame(columns=['name', 'model', 'year', 'km_driven', 'fuel', 'mileage', 'engine', 'seats'],
                 data = np.array([carBrand, carCompany, year, kmDriven, fuelType, mileage, engine, seats]).reshape(1,8)))
    
    predictedPrice = (np.round(predictionOfLrModel[0], 2) + np.round(predictionOfRfModel[0], 2))//2

    return str(predictedPrice)

if(__name__=="__main__"):
    app.run(debug=True)
