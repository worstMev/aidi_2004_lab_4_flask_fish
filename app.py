from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("./reg_model_fish.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #features 'Length1', 'Length2', 'Length3', 'Height', 'Width', 'Species' to be one hot encoded
        #species become : 'Species_Bream', 'Species_Parkki', 'Species_Perch', 'Species_Pike', 'Species_Roach', 'Species_Smelt', 'Species_Whitefish'
        # Get JSON data from request
        species_data = [
                {'column_name': 'Species_Bream', 'value': 'Bream'},
                {'column_name': 'Species_Parkki', 'value': 'Parkki'},
                {'column_name': 'Species_Perch', 'value': 'Perch'},
                {'column_name': 'Species_Pike', 'value': 'Pike'},
                {'column_name': 'Species_Roach', 'value': 'Roach'},
                {'column_name': 'Species_Smelt', 'value': 'Smelt'},
                {'column_name': 'Species_Whitefish', 'value': 'Whitefish'}
            ]
        data = request.get_json()
        
        # Convert JSON into DataFrame
        df = pd.DataFrame(data, index=[0])
        print('df :',df)
        for spec in species_data:
            df[spec.get('column_name')] = spec.get('value') == df['Species']
        df_encoded = df.drop(columns=['Species'])
        print('df :',df_encoded)
        

        # Make prediction
        prediction = model.predict(df_encoded)

        # Return the prediction
        return jsonify({"prediction": prediction[0]})
        #return "okokokok, pred"
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
