import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# ================== Load Data ==================
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# Load trained model
svc = pickle.load(open('svc.pkl', 'rb'))

# ================== Helper Functions ==================
def predict_disease(symptoms):
    input_features = [0] * len(sym_des.columns)
    prediction = svc.predict([input_features])
    return prediction

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# ================== Symptom and Disease Mapping ==================
symptoms_dict = {
    # your long symptoms dictionary here (same as you gave)
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
    # ... all other symptoms
    'yellow_crust_ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    # ... all other diseases
    27: 'Impetigo'
}

# ================== Prediction Function ==================
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    pred = svc.predict([input_vector])[0]
    return diseases_list.get(pred, "Unknown Disease")

# ================== Flask Routes ==================

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)

        if symptoms == "Symptoms" or not symptoms.strip():
            message = "Please either write symptoms or you have written misspelled symptoms."
            return render_template('index.html', message=message)
        else:
            user_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(',')]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions_list, medications_list, diet_list, workout_list = helper(predicted_disease)

            my_precautions = []
            for i in precautions_list[0]:
                my_precautions.append(i)

            return render_template('index.html', predicted_disease=predicted_disease,
                                   dis_des=dis_des,
                                   my_precautions=my_precautions,
                                   medications=medications_list,
                                   my_diet=diet_list,
                                   workout=workout_list)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

# ================== Run Flask App ==================
if __name__ == '__main__':
    app.run(debug=True)
