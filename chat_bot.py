import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import openai
from test import get_completion
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

training = pd.read_csv("Data/Training.csv")
testing = pd.read_csv("Data/Testing.csv")
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training["prognosis"]
y1 = y

reduced_data = training.groupby(training["prognosis"]).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)
testx = testing[cols]
testy = testing["prognosis"]
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)


model = SVC()
model.fit(x_train, y_train)


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty("voice", "english+f5")
    engine.setProperty("rate", 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        print("You should take the consultation from a doctor.")
    else:
        print("It might not be that bad, but you should take precautions.")


def getDescription():
    global description_list
    with open("MasterData/symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open("MasterData/symptom_severity.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open("MasterData/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(" ", "_")
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp, feature_names):
    df = pd.read_csv("Data/Training.csv")
    X = df.iloc[:, :-1]
    y = df["prognosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=20
    )
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])[0]


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

llm_model = "gpt-3.5-turbo"


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = str(input(""))

        dataset = [
            "itching",
            "skin_rash",
            "nodal_skin_eruptions",
            "continuous_sneezing",
            "shivering",
            "chills",
            "joint_pain",
            "stomach_pain",
            "acidity",
            "ulcers_on_tongue",
            "muscle_wasting",
            "vomiting",
            "burning_micturition",
            "spotting_urination",
            "fatigue",
            "weight_gain",
            "anxiety",
            "cold_hands_and_feets",
            "mood_swings",
            "weight_loss",
            "restlessness",
            "lethargy",
            "patches_in_throat",
            "irregular_sugar_level",
            "cough",
            "high_fever",
            "sunken_eyes",
            "breathlessness",
            "sweating",
            "dehydration",
            "indigestion",
            "headache",
            "yellowish_skin",
            "dark_urine",
            "nausea",
            "loss_of_appetite",
            "pain_behind_the_eyes",
            "back_pain",
            "constipation",
            "abdominal_pain",
            "diarrhoea",
            "mild_fever",
            "yellow_urine",
            "yellowing_of_eyes",
            "acute_liver_failure",
            "fluid_overload",
            "swelling_of_stomach",
            "swelled_lymph_nodes",
            "malaise",
            "blurred_and_distorted_vision",
            "phlegm",
            "throat_irritation",
            "redness_of_eyes",
            "sinus_pressure",
            "runny_nose",
            "congestion",
            "chest_pain",
            "weakness_in_limbs",
            "fast_heart_rate",
            "pain_during_bowel_movements",
            "pain_in_anal_region",
            "bloody_stool",
            "irritation_in_anus",
            "neck_pain",
            "dizziness",
            "cramps",
            "bruising",
            "obesity",
            "swollen_legs",
            "swollen_blood_vessels",
            "puffy_face_and_eyes",
            "enlarged_thyroid",
            "brittle_nails",
            "swollen_extremeties",
            "excessive_hunger",
            "extra_marital_contacts",
            "drying_and_tingling_lips",
            "slurred_speech",
            "knee_pain",
            "hip_joint_pain",
            "muscle_weakness",
            "stiff_neck",
            "swelling_joints",
            "movement_stiffness",
            "spinning_movements",
            "loss_of_balance",
            "unsteadiness",
            "weakness_of_one_body_side",
            "loss_of_smell",
            "bladder_discomfort",
            "foul_smell_of urine",
            "continuous_feel_of_urine",
            "passage_of_gases",
            "internal_itching",
            "toxic_look_(typhos)",
            "depression",
            "irritability",
            "muscle_pain",
            "altered_sensorium",
            "red_spots_over_body",
            "belly_pain",
            "abnormal_menstruation",
            "dischromic _patches",
            "watering_from_eyes",
            "increased_appetite",
            "polyuria",
            "family_history",
            "mucoid_sputum",
            "rusty_sputum",
            "lack_of_concentration",
            "visual_disturbances",
            "receiving_blood_transfusion",
            "receiving_unsterile_injections",
            "coma",
            "stomach_bleeding",
            "distention_of_abdomen",
            "history_of_alcohol_consumption",
            "fluid_overload.1",
            "blood_in_sputum",
            "prominent_veins_on_calf",
            "palpitations",
            "painful_walking",
            "pus_filled_pimples",
            "blackheads",
            "scurring",
            "skin_peeling",
            "silver_like_dusting",
            "small_dents_in_nails",
            "inflammatory_nails",
            "blister",
            "red_sore_around_nose",
        ]

        style = """Analyze the user prompt and extract keywords related to healthcare that are present in both the user prompt and the given dataset.
        Consider variations in user input. User might enter wrong spellings but in the end, RETURN ONLY THE RELEVANT KEYWORD MATCHED. NOTHING ELSE
        """

        prompt = f"""Evaluate the user prompt delimited by triple backticks.
        The dataset includes various healthcare-related keywords.
        Consider variations in user input and potential wrong spellings.
        Only respond to queries related to the dataset, else just say that I am a medical chat bot trained only on medical diseases data and I can't help with anyother query.
        Following is the dataset: {dataset}.
        Following are the thinks you need to keep in mind{style}.
        text: ```{disease_input}```
        """
        api_response = get_completion(prompt)
        parsed_response = api_response.split(":")[-1].strip()


        conf, cnf_dis = check_pattern(chk_dis, parsed_response)
        if conf == 1:
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            parsed_response = cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days = int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[
                reduced_data.loc[present_disease].values[0].nonzero()
            ]
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms, "? : ", end="")


            second_prediction = sec_predict(symptoms_exp, feature_names)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                if present_disease[0] in description_list:
                    print("You may have ", present_disease[0])
                    print(description_list[present_disease[0]])
                else:
                    print(f"No description found for {present_disease[0]}")

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print("second_prediction:", second_prediction)
                if second_prediction[0] in description_list:
                    print(description_list[second_prediction[0]])
                else:
                    print(f"No description found for {second_prediction[0]}")

            precution_list = precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for i, j in enumerate(precution_list):
                print(i + 1, ")", j)

    recurse(0, 1)

app = Flask(__name__)
CORS(app)


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_prompt = data["user_prompt"]
    response = get_completion(user_prompt)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
