import pandas as pd
import numpy as np
import streamlit as st
import ast
import pickle
st.set_page_config(
    page_title="Wellness Guidance System",
)
svc = pickle.load(open('svc.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
df = pd.read_csv('Training.csv')
description_df = pd.read_csv('description.csv')
precautions_df = pd.read_csv('precautions.csv')
medications_df = pd.read_csv('medications.csv')
diets_df = pd.read_csv('diets.csv')
workouts_df = pd.read_csv('workouts.csv')
def bring_name_in_right_format(name) :
    if '_' in name:
        return " ".join([a.strip() for a in name.split('_')])
    return name
correct_name = []
for col in df.columns:
    correct_name.append(bring_name_in_right_format(col))
def get_predicted_value(patient_symptoms):
    output = []
    for item in correct_name[:-1]:
        if item in patient_symptoms:
            output.append(1)
        else:
            output.append(0)
    return label_encoder.inverse_transform([svc.predict([output])])[0]
def helper(disease) :
    description = " ".join(description_df[description_df['Disease'] == disease]['Description'])
    precaution = precautions_df[precautions_df['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()[0]
    medication = medications_df[medications_df['Disease'] == disease]['Medication'].values[0]
    diet = diets_df[diets_df['Disease'] == disease]['Diet'].values[0]
    workout = workouts_df[workouts_df['disease'] == disease]['workout'].values.tolist()
    return description, precaution, ast.literal_eval(medication), ast.literal_eval(diet), workout


st.title('🩺 Wellness Guidance System')


# Multi-selection dropdown
patient_symptoms = st.multiselect("Select your symptoms:", correct_name)

# Predict button
if (patient_symptoms is not None and len(patient_symptoms) > 0) and st.button('Predict'):
    predicted_disease = get_predicted_value(patient_symptoms)
    st.subheader(f"📌 Predicted Disease: {predicted_disease}")
    description, precautions, medications, diet, workout = helper(predicted_disease)

    # Save results in session state
    st.session_state['predicted'] = True
    st.session_state['description'] = description
    st.session_state['precautions'] = precautions
    st.session_state['medications'] = medications
    st.session_state['diet'] = diet
    st.session_state['workout'] = workout

# Initialize or reset active section
if 'active_section' not in st.session_state:
    st.session_state['active_section'] = None

# Button handlers to update active section
# 4 columns + 2-column-wide button = total 6 columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📝   Description", use_container_width=True):
        st.session_state['active_section'] = 'description'
with col2:
    if st.button("🛡️   Precaution", use_container_width=True):
        st.session_state['active_section'] = 'precautions'
with col3:
    if st.button("💊   Medication", use_container_width=True):
        st.session_state['active_section'] = 'medications'
with col4:
    if st.button("🥗   Diet ", use_container_width=True):
        st.session_state['active_section'] = 'diet'
with col5:
    if st.button("🏋️   Lifestyle", use_container_width=True):
        st.session_state['active_section'] = 'workout'

# Show section based on active state
if st.session_state.get('predicted'):
    section = st.session_state['active_section']
    if section == 'description':
        st.info(st.session_state['description'])
    elif section == 'precautions':
        st.markdown("<ol>" + "".join(f"<li>{p}</li>" for p in st.session_state['precautions']) + "</ol>", unsafe_allow_html=True)
    elif section == 'medications':
        st.markdown("<ol>" + "".join(f"<li>{m}</li>" for m in st.session_state['medications']) + "</ol>", unsafe_allow_html=True)
    elif section == 'diet':
        st.markdown("<ol>" + "".join(f"<li>{d}</li>" for d in st.session_state['diet']) + "</ol>", unsafe_allow_html=True)
    elif section == 'workout':
        st.markdown("<ol>" + "".join(f"<li>{w}</li>" for w in st.session_state['workout']) + "</ol>", unsafe_allow_html=True)
