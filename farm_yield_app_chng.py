

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image


st.set_page_config(page_title='Farm Yield Prediction Model',layout="wide")

model=joblib.load("Farm Yield v2.sav")
env_df=pd.read_excel("new data.xlsx")
perform_df=pd.read_excel("Model Performance.xlsx")
env_df=env_df.dropna()

season_dict={'Maize':"Rabi",'Paddy':"Kharif",'Sugarcane':"Kharif", 'Wheat':"Rabi" }

#%%

# Header Display
st.markdown('<div style="text-align: center; color:#004F92 ;font-size:40px; font-weight:bold">DeepSpatial Agriverse Platform</div>', unsafe_allow_html=True)
st.markdown('<div style="background-color:#00609C;padding:7px"> <h2 style="color:white;text-align:center;">Farm Yield Predictor</h2> </div>',unsafe_allow_html=True)
st.header("")

left,right=st.columns(2)
# Take Inputs
with right:
    model_rltd,hrvst_rltd=st.columns(2)
    with model_rltd:
        crop=st.selectbox("Crop",['Maize','Paddy','Sugarcane', 'Wheat' ])
        district=st.selectbox("District",("Dehradun", "Champawat"),disabled=True)
        block=st.selectbox("Block",("Vikasnagar", "Dalu"),disabled=True)
        village=st.selectbox("Village",env_df['Village Name'].unique())
        farm_num_df=env_df.groupby(['Village Name','Crop_Name'])['Farm_ID'].unique().reset_index()
        pivot_df=pd.pivot_table(farm_num_df,index="Village Name",columns='Crop_Name',values="Farm_ID")
        pivot_df_2=pivot_df.applymap(lambda z:z[:10])
        vlg_df=pivot_df_2.loc[village,:]
        farm_opts=np.append(np.append(vlg_df[0],vlg_df[1]),vlg_df[2])
        farm_opts=np.array(farm_opts,dtype=int)
        farm=st.selectbox("Farm Number",farm_opts)
        season=season_dict[crop]
    
    with hrvst_rltd:
        last_season=st.selectbox("Select Last Harvest Season",["Kharif","Rabi"])
        last_crop=st.selectbox("Select the crop harvested", ["Wheat","Maize","Sugarcane","Paddy"])
        last_yield=st.number_input("Enter the Yield of crop (in kgs)",step=50)
        
# ['Area', 'Humidity', 'K', 'N', 'P', 'Precipitation', 'Temperature',
#        'Crop Name_Maize', 'Crop Name_Paddy', 'Crop Name_Sugarcane',
#        'Crop Name_Wheat']

inp_df=env_df.set_index(['Village Name','Farm_ID'])
n=inp_df.loc[(village,farm),'N']
p=inp_df.loc[(village,farm),'P']
k=inp_df.loc[(village,farm),'K']
ph=inp_df.loc[(village,farm),'pH']
area=inp_df.loc[(village,farm),'Area (Hectares)']
rain=inp_df.loc[(village,farm),'Rainfall']    
temp=inp_df.loc[(village,farm),'Temperature']
humid=inp_df.loc[(village,farm),'Humidity']
if crop=="Maize":
    crop_enc=[1,0,0,0]
elif crop=="Paddy":
    crop_enc=[0,1,0,0]
elif crop=="Sugarcane":
    crop_enc=[0,0,1,0]    
elif crop=="Wheat":
    crop_enc=[0,0,0,1]    


with left:
    c1,c2=st.columns(2)
    with c1:
        fig_temp = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = temp,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Temperature","font":{"size":24,"color":"red"}},  
        gauge = {
                'axis': {'range': [None, 40], 'tickwidth': 1, 'tickcolor': "darkred"},
                'bar': {'color': "#8b0000"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#FFFF00'},
                    {'range': [20, 40], 'color': '#FFA500'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 38}}))
        fig_temp.update_layout(height=250)
        st.plotly_chart(fig_temp,use_container_width=True)

    with c2:
        fig_humid = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = humid,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Humidity","font":{"size":24,"color":"darkblue"}},
            gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'cyan'},
                        {'range': [50, 100], 'color': 'royalblue'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 98}}))
        fig_humid.update_layout(height=250)
        st.plotly_chart(fig_humid,use_container_width=True)

    a1,a2,a3=st.columns(3)    
    with a2:
      crp = Image.open("imgs/"+str(crop)+".jpg")
      crp_1=crp.resize((200,150))
      st.image(crp_1)

    

if st.button("Calculate",use_container_width=True):
    inp_x=np.append([area,humid,k,n,p,rain,temp],crop_enc)
    farm_yield=np.round(model.predict([inp_x])[0],2)
    season_text="Season - "+season
    z1,z2=st.columns(2)
    with z1:
        st.subheader(season_text)
        string_2=("Crop Yield of "+str(crop) +" = "+str(farm_yield)+"(tonnes/ha)").title()
        st.subheader(string_2)   


st.markdown("<br></br>",unsafe_allow_html=True)        
x1,x2,x3=st.columns(3)
with x2:
    st.caption("Model Accuracies for Crop Advisory")
    perform_df.index=[1,2,3,4,5,6]
    perform_df['CV Accuracy']=np.round(perform_df['CV Accuracy'],2)
    perform_df['CV Accuracy']=perform_df['CV Accuracy'].apply(lambda z:str(z)+"%")
    perform_df=perform_df[['Model','CV Accuracy']]
    st.dataframe(perform_df,use_container_width=True)














