import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import leafmap.foliumap as leafmap
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import FuncFormatter
import folium as folium
#%%

st.title('Boat Analysis (2020)')

@st.cache
def load_data():
    df=pd.DataFrame()
    df=pd.read_csv("MarineTraffic_2020.csv")
    df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP, format='%Y-%m-%d %H:%M:%S')  
    return df
df=load_data()

@st.cache
def load_data_type():        
    df_type=pd.DataFrame()
    df_type=pd.read_csv('SHIPTYPES.csv')
    return df_type 
df_type=load_data_type()

@st.cache
def load_data_foc():        
    df_foc=pd.DataFrame()
    df_foc=pd.read_csv('Flags_of_convenience.csv')
    return df_foc 
df_foc=load_data_foc()


#%%

def get_data(boat_type,minLAT,maxLAT,minLON,maxLON):
    mask_area = (df['LAT'] >= float(minLAT)) & (df['LAT'] <= float(maxLAT)) & (df['LON'] >= float(minLON)) & (df['LON'] <= float(maxLON))
    df_area=pd.DataFrame()
    df_area= df.loc[mask_area]
    
    type_number=np.array([])
    for i in range(len(boat_type)):
        type_number=np.append(type_number,df_type.loc[df_type['TYPE'] == boat_type[i], 'NUMBER'])                              
    df_boat=pd.DataFrame()
    for i in range(len(type_number)):
        df_boat= pd.concat([df_boat, df_area[df_area['TYPE_GROUPED_ID']==type_number[i]]])
    
    df_boat.TIMESTAMP = pd.to_datetime(df_boat.TIMESTAMP, format='%Y-%m-%d %H:%M:%S')
    df_boat=df_boat.assign(MONTH = df_boat['TIMESTAMP'].dt.month )
    return df_boat

def Map_point(df_boat,date,boat_type):
    df_month=pd.DataFrame()
    df_month = pd.concat([df_month,df_boat.loc[(df_boat['MONTH'] == date)]])    
    y=0
    color=['red','fuchsia','orange','blue','green','purple','maroon','grey']
    for i in boat_type:
        fg = folium.FeatureGroup(name=i) 
        type=np.array([])
        type=np.append(type,df_type.loc[df_type['TYPE'] == i, 'NUMBER'])
        df_boat=pd.DataFrame()
        for a in range(len(type)):
            df_boat= pd.concat([df_boat, df_month[df_month['TYPE_GROUPED_ID']==type[a]]])        
        df_boat['count'] = 1
        lat=round(df_boat['LAT'],3)
        df_boat=df_boat.assign(LAT = lat )
        lon=round(df_boat['LON'],3)
        df_boat=df_boat.assign(LON = lon )
        df_boat = pd.DataFrame(df_boat.groupby(['LAT', 'LON'])['count'].sum().reset_index().values.tolist())
        for lat, lng in zip(df_boat[0], df_boat[1]):
            folium.CircleMarker([lat, lng],radius=1,popup=i,color=color[y],fill=True,fill_opacity=0.1).add_to(fg)
        fg.add_to(carte)  
        y+=1
    folium.LayerControl().add_to(carte)
    return df_boat

def Area(df_boat,boat_type):   
    numBoatData=pd.DataFrame()
    numBoat=pd.DataFrame()
    numBoatDataFoc=pd.DataFrame()
    numBoatFoc=pd.DataFrame()
    numBoatDataHazardous=pd.DataFrame()
    numBoatHazardous=pd.DataFrame()
    numBoatData, numBoat, numBoatDataFoc, numBoatFoc, numBoatDataHazardous, numBoatHazardous=Data(df_boat)
         
    st.pyplot(BoatType(df_boat,boat_type))
    #st.pyplot(GraphData(numBoatData,numBoatDataFoc,numBoatDataHazardous,boat_type,df_boat))
    st.pyplot(GraphBoat(numBoat,numBoatFoc,numBoatHazardous,boat_type,df_boat))
    st.pyplot(NumBoat(df_boat))
    
    df_foc_area=pd.DataFrame()
    for f in df_foc['FLAG']:
        df_foc_area= pd.concat([df_foc_area, df_boat.loc[df_boat['FLAG']==f]]) 
    st.pyplot(Circle(df_foc_area,boat_type))
        
def Data(df_area):    
    numBoatData=np.array([])
    numBoat=np.array([])
    numBoatDataFoc=np.array([])
    numBoatFoc=np.array([])
    numBoatDataHazardous=np.array([])
    numBoatHazardous=np.array([])
    
    hazardous=np.array([])
    hazardous=np.append(hazardous,df_type.loc[df_type['HAZARDOUS'] == 'YES','NUMBER']) 
    
    df_monthfoc=pd.DataFrame()
    df_monthhazardous=pd.DataFrame()
    month=sorted(df_area['MONTH'].value_counts().index)
    for m in month:
        df_month=pd.DataFrame(df_area.loc[df_area['MONTH'] == m])
        numBoatData=np.append(numBoatData,len(df_month))
        df_duplicate=pd.DataFrame(df_month)
        df_duplicate.drop_duplicates(subset = "MMSI", keep = 'first', inplace=True)
        numBoat=np.append(numBoat,len(df_duplicate))
        df_monthfoc=pd.DataFrame()
        for f in df_foc['FLAG']:
            df_monthfoc= pd.concat([df_monthfoc, df_month.loc[df_month['FLAG']==f]])
        numBoatDataFoc=np.append(numBoatDataFoc,len(df_monthfoc))
        df_monthfoc.drop_duplicates(subset = "MMSI", keep = 'first', inplace=True)
        numBoatFoc=np.append(numBoatFoc,len(df_monthfoc))
        df_monthhazardous=pd.DataFrame()
        for h in hazardous:
            df_monthhazardous= pd.concat([df_monthhazardous, df_month.loc[df_month['TYPE_GROUPED_ID']==h]])
        numBoatDataHazardous=np.append(numBoatDataHazardous,len(df_monthhazardous))
        df_monthhazardous.drop_duplicates(subset = "MMSI", keep = 'first', inplace=True)
        numBoatHazardous=np.append(numBoatHazardous,len(df_monthhazardous))
    return numBoatData, numBoat, numBoatDataFoc, numBoatFoc, numBoatDataHazardous, numBoatHazardous


def BoatType(df,boat_type):
    
    df_boat=pd.DataFrame()
    df_boat=pd.DataFrame(df['TYPE_GROUPED_ID'].value_counts())
    nb_boat=np.array([])
    for i in df_boat.index:
        nb_boat=np.append(nb_boat,df_type.loc[df_type['NUMBER'] == i, 'TYPE'])
    df_boat=df_boat.assign(TYPE = nb_boat )
    som_boat=np.array([])
    for i in df_boat.index:
        som_boat=np.append(som_boat,df_boat.loc[df_boat['TYPE']==df_boat['TYPE'][i]]['TYPE_GROUPED_ID'].sum())
    df_boat=df_boat.assign(SOMME = som_boat )
    df_boat=df_boat.sort_values(by=['SOMME'],ascending=False)
    #df_boat=df_boat.reindex(liste)
    df_boat.drop_duplicates(subset = "TYPE", keep = 'first', inplace=True)
    df_boat.set_index('TYPE', inplace = True)
    liste=list(df_boat.index)
    
    df_boat_unique=pd.DataFrame(df)
    df_boat_unique.drop_duplicates(subset = "MMSI", keep = 'first', inplace=True)
    df_boat_unique=pd.DataFrame(df_boat_unique['TYPE_GROUPED_ID'].value_counts())
    nb_boat=np.array([])
    for i in df_boat_unique.index:
        nb_boat=np.append(nb_boat,df_type.loc[df_type['NUMBER'] == i, 'TYPE'])
    df_boat_unique=df_boat_unique.assign(TYPE = nb_boat )
    som_boat=np.array([])
    for i in df_boat_unique.index:
        som_boat=np.append(som_boat,df_boat_unique.loc[df_boat_unique['TYPE']==df_boat_unique['TYPE'][i]]['TYPE_GROUPED_ID'].sum())
    df_boat_unique=df_boat_unique.assign(SOMME = som_boat )
    df_boat_unique=df_boat_unique.sort_values(by=['SOMME'],ascending=False)
    #liste=list(df_boat_unique.index)
    df_boat_unique.drop_duplicates(subset = "TYPE", keep = 'first', inplace=True)
    df_boat_unique.set_index('TYPE', inplace = True) 
    df_boat_unique=df_boat_unique.reindex(liste)
    
    X_axis = np.arange(len(df_boat.index))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(X_axis-0.2, df_boat['SOMME']* 1e-3,0.4,color='grey',edgecolor ='white')
    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%1.0fk'))
    ax.set_ylabel("Volume of data")
    ax2=ax.twinx()
    ax2.bar(X_axis+0.2, df_boat_unique['SOMME'],0.4,color='brown',edgecolor ='white')
    #ax2.set_yticks(color='red')
    plt.xticks(X_axis, df_boat.index)
    ax2.yaxis.label.set_color('maroon')
    ax2.tick_params(axis='y', colors='maroon')
    ax2.set_ylabel("Number of Boat")
    ax.set_title("%s in 2020"%(', '.join(boat_type)))
    #plt.rcParams["figure.figsize"] = (10, 6)
    return fig

def GraphData(numBoatData,numBoatDataFoc,numBoatDataHazardous,boat_type,df_boat):
    month=sorted(df_boat['MONTH'].value_counts().index)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('Set1')
    ax.bar(month, numBoatData , color='grey',edgecolor ='white')
    plt.bar(0,0,color='red')
    plt.bar(1,1,color='maroon')
    ax.bar(month, numBoatDataFoc , color='red',edgecolor ='white')
    if len(numBoatDataHazardous)!=0:
        ax.bar(month, numBoatDataHazardous , color='maroon',edgecolor ='white')
        ax.legend(['Boat','FOC','Hazardous'],bbox_to_anchor=(1, 0.50),loc='center left')
    else:
        ax.legend(['Boat','FOC'],bbox_to_anchor=(1, 0.50),loc='center left')
    ax.set_ylabel("Volume of boats")
    ax.set_xticks(range(1,13))
    ax.set_xlabel("Months")
    ax.set_title("%s in 2020"%(', '.join(boat_type)))
    return fig

def GraphBoat(numBoat,numBoatFoc,numBoatHazardous,boat_type,df_boat):
    month=sorted(df_boat['MONTH'].value_counts().index)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('Set1')
    ax.bar(month, numBoat , color='grey',edgecolor ='white')
    plt.bar(0,0,color='red')
    plt.bar(1,1,color='maroon')
    ax.bar(month, numBoatFoc , color='red',edgecolor ='white')
    if len(numBoatHazardous)!=0:
        ax.bar(month, numBoatHazardous , color='maroon',edgecolor ='white')
        #ax.legend(['Boat','FOC','Hazardous'],bbox_to_anchor=(1, 0.50),loc='center left')
        ax.legend(['Boat','FOC','Hazardous'])
    else:
        #ax.legend(['Boat','FOC'],bbox_to_anchor=(1, 0.50),loc='center left')
        ax.legend(['Boat','FOC'])
    ax.set_ylabel("Number of boats")
    ax.set_xticks(range(1,13))
    ax.set_xlabel("Months")
    ax.set_title("%s in 2020"%(', '.join(boat_type)))
    return fig

def NumBoat(df_boat):
    fig, ax = plt.subplots(figsize=(10, 6))
    df_month=pd.DataFrame()
    month=sorted(df_boat['MONTH'].value_counts().index)
    for i in df_boat['TYPE'].value_counts().index:#filter by TYPE
        df_type=pd.DataFrame(df_boat.loc[df_boat['TYPE'] == i])
        numBoat=np.array([])
        for y in month:#filter by MONTH
            df_month=pd.DataFrame(df_type.loc[df_type['MONTH'] == y])
            df_month.drop_duplicates(subset = "MMSI", keep = 'first', inplace=True)        
            numBoat=np.append(numBoat,len(df_month.MMSI))
            
        ax.plot(month,numBoat,label=i, linewidth=3)#Display data
    ax.set_xlabel("Months") 
    ax.set_ylabel("Volume of Boat") 
    ax.set_xticks(range(1,13))
    ax.set_title("%s in 2020"%(', '.join(boat_type)))
    ax.legend()
    #ax.legend(bbox_to_anchor=(1, 0.50),loc='center left')
    return fig

def Circle(df,boat_type):
    fig= plt.subplots()
    df.drop_duplicates(subset = "MMSI", keep = 'first', inplace=True)
    df_boatfoc=pd.DataFrame()
    for i in range(len(df_foc['FLAG'])):
        df_boatfoc=pd.concat([df_boatfoc, df.loc[df['FLAG']==df_foc['FLAG'][i]]])
    country=np.array([])
    for i in range(len(df_boatfoc['FLAG'].value_counts().index)):
        country=np.append(country,df_foc.loc[df_foc['FLAG'] == df_boatfoc['FLAG'].value_counts().index[i], 'COUNTRY'].iloc[0])

    df_percentages=pd.DataFrame()
    df_percentages=pd.DataFrame(country,columns = ['COUNTRY'])
    percentages=[df_boatfoc['FLAG'].value_counts()[i]*100/df_boatfoc['FLAG'].value_counts().sum() for i in range(len(df_boatfoc['FLAG'].value_counts()))]
    df_percentages=df_percentages.assign(PERCENT =  percentages)
    df_percentages = df_percentages.loc[df_percentages['PERCENT'] >= 4]
    df_percentages.loc[len(df_percentages)+1]=[ 'Other', (100-float(df_percentages['PERCENT'].sum())), ]

    plt.pie(df_percentages['PERCENT'],labels= list(df_percentages['COUNTRY'] ),autopct='%1.1f%%', pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.title("FOC %s"%(', '.join(boat_type)) + " in 2020")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    return fig
#%%


with st.form("Map"): 
    boat_type = st.multiselect('Choice ship types',sorted(df_type['TYPE'].value_counts().index),['CARGO', 'CONTAINER', 'TANKER','CARRIER']) 
    
    st.write('Entry gps coordinates in the polygone area:')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        minLAT = st.text_input("minimum Latitude", 37.51)
    with col2:
        maxLAT = st.text_input("maximum Latitude", 37.77)
    with col3:
        minLON = st.text_input("minimum Longitude", 26.30)
    with col4:
        maxLON = st.text_input("maximum Longitude", 26.61)
    submitted = st.form_submit_button("Load Data")
    if submitted:
        df_boat=get_data(boat_type,minLAT,maxLAT,minLON,maxLON)
        Area(df_boat,boat_type) 
        st.write('* FOC=Flag Of Convenience')
        st.write('** Number of boats=Number of MMSI number.')
         

df_boat=get_data(boat_type,minLAT,maxLAT,minLON,maxLON)
date=st.slider('Display months data',1,12,1) 
carte = leafmap.Map(center=[(float(minLAT)+float(maxLAT))/2, (float(minLON)+float(maxLON))/2], zoom=10)
folium.Rectangle([(np.max(df.LAT),np.min(df.LON)), (np.min(df.LAT),np.max(df.LON))]).add_to(carte)
Map_point(df_boat,date,boat_type)
carte.to_streamlit(width=700, height=500)
   
  
    
    
    
    
    
    
    
    
