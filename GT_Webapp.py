import streamlit as st

st.set_page_config(
    page_title="GTPPD-2.0",
    page_icon="GTPPD-2.0.jpeg",
    layout="wide"
)
import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import plotly.express as px

from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import time
import plotly.graph_objects as go

import os
import pickle


# ==========================================
# PAGE CONFIG
# ==========================================

# st.set_page_config(
#     page_title="GTPPD-2.0",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# ==========================================
# CREATE MODEL DIRECTORY
# ==========================================

os.makedirs("correct_models", exist_ok=True)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    /* ==========================================
       MAIN BODY BACKGROUND
    ========================================== */

    .stApp {
        background-color: white;
    }

    /* ==========================================
       SIDEBAR BACKGROUND
    ========================================== */

    section[data-testid="stSidebar"] {
        background-color: black;
        color: white !important;
    }

    /* ==========================================
    SIDEBAR CONTENT CENTERING
    ========================================== */

    section[data-testid="stSidebar"] > div {

        display: flex;
        flex-direction: column;
        justify-content: center;

    }

    /* ==========================================
    SIDEBAR TITLE
    ========================================== */

    section[data-testid="stSidebar"] h1 {

        text-align: center;
        color: white;

    }

    /* ==========================================
    SELECTBOX CONTAINER
    ========================================== */

    div[data-testid="stSidebarUserContent"] {

        padding-top: 4vh;
        height: 100%;
        margin-top: 30%;

    }

    /* ==========================================
       SIDEBAR TEXT COLOR
    ========================================== */

    section[data-testid="stSidebar"] * {
        color: white !important;
        text-align: center;
    }

    /* ==========================================
       HIDE DEPLOY BUTTON
    ========================================== */

    .stDeployButton {
        display: none !important;
    }

    /* ==========================================
    KEEP HAMBURGER MENU VISIBLE
    ========================================== */

    [data-testid="collapsedControl"] {

        display: flex !important;

        visibility: visible !important;

        background-color: black !important;

        color: white !important;

        border-radius: 8px;

        padding: 6px;

    }

    /* ==========================================
    HEADER BACKGROUND
    ========================================== */

    header[data-testid="stHeader"] {

        background-color: white !important;

    }

    /* ==========================================
       HIDE STREAMLIT MENU
    ========================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ==========================================
    HAMBURGER MENU COLOR
    ========================================== */

    button[kind="header"] {

        background-color: black !important;

        color: white !important;

        border-radius: 8px;

    }

    /* ==========================================
    CENTER ALL HEADERS
    ========================================== */

    h1 {

        text-align: center !important;
        color: #4FC3F7 !important;

    }

    h1.home-title {

        text-align: center;
        width: 100%;
        height: 100%;

        margin-top: 10%;

        color: #4FC3F7 !important;

    }

    p {

        color: black !important;

        font-size: 18px !important;

        line-height: 1.8 !important;

    }

    p.home-description {

        text-align: center;
        width: 100%;
        height: 100%;
        font-weight: 500;
        margin-top: 0%;

        color: black;

    }

    [data-testid="stMarkdownContainer"] {

        text-align: left;

        font-family: "Poppins", sans-serif;

    }

    /* ==========================================
    TOP HEADER
    ========================================== */

    header[data-testid="stHeader"] {

        background-color: white;

    }

    /* ==========================================
    SIDEBAR TOGGLE BUTTON
    ========================================== */

    [data-testid="collapsedControl"] {

        color: black !important;

    }

    /* ==========================================
       SELECTBOX STYLING
    ========================================== */

    div[data-baseweb="select"] > div {
        background-color: #1e1e1e;
        color: white;
        border-radius: 10px;
    }

    /* ==========================================
    MAIN TEXT COLOR
    ========================================== */

    html, body, [class*="css"] {
        color: black;
    }

    /* ==========================================
    DATAFRAME BACKGROUND
    ========================================== */

    [data-testid="stDataFrame"] {
        background-color: white;
        color: black;
    }

    /* ==========================================
    TABLE TEXT
    ========================================== */

    table {
        color: black !important;
    }

    /* ==========================================
    METRIC CARDS
    ========================================== */

    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #dcdcdc;
        padding: 10px;
        border-radius: 10px;
        color: black;
    }

    /* ==========================================
    SELECTBOX TEXT
    ========================================== */

    div[data-baseweb="select"] span {
        color: white !important;
    }

    /* ==========================================
    INPUT BOXES
    ========================================== */

    input {
        background-color: white !important;
        color: black !important;
    }

    /* ==========================================
    MULTISELECT
    ========================================== */

    div[data-baseweb="tag"] {
        background-color: #333333 !important;
        color: white !important;
    }

    /* ==========================================
    SLIDER TEXT
    ========================================== */

    .stSlider label {
        color: black !important;
    }

    /* ==========================================
    SUCCESS MESSAGE
    ========================================== */

    div[data-testid="stAlert"] {
        color: black;
    }

    /* ==========================================
    SUBHEADERS
    ========================================== */

    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    /* ==========================================
    PARAGRAPH TEXT
    ========================================== */

    p {
        color: black !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# DOWNLOAD FUNCTION
# ==========================================

def download_file(file_id, output_path):

    if not os.path.exists(output_path):

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(
            url,
            output_path,
            quiet=False
        )


# # ==========================================
# # DOWNLOAD MODEL FILES
# # ==========================================

download_file(
    "1uoOJMFP9HQtx31Pq4qiq1GKvQKB7as41",
    "correct_models/EXHAUST TEMPERATURE_ANN.pth"
)

download_file(
    "1xxhBGpL-be5_Sdt4tIVksXRGJEuWqTi8",
    "correct_models/EXHAUST TEMPERATURE_Decision Tree.pkl"
)

download_file(
    "1ItB1z-R9RvGWPMszSDdZLAD4dXC_ZPO3",
    "correct_models/EXHAUST TEMPERATURE_metrics.csv"
)

download_file(
    "1OT8Kap6VB6uekkNqk22Mc5X-Id3BZ8V3",
    "correct_models/EXHAUST TEMPERATURE_Random Forest.pkl"
)

download_file(
    "1Nq8Yz2jRmLKWS9_eF1JDvKXRL7umM_V-",
    "correct_models/EXHAUST TEMPERATURE_SVM.pkl"
)

download_file(
    "1OSp81dDcYT7qgx3Da8GWSLADXNfdOwhS",
    "correct_models/EXHAUST TEMPERATURE_XGBoost.pkl"
)

download_file(
    "1annVK83DkzIOc1G2x8WEFW149svk2iG9",
    "correct_models/features.pkl"
)

download_file(
    "15xS27M2H91KaFJmJA7IlKUV016gH2mh_",
    "correct_models/OUTPUT POWER_ANN.pth"
)

download_file(
    "1iNTUZQcWQ2AbayGRuBK6-q013yEJ5Gah",
    "correct_models/OUTPUT POWER_Decision Tree.pkl"
)

download_file(
    "1RaQm0BoiZdSQO-cpdL_fd4TxVORWB0Rn",
    "correct_models/OUTPUT POWER_metrics.csv"
)

download_file(
    "1h8vN4D_OYvyFc8Fa83WDvvLoTBt6EeSx",
    "correct_models/OUTPUT POWER_Random Forest.pkl"
)

download_file(
    "1drpQtgU8YAYUOXCvbutcxWqCxubxASmb",
    "correct_models/OUTPUT POWER_SVM.pkl"
)

download_file(
    "1uHvQfXmP9r7zf5od-4REk39o1xbvu7FN",
    "correct_models/OUTPUT POWER_XGBoost.pkl"
)

download_file(
    "1-HxpgA8BD2s5Zo0dqnuwDFw4ChtqjYDn",
    "correct_models/scaler.pkl"
)

download_file(
    "1KhbVZf67LQljHp-_z9rAH7OCjEaOCU1U",
    "correct_models/SFC_ANN.pth"
)

download_file(
    "1S94gkAKMOk7yDxbhHYsZcMnkiGyRLvGD",
    "correct_models/SFC_Decision Tree.pkl"
)

download_file(
    "1HzffTa7cRZrMZGXsSn3UpZW40zCnXbXD",
    "correct_models/SFC_metrics.csv"
)

download_file(
    "1CyBPKTt3B6UrtPJtCOy7kiTI4jORynbl",
    "correct_models/SFC_Random Forest.pkl"
)

download_file(
    "1WScwYFIIn64b3-AI40dggVIHTtfyWb6_",
    "correct_models/SFC_SVM.pkl"
)

download_file(
    "1V1zqAEeK403e90ijeXBcaDmo5l6bSl9u",
    "correct_models/SFC_XGBoost.pkl"
)

download_file(
    "1AGgaJhHg9dMbdLSXa2rjwUAtBMGk03AP",
    "correct_models/X_test_eff.pkl"
)

download_file(
    "1udpfR4tbItq_2_jEiVms04Cx_GuIVv0B",
    "correct_models/X_test_power.pkl"
)

download_file(
    "1Qcnd8Abit0bAiMfABHQ70bVGe63-pKKb",
    "correct_models/X_test_sfc.pkl"
)

download_file(
    "1YtzAAznpdGVry21oRkoK0smP-V0K9WTp",
    "correct_models/y_test_eff.pkl"
)

download_file(
    "1ZRpWAipfcKj1KWqS0IGhyXydHTBBNm3h",
    "correct_models/y_test_power.pkl"
)

download_file(
    "1LMKMMlVkFwQbS8TQR9TJgWYMfUsAVBaT",
    "correct_models/y_test_sfc.pkl"
)

import os

st.write(
    "EXHAUST ANN:",
    os.path.getsize(
        "correct_models/EXHAUST TEMPERATURE_ANN.pth"
    )
)

st.write(
    "OUTPUT ANN:",
    os.path.getsize(
        "correct_models/OUTPUT POWER_ANN.pth"
    )
)

st.write(
    "SFC ANN:",
    os.path.getsize(
        "correct_models/SFC_ANN.pth"
    )
)

# st.markdown("""
#     <style>

#     div.about-container{

#         display:flex;

#         flex-direction:column;

#         justify-content:center;

#         align-items:center;

#         text-align:center;

#         padding-top:80px;

#         padding-left:50px;

#         padding-right:50px;

#     }

#     p.about-text{

#         font-size:20px;

#         line-height:1.8;

#         max-width:900px;

#         color:#333333;

#         margin-top:20px;

#     }

#     ul.about-list{

#         font-size:20px;

#         line-height:2;

#         margin-top:20px;

#         text-align:left;

#     }

#     p.about-footer{

#         margin-top:30px;

#         font-size:18px;

#         font-weight:bold;

#         color:#111111;

#     }

#     </style>
#     """, unsafe_allow_html=True
# )

# st.sidebar.title("Navigation")

st.sidebar.markdown(
    "<h2 style='color:white;'>Navigation</h2>",
    unsafe_allow_html=True
)

section = st.sidebar.selectbox(
    " ",
    [
        "Home",
        "Dataset Overview",
        "Data Preprocessing and Visualization",
        "Model Training and Evaluation",
        "New Prediction",
        "About the Project",
    ]
)

# ==========================================
    # LOAD DATASET
    # ==========================================

file_path = "Gas_Turbine_Dataset.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(file_path, sheet_name="combined_data")
    return df

df = load_data()

if section == "Home":
    #st.title("GT PERFORMANCE PROGNOSTICS DASHBOARD")
    st.markdown("""
    <h1 class="home-title">
        GT PERFORMANCE PROGNOSTICS DASHBOARD
    </h1>

    <p class="home-description">
        A very fast and reliable prediction web application for gas turbines.
    </p>
    """, unsafe_allow_html=True)
    # st.markdown(
    #     """
    #     <div style="
    #         text-align: center;
    #         padding-top: 12%;
    #     ">

    #         <h1>
    #             Gas Turbine Performance Dashboard
    #         </h1>

    #         <p style="
    #             font-size:20px;
    #             width:80%;
    #             margin:auto;
    #             line-height:1.8;
    #         ">

    #             This intelligent dashboard is designed for
    #             advanced gas turbine monitoring, visualization,
    #             machine learning prediction, and performance analysis.

    #             The system integrates data preprocessing,
    #             interactive visualization, machine learning models,
    #             deep learning prediction, and future forecasting
    #             techniques for industrial turbine analytics.

    #         </p>

    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )
elif section == "Dataset Overview":
    # ==========================================
    # STREAMLIT PAGE CONFIG
    # ==========================================

    # st.set_page_config(
    #     page_title="Gas Turbine Dashboard",
    #     layout="wide"
    # )

    st.title("Gas Turbine Dataset Overview")



    # ==========================================
    # DISPLAY DATASET INFORMATION
    # ==========================================

    def display_dataset_info(df):

        st.subheader("Dataset Shape")
        st.write(df.shape)

        st.subheader("First 5 Rows")
        st.dataframe(df.head())

        st.subheader("Dataset Info")

        # Capture df.info() output
        from io import StringIO

        buffer = StringIO()
        df.info(buf=buffer)

        s = buffer.getvalue()

        st.text(s)

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(
            columns={
                "index": "Feature",
                0: "Missing Values"
            }
        ))

        st.subheader("Data Types")
        st.dataframe(df.dtypes.reset_index().rename(
            columns={
                "index": "Feature",
                0: "Data Type"
            }
        ))


    display_dataset_info(df)
    

elif section == "Data Preprocessing and Visualization":

    st.title("Data Preprocessing & Visualization")

    # ==========================================
    # HANDLE OUTLIERS FUNCTION
    # ==========================================

    def handle_outliers(df):

        df = df.copy()

        # ==========================================
        # SELECT NUMERICAL & CATEGORICAL COLUMNS
        # ==========================================

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # ==========================================
        # HANDLE MISSING VALUES
        # ==========================================

        df[numerical_cols] = df[numerical_cols].fillna(
            df[numerical_cols].mean()
        )

        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        st.subheader("Missing Values After Cleaning")
        st.dataframe(df.isnull().sum().reset_index().rename(
            columns={
                "index": "Feature",
                0: "Missing Values"
            }
        ))

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df.head())

        # ==========================================
        # FEATURE SELECTION
        # ==========================================

        selected_feature = st.selectbox(
            "Choose a Feature",
            numerical_cols
)

        # ==========================================
        # HISTOGRAMS
        # ==========================================

        st.subheader("Feature Distributions")

        for col in [selected_feature]:
            fig, ax = plt.subplots(figsize=(10, 5))

            sns.histplot(df[col], kde=True, ax=ax)

            ax.set_title(f"Distribution of {col}")

            st.pyplot(fig)

        # ==========================================
        # BOXPLOTS BEFORE OUTLIER REMOVAL
        # ==========================================

        st.subheader("Boxplots Before Outlier Reduction")

        for col in [selected_feature]:

            fig, ax = plt.subplots(figsize=(6, 4))

            sns.boxplot(x=df[col], ax=ax)

            ax.set_title(f"Boxplot of {col}")

            st.pyplot(fig)

        # ==========================================
        # OUTLIER HANDLING
        # ==========================================

        for col in numerical_cols:

            # Quantile clipping
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)

            df[col] = np.clip(df[col], lower, upper)

            # IQR Method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df[col] = np.where(
                df[col] < lower_bound,
                lower_bound,
                df[col]
            )

            df[col] = np.where(
                df[col] > upper_bound,
                upper_bound,
                df[col]
            )

        st.success("IQR Outlier Reduction Applied")

        # ==========================================
        # SAVITZKY-GOLAY SMOOTHING
        # ==========================================

        for col in numerical_cols:

            if len(df[col]) > 11:

                df[col] = savgol_filter(
                    df[col],
                    window_length=15,
                    polyorder=2
                )

        st.success("Savitzky-Golay Smoothing Applied")

        # ==========================================
        # ROBUST SCALING
        # ==========================================

        scaler = RobustScaler()

        scaled_data = scaler.fit_transform(df[numerical_cols])

        df_scaled = pd.DataFrame(
            scaled_data,
            columns=numerical_cols
        )

        st.success("Robust Scaling Completed")

        # ==========================================
        # PCA
        # ==========================================

        pca = PCA(n_components=0.95)

        X_pca = pca.fit_transform(df_scaled)

        st.success("PCA Completed")

        st.write("Original Shape:", df_scaled.shape)

        st.write("Reduced Shape:", X_pca.shape)

        # ==========================================
        # FINAL DATASET
        # ==========================================

        st.subheader("Final Cleaned Dataset Shape")

        st.write(df_scaled.shape)

        st.subheader("First 5 Rows of Scaled Dataset")

        st.dataframe(df_scaled.head())

        # ==========================================
        # BOXPLOTS AFTER OUTLIER REMOVAL
        # ==========================================

        st.subheader("Boxplots After Outlier Reduction")

        for col in [selected_feature]:

            fig, ax = plt.subplots(figsize=(6, 4))

            sns.boxplot(x=df_scaled[col], ax=ax)

            ax.set_title(f"Cleaned Boxplot of {col}")

            st.pyplot(fig)
        
        # ==========================================
        # NUMERICAL FEATURES
        # ==========================================

        numerical_df = df.select_dtypes(
            include=['int64', 'float64']
        )

        # ==========================================
        # CORRELATION HEATMAP
        # ==========================================

        st.subheader("Correlation Heatmap")

        corr_matrix = numerical_df.corr()

        fig, ax = plt.subplots(figsize=(15, 10))

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap of Gas Turbine Dataset"
        )

        st.pyplot(fig)

        st.title("Gas Turbine Data Visualization")

    

        # ==========================================
        # FEATURE SELECTION FOR 3D PLOTS
        # ==========================================

        st.subheader("3D Feature Visualization")

        col1 = st.selectbox(
            "Select X-axis Feature",
            numerical_df.columns,
            index=25
        )

        col2 = st.selectbox(
            "Select Y-axis Feature",
            numerical_df.columns,
            index=7
        )

        col3 = st.selectbox(
            "Select Z-axis Feature",
            numerical_df.columns,
            index=0
        )

        # ==========================================
        # 3D SCATTER PLOT
        # ==========================================

        # fig = plt.figure(figsize=(10, 8))

        # ax = fig.add_subplot(
        #     111,
        #     projection='3d'
        # )

        # ax.scatter(
        #     df[col1],
        #     df[col2],
        #     df[col3],
        #     alpha=0.7
        # )

        # ax.set_xlabel(col1)
        # ax.set_ylabel(col2)
        # ax.set_zlabel(col3)

        # ax.set_title(
        #     "3D Visualization of Gas Turbine Performance"
        # )

        # st.pyplot(fig)

        # ==========================================
        # COLORED 3D SCATTER PLOT
        # ==========================================

        st.subheader("Colored 3D Visualization")

        angle = st.slider(
            "Rotate View Angle",
            min_value=0,
            max_value=360,
            value=45
        )

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(
            111,
            projection='3d'
        )

        scatter = ax.scatter(
            df[col1],
            df[col2],
            df[col3],
            c=df[col3],
            alpha=0.7
        )

        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_zlabel(col3)

        ax.set_title(
            "Colored 3D Gas Turbine Visualization"
        )

        fig.colorbar(
            scatter,
            ax=ax,
            shrink=0.5
        )

        # Rotation
        ax.view_init(
            elev=30,
            azim=angle
        )

        st.pyplot(fig)

        # # ==========================================
        # # ANIMATION SECTION
        # # ==========================================

        # st.subheader("Interactive 3D Rotation")

        # angle = st.slider(
        #     "Rotate View Angle",
        #     min_value=0,
        #     max_value=360,
        #     value=45
        # )

        # fig = plt.figure(figsize=(10, 8))

        # ax = fig.add_subplot(
        #     111,
        #     projection='3d'
        # )

        # scatter = ax.scatter(
        #     df[col1],
        #     df[col2],
        #     df[col3],
        #     c=df[col3],
        #     alpha=0.7
        # )

        # ax.set_xlabel(col1)
        # ax.set_ylabel(col2)
        # ax.set_zlabel(col3)

        # ax.set_title(
        #     "Animated Gas Turbine Visualization"
        # )

        # # Rotation
        # ax.view_init(
        #     elev=30,
        #     azim=angle
        # )

        # st.pyplot(fig)

        # import plotly.graph_objects as go

        # ==========================================
        # AUTO ANIMATED 3D PLOT
        # ==========================================

        st.subheader("Auto Animated 3D Visualization")

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=df[col1],
                    y=df[col2],
                    z=df[col3],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=df[col3],
                        colorscale='Viridis',
                        opacity=0.8
                    )
                )
            ]
        )

        # Animation frames
        frames = []

        for angle in range(0, 360, 5):

            camera = dict(
                eye=dict(
                    x=2 * np.cos(np.radians(angle)),
                    y=2 * np.sin(np.radians(angle)),
                    z=1
                )
            )

            frames.append(
                go.Frame(
                    layout=dict(
                        scene_camera=camera
                    )
                )
            )

        fig.frames = frames

        # Play button
        fig.update_layout(
            scene=dict(
                xaxis_title=col1,
                yaxis_title=col2,
                zaxis_title=col3
            ),
            title="Auto Rotating Gas Turbine Visualization",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    fromcurrent=True
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        return df_scaled, X_pca


    # ==========================================
    # RUN FUNCTION
    # ==========================================

    df_scaled, X_pca = handle_outliers(df)
    
#elif section == "Data Visualization":

    
    # # ==========================================
    # # PLOTLY INTERACTIVE 3D PLOT
    # # ==========================================

    # st.subheader("Interactive 3D Plot")

    # fig_plotly = px.scatter_3d(
    #     df,
    #     x=col1,
    #     y=col2,
    #     z=col3,
    #     color=col3,
    #     title="Interactive Gas Turbine 3D Plot"
    # )

    # st.plotly_chart(
    #     fig_plotly,
    #     use_container_width=True
    # )

    # ==========================================
    # SECOND 3D VISUALIZATION
    # ==========================================

    # st.subheader("Turbine Performance Space")

    # fig = plt.figure(figsize=(10, 8))

    # ax = fig.add_subplot(
    #     111,
    #     projection='3d'
    # )

    # scatter = ax.scatter(
    #     df[col1],
    #     df[col2],
    #     df[col3],
    #     c=df[col3],
    #     alpha=0.7
    # )

    # ax.set_xlabel(col1)
    # ax.set_ylabel(col2)
    # ax.set_zlabel(col3)

    # ax.set_title(
    #     "Turbine Performance Space"
    # )

    # st.pyplot(fig)
    


    # # ==========================================
    # # CONTINUOUS 3D ANIMATION
    # # ==========================================

    # st.subheader("Continuous 3D Animation")

    # plot_placeholder = st.empty()

    # angle = 0

    # while True:

    #     fig = go.Figure(
    #         data=[
    #             go.Scatter3d(
    #                 x=df[col1],
    #                 y=df[col2],
    #                 z=df[col3],
    #                 mode='markers',
    #                 marker=dict(
    #                     size=4,
    #                     color=df[col3],
    #                     colorscale='Viridis',
    #                     opacity=0.8
    #                 )
    #             )
    #         ]
    #     )

    #     # Rotating camera
    #     fig.update_layout(
    #         scene=dict(
    #             xaxis_title=col1,
    #             yaxis_title=col2,
    #             zaxis_title=col3,
    #             camera=dict(
    #                 eye=dict(
    #                     x=2 * np.cos(np.radians(angle)),
    #                     y=2 * np.sin(np.radians(angle)),
    #                     z=1
    #                 )
    #             )
    #         ),
    #         title="Continuous Rotating Gas Turbine Visualization"
    #     )

    #     plot_placeholder.plotly_chart(
    #         fig,
    #         use_container_width=True
    #     )

    #     time.sleep(0.05)
    #     angle += 5

elif section == "Model Training and Evaluation":
    
    st.title("Model Training and Evaluation")

    # import os

    # st.write(
    #     "features.pkl exists:",
    #     os.path.exists("correct_models/features.pkl")
    # )
    
    # st.write(
    #     "features.pkl size:",
    #     os.path.getsize("correct_models/features.pkl")
    # )
    
    # with open("correct_models/features.pkl", "rb") as f:
    #     st.write(
    #         "First 100 bytes:",
    #         f.read(100)
    #     )

    # ==========================================
    # LOAD SAVED FILES
    # ==========================================

    scaler = pickle.load(
        open("correct_models/scaler.pkl", "rb")
    )

    features = pickle.load(
        open("correct_models/features.pkl", "rb")
    )

    # ==========================================
    # TARGET VARIABLES
    # ==========================================

    target_power = df['OUTPUT POWER']

    target_sfc = df['FUEL GAS FLOW']

    target_efficiency = df['EXHAUST TEMPERATURE']

    # ==========================================
    # DASHBOARD OVERVIEW
    # ==========================================

    st.subheader("Selected Features")

    st.write(features)

    st.subheader("Target Variables")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Target 1",
            target_power.name
        )

    with col2:
        st.metric(
            "Target 2",
            target_sfc.name
        )

    with col3:
        st.metric(
            "Target 3",
            target_efficiency.name
        )

    # # ==========================================
    # # FEATURES
    # # ==========================================

    # X = df[features]

    # X = X.select_dtypes(include=[np.number])

    # X = X.fillna(X.mean())

    # # ==========================================
    # # SCALE FEATURES
    # # ==========================================

    # X_scaled = scaler.transform(X)

    # ==========================================
    # SPLITS
    # ==========================================

    # X_train_power, X_test_power, y_train_power, y_test_power = train_test_split(
    #     X_scaled,
    #     target_power,
    #     test_size=0.20,
    #     random_state=400
    # )

    # X_train_sfc, X_test_sfc, y_train_sfc, y_test_sfc = train_test_split(
    #     X_scaled,
    #     target_sfc,
    #     test_size=0.25,
    #     random_state=42
    # )

    # X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(
    #     X_scaled,
    #     target_efficiency,
    #     test_size=0.25,
    #     random_state=42
    # )

    # ==========================================
    # LOAD SAVED TEST DATA
    # ==========================================

    X_test_power = pickle.load(
        open(
            "correct_models/X_test_power.pkl",
            "rb"
        )
    )

    y_test_power = pickle.load(
        open(
            "correct_models/y_test_power.pkl",
            "rb"
        )
    )

    X_test_sfc = pickle.load(
        open(
            "correct_models/X_test_sfc.pkl",
            "rb"
        )
    )

    y_test_sfc = pickle.load(
        open(
            "correct_models/y_test_sfc.pkl",
            "rb"
        )
    )

    X_test_eff = pickle.load(
        open(
            "correct_models/X_test_eff.pkl",
            "rb"
        )
    )

    y_test_eff = pickle.load(
        open(
            "correct_models/y_test_eff.pkl",
            "rb"
        )
    )

        # ==========================================
        # TARGET DATA
        # ==========================================

    target_data = {

        "OUTPUT POWER": {

            "X_test": X_test_power,
            "y_test": y_test_power

        },

        "SFC": {

            "X_test": X_test_sfc,
            "y_test": y_test_sfc

        },

        "EXHAUST TEMPERATURE": {

            "X_test": X_test_eff,
            "y_test": y_test_eff

        }
    }

        # ==========================================
        # TARGET DROPDOWN
        # ==========================================

    selected_target = st.selectbox(

            "Choose Target Variable",

            list(target_data.keys())

        )

        # ==========================================
        # MODEL DROPDOWN
        # ==========================================

    model_options = [

        "ANN",
        "Random Forest",
        "Decision Tree",
        "SVM",
        "XGBoost"

    ]

    selected_model = st.selectbox(
        "Choose Model",
        
        model_options

        )

    # # ==========================================
    # # LOAD MODEL
    # # ==========================================

    # model_path = f"saved_models/{selected_target}_{selected_model}.pkl"

    # model = pickle.load(
    #     open(model_path, "rb")
    # )

    # ==========================================
    # ANN MODEL CLASS
    # ==========================================

    class ANNModel(nn.Module):

        def __init__(self, input_size):

            super(ANNModel, self).__init__()

            self.network = nn.Sequential(

                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.ReLU(),

                nn.Linear(128, 64),
                nn.ReLU(),

                nn.Linear(64, 32),
                nn.ReLU(),

                nn.Linear(32, 1)

            )

        def forward(self, x):

            return self.network(x)

    # ==========================================
    # LOAD MODEL
    # ==========================================

    if selected_model == "ANN":

        input_size = len(features)

        model = ANNModel(input_size)

        model.load_state_dict(

            torch.load(

                f"correct_models/{selected_target}_ANN.pth",

                map_location=torch.device("cpu"),
                weights_only=False
            )
        )

        model.eval()

    else:

        model_path = f"correct_models/{selected_target}_{selected_model}.pkl"

        model = pickle.load(

            open(model_path, "rb")
        )

    # ==========================================
    # LOAD METRICS
    # ==========================================

    metrics_path = f"correct_models/{selected_target}_metrics.csv"

    metrics_df = pd.read_csv(metrics_path)

    selected_metrics = metrics_df[
        metrics_df["Model"] == selected_model
    ]

    # ==========================================
    # DISPLAY METRICS
    # ==========================================

    st.subheader(f"{selected_model} Metrics")

    st.dataframe(
        selected_metrics,
        use_container_width=True
    )

    # ==========================================
    # GET TEST DATA
    # ==========================================

    X_test = target_data[selected_target]["X_test"]

    y_test = target_data[selected_target]["y_test"]

    # ==========================================
    # PREDICTIONS
    # ==========================================

    # y_pred = model.predict(X_test)

    # ==========================================
    # PREDICTIONS
    # ==========================================

    if selected_model == "ANN":

        X_tensor = torch.tensor(
            X_test,
            dtype=torch.float32
        )

        with torch.no_grad():

            y_pred = model(
                X_tensor
            ).numpy().flatten()

    else:

        y_pred = model.predict(X_test)

    # ==========================================
    # ACTUAL VS PREDICTED
    # ==========================================

    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.set_style("darkgrid")

    ax.plot(

        y_pred[:200],

        label="Predicted",

        linestyle='--',

        linewidth=2

    )

    ax.plot(

        y_test.values[:200],

        label="Actual",

        linewidth=2.5

    )

    ax.set_title(

        f"{selected_model} - {selected_target}"

    )

    ax.set_xlabel("Samples")

    ax.set_ylabel(selected_target)

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    # ==========================================
    # FUTURE PREDICTIONS
    # ==========================================

    future_steps = 25

    last_data = X_test[-future_steps:]

    # future_predictions = model.predict(last_data)


    if selected_model == "ANN":

        future_tensor = torch.tensor(
            last_data,
            dtype=torch.float32
        )

        with torch.no_grad():

            future_predictions = model(
                future_tensor
            ).numpy().flatten()

    else:

        future_predictions = model.predict(last_data)

    # ==========================================
    # FUTURE PREDICTION GRAPH
    # ==========================================

    st.subheader("Future Prediction")

    fig, ax = plt.subplots(figsize=(14, 6))

    actual_values = y_test.values[:195]

    predicted_values = y_pred[:195]

    x_existing = np.arange(len(actual_values))

    x_future = np.arange(

        len(actual_values),

        len(actual_values) + future_steps

    )

    # Actual Values
    ax.plot(

        x_existing,

        actual_values,

        label='Actual',

        linewidth=2.5

    )

    # Existing Predictions
    ax.plot(

        x_existing,

        predicted_values,

        linestyle='--',

        linewidth=2,

        label='Predicted Existing'

    )

    # Future Predictions
    ax.plot(

        x_future,

        future_predictions,

        linewidth=2,

        label='Future Prediction',

        linestyle='--'

    )

    # Separator
    ax.axvline(

        x=len(actual_values)-1,

        linestyle=':',

        linewidth=2

    )

    ax.set_title(

        f"{selected_model} Future Prediction"

    )

    ax.set_xlabel(

        'Samples / Future Cycles'

    )

    ax.set_ylabel(

        selected_target

    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    # ==========================================
    # MODEL COMPARISON
    # ==========================================

    st.subheader("Model Comparison")

    comparison_df = pd.read_csv(metrics_path)

    # ==========================================
    # BEST MODEL IDENTIFICATION
    # ==========================================

    best_model_row = comparison_df.loc[
        comparison_df["R2 Score"].idxmax()
    ]

    best_model_name = best_model_row["Model"]

    best_r2 = best_model_row["R2 Score"]

    best_rmse = best_model_row["RMSE"]

    st.subheader("Best Performing Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Model",
            best_model_name
        )

    with col2:
        st.metric(
            "Best R2 Score",
            round(best_r2, 4)
        )

    with col3:
        st.metric(
            "RMSE",
            round(best_rmse, 4)
        )

    # ==========================================
    # SUCCESS MESSAGE
    # ==========================================

    st.success(
        f"{best_model_name} is currently the best model for {selected_target}"
    )

    colors = plt.cm.viridis(
        np.linspace(
            0.25,
            0.65,
            len(comparison_df)
        )
    )


    # R2 SCORE
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(

        comparison_df["Model"],

        comparison_df["R2 Score"],

        color = colors

    )

    ax.set_title(

        f"{selected_target} - R2 Score"

    )

    ax.set_xlabel("Models")

    ax.set_ylabel("R2 Score")

    ax.grid(True)

    st.pyplot(fig)

    # RMSE

    colors1 = plt.cm.Set3(
        np.linspace(
            0,
            1,
            len(comparison_df)
        )
    )
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(

        comparison_df["Model"],

        comparison_df["RMSE"],

        color = plt.cm.viridis(
        np.linspace(
             0.15,
             0.65,
            len(comparison_df)
        )
    )

    )

    ax.set_title(

        f"{selected_target} - RMSE"

    )

    ax.set_xlabel("Models")

    ax.set_ylabel("RMSE")

    ax.grid(True)

    st.pyplot(fig)

elif section == "New Prediction":
    st.title("New Prediction for Gas Turbine Performance")

    st.subheader("Enter Feature Values")

    # ==========================================
    # LOAD SAVED FILES
    # ==========================================

    scaler = pickle.load(
        open("correct_models/scaler.pkl", "rb")
    )

    features = pickle.load(
        open("correct_models/features.pkl", "rb")
    )

    # ==========================================
    # USER INPUTS
    # ==========================================

    user_inputs = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        if i % 2 == 0:

            with col1:

                value = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.2f"
                )

        else:

            with col2:

                value = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.2f"
                )

        user_inputs.append(value)

    # ==========================================
    # PREDICTION BUTTON
    # ==========================================

    if st.button("Predict Turbine Performance"):

        # ==========================================
        # CONVERT INPUT
        # ==========================================

        new_data = np.array([user_inputs])

        new_data_scaled = scaler.transform(new_data)

        # ==========================================
        # TARGETS
        # ==========================================

        targets = [
            "OUTPUT POWER",
            "SFC",
            "EXHAUST TEMPERATURE"
        ]

        predictions = {}

        # ==========================================
        # ANN MODEL CLASS
        # ==========================================

        class ANNModel(nn.Module):

            def __init__(self, input_size):

                super(ANNModel, self).__init__()

                self.network = nn.Sequential(

                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(256, 128),
                    nn.ReLU(),

                    nn.Linear(128, 64),
                    nn.ReLU(),

                    nn.Linear(64, 32),
                    nn.ReLU(),

                    nn.Linear(32, 1)

                )

            def forward(self, x):

                return self.network(x)

        # ==========================================
        # LOOP THROUGH TARGETS
        # ==========================================

        for target in targets:

            # ==========================================
            # LOAD METRICS
            # ==========================================

            metrics_path = f"correct_models/{target}_metrics.csv"

            metrics_df = pd.read_csv(metrics_path)

            # ==========================================
            # BEST MODEL
            # ==========================================

            best_model_row = metrics_df.loc[
                metrics_df["R2 Score"].idxmax()
            ]

            best_model_name = best_model_row["Model"]
            best_rmse = best_model_row["RMSE"]

            # ==========================================
            # LOAD MODEL
            # ==========================================

            if best_model_name == "ANN":

                input_size = len(features)

                model = ANNModel(input_size)

                model.load_state_dict(

                    torch.load(
                        f"correct_models/{target}_ANN.pth",
                        map_location=torch.device("cpu")
                    )

                )

                model.eval()

                tensor_data = torch.tensor(
                    new_data_scaled,
                    dtype=torch.float32
                )

                with torch.no_grad():

                    prediction = model(
                        tensor_data
                    ).numpy().flatten()[0]

            else:

                model_path = f"correct_models/{target}_{best_model_name}.pkl"

                model = pickle.load(
                    open(model_path, "rb")
                )

                prediction = model.predict(
                    new_data_scaled
                )[0]

            # ==========================================
            # STORE RESULTS
            # ==========================================

            predictions[target] = {

                "model": best_model_name,

                "prediction": prediction,

                "rmse": best_rmse

            }

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================

        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "OUTPUT POWER",
                f"{predictions['OUTPUT POWER']['prediction']:.2f} ± {predictions['OUTPUT POWER']['rmse']:.2f}"
            )

            # st.write(
            #     f"Best Model: {predictions['OUTPUT POWER']['model']}"
            # )

        with c2:

            st.metric(
                "FUEL GAS FLOW",
                f"{predictions['SFC']['prediction']:.2f} ± {predictions['SFC']['rmse']:.2f}"
            )

            # st.write(
            #     f"Best Model: {predictions['SFC']['model']}"
            # )

        with c3:

            st.metric(
                "EXHAUST TEMPERATURE",
                f"{predictions['EXHAUST TEMPERATURE']['prediction']:.2f} ± {predictions['EXHAUST TEMPERATURE']['rmse']:.2f}"
            )

            st.subheader("Future Forecast Using User Input")

            # ==========================================
            # LOAD HISTORICAL DATA
            # ==========================================

            y_test_power = pickle.load(
                open("correct_models/y_test_power.pkl", "rb")
            )

            y_test_sfc = pickle.load(
                open("correct_models/y_test_sfc.pkl", "rb")
            )

            y_test_eff = pickle.load(
                open("correct_models/y_test_eff.pkl", "rb")
            )
    
    
    # # ==========================================
    #     # FUTURE FORECAST VISUALIZATION
    #     # ==========================================

    #     st.subheader(
    #         "Future Forecast Using New Input Data"
    #     )

    #     forecast_targets = {

    #         "OUTPUT POWER": {

    #             "history": y_test_power.values,

    #             "prediction": predictions[
    #                 "OUTPUT POWER"
    #             ]["prediction"]

    #         },

    #         "FUEL GAS FLOW": {

    #             "history": y_test_sfc.values,

    #             "prediction": predictions[
    #                 "SFC"
    #             ]["prediction"]

    #         },

    #         "EXHAUST TEMPERATURE": {

    #             "history": y_test_eff.values,

    #             "prediction": predictions[
    #                 "EXHAUST TEMPERATURE"
    #             ]["prediction"]

    #         }

    #     }

    #     # ==========================================
    #     # OUTPUT POWER
    #     # ==========================================

    #     st.subheader("OUTPUT POWER Forecast")

    #     history = forecast_targets[
    #         "OUTPUT POWER"
    #     ]["history"][-50:]

    #     future_value = forecast_targets[
    #         "OUTPUT POWER"
    #     ]["prediction"]

    #     fig, ax = plt.subplots(figsize=(12, 5))

    #     x_hist = np.arange(len(history))
    #     x_future = np.array([len(history)])

    #     ax.plot(
    #         x_hist,
    #         history,
    #         linewidth=2,
    #         label="Historical Data"
    #     )

    #     ax.scatter(
    #         x_future,
    #         [future_value],
    #         s=120,
    #         label="Prediction"
    #     )

    #     ax.plot(
    #         [x_hist[-1], x_future[0]],
    #         [history[-1], future_value],
    #         linestyle='--'
    #     )

    #     ax.set_title("OUTPUT POWER")
    #     ax.legend()
    #     ax.grid(True)

    #     st.pyplot(fig)

    #     # ==========================================
    #     # FUEL GAS FLOW
    #     # ==========================================

    #     st.subheader("FUEL GAS FLOW Forecast")

    #     history = forecast_targets[
    #         "FUEL GAS FLOW"
    #     ]["history"][-50:]

    #     future_value = forecast_targets[
    #         "FUEL GAS FLOW"
    #     ]["prediction"]

    #     fig, ax = plt.subplots(figsize=(12, 5))

    #     x_hist = np.arange(len(history))
    #     x_future = np.array([len(history)])

    #     ax.plot(
    #         x_hist,
    #         history,
    #         linewidth=2,
    #         label="Historical Data"
    #     )

    #     ax.scatter(
    #         x_future,
    #         [future_value],
    #         s=120,
    #         label="Prediction"
    #     )

    #     ax.plot(
    #         [x_hist[-1], x_future[0]],
    #         [history[-1], future_value],
    #         linestyle='--'
    #     )

    #     ax.set_title("FUEL GAS FLOW")
    #     ax.legend()
    #     ax.grid(True)

    #     st.pyplot(fig)

    #     # ==========================================
    #     # EXHAUST TEMPERATURE
    #     # ==========================================

    #     st.subheader("EXHAUST TEMPERATURE Forecast")

    #     history = forecast_targets[
    #         "EXHAUST TEMPERATURE"
    #     ]["history"][-50:]

    #     future_value = forecast_targets[
    #         "EXHAUST TEMPERATURE"
    #     ]["prediction"]

    #     fig, ax = plt.subplots(figsize=(12, 5))

    #     x_hist = np.arange(len(history))
    #     x_future = np.array([len(history)])

    #     ax.plot(
    #         x_hist,
    #         history,
    #         linewidth=2,
    #         label="Historical Data"
    #     )

    #     ax.scatter(
    #         x_future,
    #         [future_value],
    #         s=120,
    #         label="Prediction"
    #     )

    #     ax.plot(
    #         [x_hist[-1], x_future[0]],
    #         [history[-1], future_value],
    #         linestyle='--'
    #     )

    #     ax.set_title("EXHAUST TEMPERATURE")
    #     ax.legend()
    #     ax.grid(True)

    #     st.pyplot(fig)        # )
        
if section == "About the Project":
    
    st.html("""

    <div style='
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
        padding-top:30px;
        padding-left:50px;
        padding-right:50px;
        width: 80%;
        margin: 0 auto;
    '>

        <h1 style='
            font-size:55px;
            margin-bottom:30px;
            color: #4FC3F7 !important;
        '>

            About the Project

        </h1>

        <p style='
            font-size:22px;
            line-height:1.8;
            max-width:1000px;
            color:#333333;
            margin-top: 0px;
        '>

            GTPPD-2.0 is an intelligent machine learning
            dashboard designed for gas turbine
            performance analysis, monitoring,
            prediction, and visualization.

        </p>
            
            <p style='
                font-size:15px !important;
                margin-top:20px;'>
                Designed and Developed by: Olawuyi David Oluwatoni</p>

        <p style='
            font-size:15px !important;
            margin-top:30px;
            color:#111111;
            position: relative;
            bottom: -100px;
        '>

            Powered by Python - Streamlit - scikit-learn - PyTorch - XGBoost - Matplotlib - Seaborn- Plotly

        </p>

    </div>

    """)
    
    # features = df.drop(columns=['OUTPUT POWER', 'FUEL GAS FLOW', 'GT UNIT', 'EXHAUST TEMPERATURE', 
    #                         """'FUEL GAS PRESSURE2', 'ALLOWABLE SPREAD', 'LUBE OIL HEADER PRESSURE', 'TORQUE ANGLE'"""], errors='ignore').columns
    #                         #'EXHAUST SPREAD 1',  'LUBE OIL HEADER TEMPERATURE', 
    #                         #'COMPRESSOR INLET PRESSURE', 'GT SPEED', 'COMPRESSOR DISCHARGE PRESSURE', 'REACTIVE POWER', 'EXHAUST SPREAD 2', 'DEW POINT', 'COALESCING FILTER OUTLET TEMPERATURE',
    #                         #'COALESCING FILTER OUTLET PRESSURE', 'FUEL GAS PRESSURE1', 'SCRUBBER INTLET PRESSURE', 'IGV POSITION', 'MAXIMUM VIBRATION', 'FUEL GAS TEMPERATURE'], errors='ignore').columns

    # target_power = df['OUTPUT POWER']
    # target_sfc = df['FUEL GAS FLOW']
    # target_efficiency = df['EXHAUST TEMPERATURE']

    # st.title("Gas Turbine Machine Learning Dashboard")

    # # ==========================================
    # # DASHBOARD OVERVIEW
    # # ==========================================

    # st.subheader("Selected Features")

    # st.write(features)

    # st.subheader("Target Variables")

    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.metric(
    #         "Target 1",
    #         target_power.name
    #     )

    # with col2:
    #     st.metric(
    #         "Target 2",
    #         target_sfc.name
    #     )

    # with col3:
    #     st.metric(
    #         "Target 3",
    #         target_efficiency.name
    #     )

    # # ==========================================
    # # FEATURES
    # # ==========================================

    # X = df[features]

    # # ==========================================
    # # TARGETS
    # # ==========================================

    # y_power = target_power
    # y_sfc = target_sfc
    # y_efficiency = target_efficiency


    # # ==========================================
    # # HANDLE MISSING VALUES
    # # ==========================================

    # # # Numerical Columns
    # # numerical_cols = X.select_dtypes(
    # #     include=['int64', 'float64']
    # # ).columns

    # # # Fill numerical missing values
    # # X[numerical_cols] = X[numerical_cols].fillna(
    # #     X[numerical_cols].mean()
    # # )

    # # # Categorical Columns
    # # categorical_cols = X.select_dtypes(
    # #     include=['object']
    # # ).columns

    # # # Fill categorical missing values
    # # for col in categorical_cols:

    # #     X[col] = X[col].fillna(
    # #         X[col].mode()[0]
    # #     )

    # # ==========================================
    # # KEEP ONLY NUMERICAL FEATURES
    # # ==========================================

    # X = X.select_dtypes(
    #     include=[np.number]
    # )

    # # ==========================================
    # # REMOVE NaN VALUES
    # # ==========================================

    # X = X.fillna(X.mean())

    # y_power = y_power.fillna(y_power.mean())

    # y_sfc = y_sfc.fillna(y_sfc.mean())

    # y_efficiency = y_efficiency.fillna(
    #     y_efficiency.mean()
    # )

    # # ==========================================
    # # FEATURE SCALING
    # # ==========================================

    # scaler = StandardScaler()

    # X_scaled = scaler.fit_transform(X)

    # st.success("Feature Scaling Completed")

    # # ==========================================
    # # TRAIN TEST SPLITS
    # # ==========================================

    # X_train_power, X_test_power, y_train_power, y_test_power = train_test_split(
    #     X_scaled,
    #     y_power,
    #     test_size=0.20,
    #     random_state=400
    # )

    # X_train_sfc, X_test_sfc, y_train_sfc, y_test_sfc = train_test_split(
    #     X_scaled,
    #     y_sfc,
    #     test_size=0.25,
    #     random_state=42
    # )

    # X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(
    #     X_scaled,
    #     y_efficiency,
    #     test_size=0.25,
    #     random_state=42
    # )

    # # ==========================================
    # # DATASET CARDS
    # # ==========================================

    # st.subheader("Dataset Split Information")

    # c1, c2, c3 = st.columns(3)

    # with c1:

    #     st.info("OUTPUT POWER")

    #     st.write("Train Shape:")
    #     st.write(X_train_power.shape)

    #     st.write("Test Shape:")
    #     st.write(X_test_power.shape)

    # with c2:

    #     st.info("SFC")

    #     st.write("Train Shape:")
    #     st.write(X_train_sfc.shape)

    #     st.write("Test Shape:")
    #     st.write(X_test_sfc.shape)

    # with c3:

    #     st.info("EFFICIENCY")

    #     st.write("Train Shape:")
    #     st.write(X_train_eff.shape)

    #     st.write("Test Shape:")
    #     st.write(X_test_eff.shape)

    # # ==========================================
    # # MODEL OVERVIEW
    # # ==========================================

    # st.subheader("Machine Learning Models")

    # model_df = pd.DataFrame({

    #     "Model": [
    #         "ANN",
    #         "Random Forest",
    #         "Decision Tree",
    #         "SVM",
    #         "XGBoost"
    #     ],

    #     "Purpose": [
    #         "Neural Networks",
    #         "Ensemble Learning",
    #         "Tree-Based Method",
    #         "Kernel Learning",
    #         "Boosting Method"
    #     ]
    # })

    # st.dataframe(
    #     model_df,
    #     use_container_width=True
    # )

    # # ==========================================
    # # MODEL DEFINITIONS
    # # ==========================================

    # class ANNModel(nn.Module):

    #     def __init__(self, input_size):

    #         super(ANNModel, self).__init__()

    #         self.network = nn.Sequential(

    #             nn.Linear(input_size, 512),
    #             nn.BatchNorm1d(512),
    #             nn.ReLU(),
    #             nn.Dropout(0.3),

    #             nn.Linear(512, 256),
    #             nn.BatchNorm1d(256),
    #             nn.ReLU(),
    #             nn.Dropout(0.3),

    #             nn.Linear(256, 128),
    #             nn.ReLU(),

    #             nn.Linear(128, 64),
    #             nn.ReLU(),

    #             nn.Linear(64, 32),
    #             nn.ReLU(),

    #             nn.Linear(32, 1)

    #         )

    #     def forward(self, x):

    #         return self.network(x)

    # models = {

    #     "ANN": "PyTorch_ANN",

    #     "Random Forest": RandomForestRegressor(
    #         n_estimators=300,
    #         random_state=420
    #     ),

    #     "Decision Tree": DecisionTreeRegressor(
    #         random_state=420,
    #         criterion='squared_error'
    #     ),

    #     "SVM": SVR(
    #         kernel='rbf'
    #     ),

    #     "XGBoost": XGBRegressor(
    #         n_estimators=300,
    #         random_state=420
    #     )
    # }

    # # ==========================================
    # # EVALUATION FUNCTION
    # # ==========================================

    # def evaluate_models(
    #     X_train,
    #     X_test,
    #     y_train,
    #     y_test,
    #     target_name
    # ):

    #     results = []

    #     for name, model in models.items():

    #         # ==========================================
    #         # PYTORCH ANN
    #         # ==========================================

    #         if name == "ANN":

    #             # Convert to tensors
    #             X_train_tensor = torch.tensor(
    #                 X_train,
    #                 dtype=torch.float32
    #             )

    #             X_test_tensor = torch.tensor(
    #                 X_test,
    #                 dtype=torch.float32
    #             )

    #             y_train_tensor = torch.tensor(
    #                 y_train.values,
    #                 dtype=torch.float32
    #             ).view(-1, 1)

    #             # DataLoader
    #             train_dataset = TensorDataset(
    #                 X_train_tensor,
    #                 y_train_tensor
    #             )

    #             train_loader = DataLoader(
    #                 train_dataset,
    #                 batch_size=64,
    #                 shuffle=True
    #             )

    #             # Initialize ANN
    #             input_size = X_train.shape[1]

    #             ann_model = ANNModel(input_size)

    #             criterion = nn.MSELoss()

    #             optimizer = optim.Adam(
    #                 ann_model.parameters(),
    #                 lr=0.001
    #             )

    #             # Training Loop
    #             epochs = 1000

    #             losses = []

    #             for epoch in range(epochs):

    #                 ann_model.train()

    #                 epoch_loss = 0

    #                 for batch_X, batch_y in train_loader:

    #                     predictions = ann_model(batch_X)

    #                     loss = criterion(
    #                         predictions,
    #                         batch_y
    #                     )

    #                     optimizer.zero_grad()

    #                     loss.backward()

    #                     optimizer.step()

    #                     epoch_loss += loss.item()

    #                 avg_loss = epoch_loss / len(train_loader)

    #                 losses.append(avg_loss)

    #                 # st.write(
    #                 #     f"Epoch [{epoch+1}/{epochs}] "
    #                 #     f"Loss: {avg_loss:.6f}"
    #                 # )

    #             # Prediction
    #             ann_model.eval()

    #             with torch.no_grad():

    #                 y_pred = ann_model(
    #                     X_test_tensor
    #                 ).numpy().flatten()

    #         # ==========================================
    #         # SCIKIT-LEARN MODELS
    #         # ==========================================

    #         else:

    #             model.fit(X_train, y_train)

    #             y_pred = model.predict(X_test)

    #         # ==========================================
    #         # METRICS
    #         # ==========================================

    #         mae = mean_absolute_error(
    #             y_test,
    #             y_pred
    #         )

    #         mse = mean_squared_error(
    #             y_test,
    #             y_pred
    #         )

    #         rmse = np.sqrt(mse)

    #         r2 = r2_score(
    #             y_test,
    #             y_pred
    #         )

    #         # ==========================================
    #         # SAVE RESULTS
    #         # ==========================================

    #         results.append({

    #             'Model': name,
    #             'MAE': round(mae, 3),
    #             'RMSE': round(rmse, 3),
    #             'R2 Score': round(r2, 3)

    #         })

    #         # ==========================================
    #         # ACTUAL VS PREDICTED PLOT
    #         # ==========================================

    #         sns.set_style("darkgrid")

    #         fig, ax = plt.subplots(figsize=(12, 6))

    #         ax.plot(
    #             y_pred[:100],
    #             label="Predicted"
    #         )

    #         ax.plot(
    #             y_test.values[:100],
    #             label='Actual',
    #             linewidth=2
    #         )

    #         ax.set_title(
    #             f'{name} (Actual vs Predicted) - {target_name}'
    #         )

    #         ax.set_xlabel('Samples')

    #         ax.set_ylabel(target_name)

    #         ax.legend()

    #         ax.grid(True)

    #         st.pyplot(fig)

    #         # ==========================================
    #         # FUTURE PREDICTION VISUALIZATION
    #         # ==========================================

    #         sns.set_style("darkgrid")

    #         fig, ax = plt.subplots(figsize=(14, 6))

    #         # Existing Data
    #         actual_values = y_test.values[:100]

    #         predicted_values = y_pred[:100]

    #         x_existing = np.arange(len(actual_values))

    #         # Future Data
    #         future_steps = 30

    #         last_data = X_test[-future_steps:]

    #         # ==========================================
    #         # FUTURE PREDICTIONS
    #         # ==========================================

    #         if name == "ANN":

    #             future_tensor = torch.tensor(
    #                 last_data,
    #                 dtype=torch.float32
    #             )

    #             ann_model.eval()

    #             with torch.no_grad():

    #                 future_predictions = ann_model(
    #                     future_tensor
    #                 ).numpy().flatten()

    #         else:

    #             future_predictions = model.predict(
    #                 last_data
    #             )

    #         # Future Axis
    #         x_future = np.arange(
    #             len(actual_values),
    #             len(actual_values) + future_steps
    #         )

    #         # Actual Values
    #         ax.plot(
    #             x_existing,
    #             actual_values,
    #             label='Actual',
    #             linewidth=3
    #         )

    #         # Predicted Existing
    #         ax.plot(
    #             x_existing,
    #             predicted_values,
    #             label='Predicted Existing',
    #             linestyle='--',
    #             linewidth=2
    #         )

    #         # Future Predictions
    #         ax.plot(
    #             x_future,
    #             future_predictions,
    #             label='Future Prediction',
    #             linewidth=3
    #         )

    #         # Separator
    #         ax.axvline(
    #             x=len(actual_values)-1,
    #             linestyle=':',
    #             linewidth=2
    #         )

    #         ax.set_title(
    #             f'{name} Future Prediction - {target_name}'
    #         )

    #         ax.set_xlabel(
    #             'Samples / Future Cycles'
    #         )

    #         ax.set_ylabel(target_name)

    #         ax.legend()

    #         ax.grid(True)

    #         st.pyplot(fig)

    #     # ==========================================
    #     # RESULTS DATAFRAME
    #     # ==========================================

    #     results_df = pd.DataFrame(results)

    #     st.write(
    #         f"MODEL PERFORMANCE FOR {target_name}"
    #     )

    #     st.dataframe(results_df)

    #     return results_df


    # # ==========================================
    # # TRAINING BUTTON
    # # ==========================================

    # if st.button("Train All Models"):

    #     st.session_state["trained"] = True

    #     progress_bar = st.progress(0)

    #     status_text = st.empty()

    #     # ==========================================
    #     # STORE RESULTS
    #     # ==========================================

    #     all_results = {}

    #     target_datasets = {

    #         "OUTPUT POWER": (
    #             X_train_power,
    #             X_test_power,
    #             y_train_power,
    #             y_test_power
    #         ),

    #         "SFC": (
    #             X_train_sfc,
    #             X_test_sfc,
    #             y_train_sfc,
    #             y_test_sfc
    #         ),

    #         "EXHAUST TEMPERATURE": (
    #             X_train_eff,
    #             X_test_eff,
    #             y_train_eff,
    #             y_test_eff
    #         )
    #     }

    #     # ==========================================
    #     # TRAIN ALL TARGETS
    #     # ==========================================

    #     for idx, (target_name, data) in enumerate(target_datasets.items()):

    #         status_text.write(
    #             f"Training models for {target_name}..."
    #         )

    #         X_train, X_test, y_train, y_test = data

    #         target_results = {}

    #         for name, model in models.items():

    #             # ==========================================
    #             # ANN
    #             # ==========================================

    #             if name == "ANN":

    #                 X_train_tensor = torch.tensor(
    #                     X_train,
    #                     dtype=torch.float32
    #                 )

    #                 X_test_tensor = torch.tensor(
    #                     X_test,
    #                     dtype=torch.float32
    #                 )

    #                 y_train_tensor = torch.tensor(
    #                     y_train.values,
    #                     dtype=torch.float32
    #                 ).view(-1, 1)

    #                 train_dataset = TensorDataset(
    #                     X_train_tensor,
    #                     y_train_tensor
    #                 )

    #                 train_loader = DataLoader(
    #                     train_dataset,
    #                     batch_size=64,
    #                     shuffle=True
    #                 )

    #                 input_size = X_train.shape[1]

    #                 ann_model = ANNModel(input_size)

    #                 criterion = nn.MSELoss()

    #                 optimizer = optim.Adam(
    #                     ann_model.parameters(),
    #                     lr=0.001
    #                 )

    #                 epochs = 100

    #                 for epoch in range(epochs):

    #                     ann_model.train()

    #                     for batch_X, batch_y in train_loader:

    #                         predictions = ann_model(batch_X)

    #                         loss = criterion(
    #                             predictions,
    #                             batch_y
    #                         )

    #                         optimizer.zero_grad()

    #                         loss.backward()

    #                         optimizer.step()

    #                 ann_model.eval()

    #                 with torch.no_grad():

    #                     y_pred = ann_model(
    #                         X_test_tensor
    #                     ).numpy().flatten()

    #                 future_tensor = torch.tensor(
    #                     X_test[-30:],
    #                     dtype=torch.float32
    #                 )

    #                 with torch.no_grad():

    #                     future_predictions = ann_model(
    #                         future_tensor
    #                     ).numpy().flatten()

    #             # ==========================================
    #             # SKLEARN MODELS
    #             # ==========================================

    #             else:

    #                 model.fit(X_train, y_train)

    #                 y_pred = model.predict(X_test)

    #                 future_predictions = model.predict(
    #                     X_test[-30:]
    #                 )

    #             # ==========================================
    #             # METRICS
    #             # ==========================================

    #             mae = mean_absolute_error(
    #                 y_test,
    #                 y_pred
    #             )

    #             mse = mean_squared_error(
    #                 y_test,
    #                 y_pred
    #             )

    #             rmse = np.sqrt(mse)

    #             r2 = r2_score(
    #                 y_test,
    #                 y_pred
    #             )

    #             # ==========================================
    #             # SAVE MODEL RESULTS
    #             # ==========================================

    #             target_results[name] = {

    #                 "y_test": y_test,
    #                 "y_pred": y_pred,
    #                 "future_predictions": future_predictions,

    #                 "metrics": {

    #                     "MAE": round(mae, 3),
    #                     "RMSE": round(rmse, 3),
    #                     "R2 Score": round(r2, 3)

    #                 }
    #             }

    #         all_results[target_name] = target_results

    #         progress_bar.progress(
    #             (idx + 1) / len(target_datasets)
    #         )

    #     # ==========================================
    #     # SAVE TO SESSION
    #     # ==========================================

    #     st.session_state["results"] = all_results

    #     st.success(
    #         "All Models Trained Successfully"
    #     )

    # # ==========================================
    # # DISPLAY TRAINED RESULTS
    # # ==========================================

    # if "trained" in st.session_state:

    #     results_data = st.session_state["results"]

    #     # ==========================================
    #     # TARGET DROPDOWN
    #     # ==========================================

    #     selected_target = st.selectbox(

    #         "Choose Target Variable",

    #         list(results_data.keys())

    #     )

    #     # ==========================================
    #     # MODEL DROPDOWN
    #     # ==========================================

    #     selected_model = st.selectbox(

    #         "Choose Model",

    #         list(results_data[selected_target].keys())

    #     )

    #     # ==========================================
    #     # GET MODEL DATA
    #     # ==========================================

    #     model_data = results_data[
    #         selected_target
    #     ][
    #         selected_model
    #     ]

    #     metrics = model_data["metrics"]

    #     y_test = model_data["y_test"]

    #     y_pred = model_data["y_pred"]

    #     future_predictions = model_data[
    #         "future_predictions"
    #     ]

    #     # ==========================================
    #     # METRICS TABLE
    #     # ==========================================

    #     st.subheader(
    #         f"{selected_model} Metrics"
    #     )

    #     metrics_df = pd.DataFrame(
    #         [metrics]
    #     )

    #     st.dataframe(metrics_df)

    #     # ==========================================
    #     # ACTUAL VS PREDICTED
    #     # ==========================================

    #     fig, ax = plt.subplots(
    #         figsize=(12, 6)
    #     )

    #     sns.set_style("darkgrid")

    #     ax.plot(
    #         y_pred[:100],
    #         label="Predicted"
    #     )

    #     ax.plot(
    #         y_test.values[:100],
    #         label="Actual",
    #         linewidth=2
    #     )

    #     ax.set_title(
    #         f"{selected_model} - {selected_target}"
    #     )

    #     ax.set_xlabel("Samples")

    #     ax.set_ylabel(selected_target)

    #     ax.legend()

    #     ax.grid(True)

    #     st.pyplot(fig)

    #     # ==========================================
    #     # FUTURE PREDICTION
    #     # ==========================================

    #     fig, ax = plt.subplots(
    #         figsize=(14, 6)
    #     )

    #     actual_values = y_test.values[:100]

    #     predicted_values = y_pred[:100]

    #     x_existing = np.arange(
    #         len(actual_values)
    #     )

    #     x_future = np.arange(

    #         len(actual_values),

    #         len(actual_values) + 30

    #     )

    #     ax.plot(
    #         x_existing,
    #         actual_values,
    #         label='Actual',
    #         linewidth=3
    #     )

    #     ax.plot(
    #         x_existing,
    #         predicted_values,
    #         linestyle='--',
    #         linewidth=2,
    #         label='Predicted Existing'
    #     )

    #     ax.plot(
    #         x_future,
    #         future_predictions,
    #         linewidth=3,
    #         label='Future Prediction'
    #     )

    #     ax.axvline(
    #         x=len(actual_values)-1,
    #         linestyle=':',
    #         linewidth=2
    #     )

    #     ax.set_title(
    #         f"{selected_model} Future Prediction"
    #     )

    #     ax.set_xlabel(
    #         'Samples / Future Cycles'
    #     )

    #     ax.set_ylabel(
    #         selected_target
    #     )

    #     ax.legend()

    #     ax.grid(True)

    #     st.pyplot(fig)
    # # ==========================================
    # # OUTPUT POWER MODELS
    # # ==========================================

    # results_power = evaluate_models(

    #     X_train_power,
    #     X_test_power,
    #     y_train_power,
    #     y_test_power,
    #     "OUTPUT POWER"

    # )

    # # ==========================================
    # # SFC MODELS
    # # ==========================================

    # results_sfc = evaluate_models(

    #     X_train_sfc,
    #     X_test_sfc,
    #     y_train_sfc,
    #     y_test_sfc,
    #     "SFC"

    # )

    # # ==========================================
    # # EFFICIENCY MODELS
    # # ==========================================

    # results_efficiency = evaluate_models(

    #     X_train_eff,
    #     X_test_eff,
    #     y_train_eff,
    #     y_test_eff,
    #     "EXHAUST TEMPERATURE"

    # )

    # # ==========================================
    # # MODEL COMPARISON
    # # ==========================================

    # def compare_models(results_df, title):

    #     model_names = results_df['Model']

    #     r2_scores = results_df['R2 Score']

    #     rmse_scores = results_df['RMSE']

    #     # ==========================================
    #     # R2 SCORE
    #     # ==========================================

    #     sns.set_style("darkgrid")

    #     fig, ax = plt.subplots(figsize=(10, 5))

    #     ax.bar(
    #         model_names,
    #         r2_scores
    #     )

    #     ax.set_title(f"{title} - R2 Score")

    #     ax.set_xlabel("Models")

    #     ax.set_ylabel("R2 Score")

    #     ax.grid(True)

    #     st.pyplot(fig)

    #     # ==========================================
    #     # RMSE
    #     # ==========================================

    #     fig, ax = plt.subplots(figsize=(10, 5))

    #     ax.bar(
    #         model_names,
    #         rmse_scores
    #     )

    #     ax.set_title(f"{title} - RMSE")

    #     ax.set_xlabel("Models")

    #     ax.set_ylabel("RMSE")

    #     ax.grid(True)

    #     st.pyplot(fig)


    # # ==========================================
    # # COMPARE ALL TARGETS
    # # ==========================================

    # compare_models(
    #     results_power,
    #     "OUTPUT POWER"
    # )

    # compare_models(
    #     results_sfc,
    #     "SFC"
    # )

    # compare_models(
    #     results_efficiency,
    #     "EXHAUST TEMPERATURE"
    # )

    # # ==========================================
    # # TRAINING BUTTON
    # # ==========================================

    # if st.button("Train All Models"):

    #     progress_bar = st.progress(0)

    #     status_text = st.empty()

    #     for i, model_name in enumerate(models.keys()):

    #         status_text.write(
    #             f"Training {model_name}..."
    #         )

    #         progress_bar.progress(
    #             (i + 1) / len(models)
    #         )

    #     st.success(
    #         "All Models Initialized Successfully"
    #     )
# elif section == "ML Models":

#     st.title("Machine Learning Models")

#     # ==========================================
#     # FEATURE SELECTION
#     # ==========================================

#     st.subheader("Select Features & Targets")

#     numerical_cols = df.select_dtypes(
#         include=['int64', 'float64']
#     ).columns.tolist()

#     # Select Features
#     features = st.multiselect(
#         "Select Feature Columns",
#         numerical_cols,
#         default=numerical_cols[:5]
#     )

#     # Select Targets
#     target_power = st.selectbox(
#         "Select Output Power Target",
#         numerical_cols
#     )

#     target_sfc = st.selectbox(
#         "Select SFC Target",
#         numerical_cols,
#         index=1
#     )

#     target_efficiency = st.selectbox(
#         "Select Efficiency Target",
#         numerical_cols,
#         index=2
#     )

#     # ==========================================
#     # DATA PREPARATION
#     # ==========================================

#     if len(features) > 0:

#         # Features
#         X = df[features]

#         # Targets
#         y_power = df[target_power]
#         y_sfc = df[target_sfc]
#         y_efficiency = df[target_efficiency]

#         # ==========================================
#         # FEATURE SCALING
#         # ==========================================

#         scaler = StandardScaler()

#         X_scaled = scaler.fit_transform(X)

#         st.success("Feature Scaling Completed")

#         # ==========================================
#         # TRAIN TEST SPLITS
#         # ==========================================

#         X_train_power, X_test_power, y_train_power, y_test_power = train_test_split(
#             X_scaled,
#             y_power,
#             test_size=0.20,
#             random_state=400
#         )

#         X_train_sfc, X_test_sfc, y_train_sfc, y_test_sfc = train_test_split(
#             X_scaled,
#             y_sfc,
#             test_size=0.20,
#             random_state=42
#         )

#         X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(
#             X_scaled,
#             y_efficiency,
#             test_size=0.25,
#             random_state=42
#         )

#         # ==========================================
#         # DISPLAY SHAPES
#         # ==========================================

#         st.subheader("Dataset Shapes")

#         st.write("OUTPUT POWER TRAINING SHAPE")
#         st.write(X_train_power.shape, y_train_power.shape)

#         st.write("OUTPUT POWER TESTING SHAPE")
#         st.write(X_test_power.shape, y_test_power.shape)

#         st.write("SFC TRAINING SHAPE")
#         st.write(X_train_sfc.shape, y_train_sfc.shape)

#         st.write("EFFICIENCY TRAINING SHAPE")
#         st.write(X_train_eff.shape, y_train_eff.shape)

#         # ==========================================
#         # ANN MODEL
#         # ==========================================

#         class ANNModel(nn.Module):

#             def __init__(self, input_size):

#                 super(ANNModel, self).__init__()

#                 self.network = nn.Sequential(

#                     nn.Linear(input_size, 512),
#                     nn.BatchNorm1d(512),
#                     nn.ReLU(),
#                     nn.Dropout(0.3),

#                     nn.Linear(512, 256),
#                     nn.BatchNorm1d(256),
#                     nn.ReLU(),
#                     nn.Dropout(0.3),

#                     nn.Linear(256, 128),
#                     nn.ReLU(),

#                     nn.Linear(128, 64),
#                     nn.ReLU(),

#                     nn.Linear(64, 32),
#                     nn.ReLU(),

#                     nn.Linear(32, 1)

#                 )

#             def forward(self, x):

#                 return self.network(x)

#         # ==========================================
#         # MODELS
#         # ==========================================

#         models = {

#             "ANN": "PyTorch_ANN",

#             "Random Forest": RandomForestRegressor(
#                 n_estimators=300,
#                 random_state=42
#             ),

#             "Decision Tree": DecisionTreeRegressor(
#                 random_state=42,
#                 criterion='squared_error'
#             ),

#             "SVM": SVR(
#                 kernel='rbf'
#             ),

#             "XGBoost": XGBRegressor(
#                 n_estimators=400,
#                 random_state=42
#             )
#         }

#         # ==========================================
#         # EVALUATION FUNCTION
#         # ==========================================

#         def evaluate_models(
#             X_train,
#             X_test,
#             y_train,
#             y_test,
#             target_name
#         ):

#             results = []

#             progress_bar = st.progress(0)

#             for i, (name, model) in enumerate(models.items()):

#                 st.subheader(f"{name} - {target_name}")

#                 # ==========================================
#                 # PYTORCH ANN
#                 # ==========================================

#                 if name == "ANN":

#                     X_train_tensor = torch.tensor(
#                         X_train,
#                         dtype=torch.float32
#                     )

#                     X_test_tensor = torch.tensor(
#                         X_test,
#                         dtype=torch.float32
#                     )

#                     y_train_tensor = torch.tensor(
#                         y_train.values,
#                         dtype=torch.float32
#                     ).view(-1, 1)

#                     train_dataset = TensorDataset(
#                         X_train_tensor,
#                         y_train_tensor
#                     )

#                     train_loader = DataLoader(
#                         train_dataset,
#                         batch_size=64,
#                         shuffle=True
#                     )

#                     input_size = X_train.shape[1]

#                     ann_model = ANNModel(input_size)

#                     criterion = nn.MSELoss()

#                     optimizer = optim.Adam(
#                         ann_model.parameters(),
#                         lr=0.001
#                     )

#                     epochs = st.slider(
#                         f"Epochs for {target_name}",
#                         10,
#                         1000,
#                         100
#                     )

#                     losses = []

#                     loss_placeholder = st.empty()

#                     for epoch in range(epochs):

#                         ann_model.train()

#                         epoch_loss = 0

#                         for batch_X, batch_y in train_loader:

#                             predictions = ann_model(batch_X)

#                             loss = criterion(
#                                 predictions,
#                                 batch_y
#                             )

#                             optimizer.zero_grad()

#                             loss.backward()

#                             optimizer.step()

#                             epoch_loss += loss.item()

#                         avg_loss = epoch_loss / len(train_loader)

#                         losses.append(avg_loss)

#                         loss_placeholder.write(
#                             f"Epoch [{epoch+1}/{epochs}] "
#                             f"Loss: {avg_loss:.6f}"
#                         )

#                     # Prediction
#                     ann_model.eval()

#                     with torch.no_grad():

#                         y_pred = ann_model(
#                             X_test_tensor
#                         ).numpy().flatten()

#                     # Loss Plot
#                     fig, ax = plt.subplots(figsize=(10, 5))

#                     ax.plot(losses)

#                     ax.set_title(
#                         f"{target_name} ANN Training Loss"
#                     )

#                     ax.set_xlabel("Epoch")

#                     ax.set_ylabel("Loss")

#                     st.pyplot(fig)

#                 # ==========================================
#                 # SCIKIT-LEARN MODELS
#                 # ==========================================

#                 else:

#                     model.fit(X_train, y_train)

#                     y_pred = model.predict(X_test)

#                 # ==========================================
#                 # METRICS
#                 # ==========================================

#                 mae = mean_absolute_error(
#                     y_test,
#                     y_pred
#                 )

#                 mse = mean_squared_error(
#                     y_test,
#                     y_pred
#                 )

#                 rmse = np.sqrt(mse)

#                 r2 = r2_score(
#                     y_test,
#                     y_pred
#                 )

#                 results.append({
#                     "Model": name,
#                     "MAE": mae,
#                     "RMSE": rmse,
#                     "R2 Score": r2
#                 })

#                 # ==========================================
#                 # DISPLAY METRICS
#                 # ==========================================

#                 st.write(f"MAE: {mae:.4f}")
#                 st.write(f"RMSE: {rmse:.4f}")
#                 st.write(f"R2 Score: {r2:.4f}")

#                 # ==========================================
#                 # ACTUAL VS PREDICTED
#                 # ==========================================

#                 fig, ax = plt.subplots(figsize=(8, 5))

#                 ax.scatter(
#                     y_test,
#                     y_pred,
#                     alpha=0.7
#                 )

#                 ax.set_xlabel("Actual")

#                 ax.set_ylabel("Predicted")

#                 ax.set_title(
#                     f"{name} - Actual vs Predicted"
#                 )

#                 st.pyplot(fig)

#                 progress_bar.progress(
#                     (i + 1) / len(models)
#                 )

#             return pd.DataFrame(results)

#         # ==========================================
#         # RUN MODELS
#         # ==========================================

#         if st.button("Train Models"):

#             st.subheader("OUTPUT POWER RESULTS")

#             results_power = evaluate_models(
#                 X_train_power,
#                 X_test_power,
#                 y_train_power,
#                 y_test_power,
#                 "Output Power"
#             )

#             st.dataframe(results_power)

#             st.subheader("SFC RESULTS")

#             results_sfc = evaluate_models(
#                 X_train_sfc,
#                 X_test_sfc,
#                 y_train_sfc,
#                 y_test_sfc,
#                 "SFC"
#             )

#             st.dataframe(results_sfc)

#             st.subheader("EFFICIENCY RESULTS")

#             results_eff = evaluate_models(
#                 X_train_eff,
#                 X_test_eff,
#                 y_train_eff,
#                 y_test_eff,
#                 "Efficiency"
#             )

#            st.dataframe(results_eff)

# elif section == "Deep Learning":
#     st.markdown()

# elif section == "Evaluation":
    # ==========================================
    # EVALUATION FUNCTION
    # ==========================================

    

# elif section == "Prediction":
#     st.markdown()

# elif section == "Contact":
#     st.markdown()


# /* ==========================================
# CUSTOM HEADER DESIGN
# ========================================== */

# h1, h2, h3, h4, h5, h6 {

#     color: black !important;

#     text-align: center !important;

#     font-family: "Poppins", sans-serif;

#     letter-spacing: 1px;

# }

# h1 {

#     font-size: 42px !important;

#     font-weight: 800 !important;

# }

# h2 {

#     font-size: 32px !important;

#     font-weight: 700 !important;

# }

# h3 {

#     font-size: 26px !important;

#     font-weight: 600 !important;

# }

# h1::after,
# h2::after {

#     content: "";

#     display: block;

#     width: 80px;

#     height: 4px;

#     background: black;

#     margin: 10px auto;

#     border-radius: 10px;

# }
