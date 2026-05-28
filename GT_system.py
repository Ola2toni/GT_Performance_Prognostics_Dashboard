
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.inspection import permutation_importance

# import pickle
# import os
# Load Excel file
file_path = "Gas_Turbine_Dataset.xlsx"

# Read the combined dataset sheet
df = pd.read_excel(file_path, sheet_name="combined_data")

def display_dataset_info(df):
    print("Dataset Loaded:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)

display_dataset_info(df)


def handle_outliers(df):
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    # Fill numerical columns with mean
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    print("\nMissing Values After Cleaning:\n")
    print(df.isnull().sum())
    
    
    print("\nCleaned Dataset:")
    print(df.head())
    # for col in df:
    #     sns.set_style("darkgrid")
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(df[col], kde=True)
    #     plt.title(f'Distribution of {col}')
    #     plt.show()

    # for col in df:
    #     sns.set_style("darkgrid")
    #     plt.figure(figsize=(5, 4))
    #     sns.boxplot(data=df[col])
    #     plt.xticks(rotation=45)
    #     plt.title(f"Boxplot of {col}")
    #     plt.show()
        

    for col in numerical_cols:

        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        
        df[col] = np.clip(
        df[col],
        lower,
        upper
    )
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    print("IQR outlier reduction applied to all features\n")


    for col in numerical_cols:
        if len(df[col]) > 11:
            df[col] = savgol_filter(
                df[col], window_length=11, polyorder=2
                )
    print("Savitzky-Golay smoothing applied.")

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    df_scaled = pd.DataFrame(
    scaled_data,
    columns=numerical_cols
    )
    print("Robust scaling completed.")
    
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(df_scaled)
    print("PCA completed.")
    print("Original Shape:", df_scaled.shape)
    print("Reduced Shape:", X_pca.shape)
    
    print("\nFinal Cleaned Dataset Shape:")
    print(df_scaled.shape)
    print("\nFirst 5 Rows:")
    print(df_scaled.head())


handle_outliers(df)


# Outlier Cleaned Dataset
# for col in df:
#     sns.set_style("darkgrid")
#     plt.figure(figsize=(5, 4))
#     sns.boxplot(data=df[col])
#     plt.xticks(rotation=45)
#     plt.title(f"Boxplot of {col}")
#     plt.show()


# df["SFC"] = df["FUEL GAS FLOW"] / df["OUTPUT POWER"]
# df["EFFICIENCY"] = (df["OUTPUT POWER"] / (df["FUEL GAS FLOW"] * 42)) * 100

print(df[["OUTPUT POWER", "FUEL GAS FLOW", "EXHAUST TEMPERATURE"]])

# Determine the Features and Target Variables
features = df.drop(columns=['OUTPUT POWER', 'FUEL GAS FLOW', 'GT UNIT', 'EXHAUST TEMPERATURE', 
                            """'FUEL GAS PRESSURE2', 'ALLOWABLE SPREAD', 'LUBE OIL HEADER PRESSURE', 'TORQUE ANGLE'"""], errors='ignore').columns
                            #'EXHAUST SPREAD 1',  'LUBE OIL HEADER TEMPERATURE', 
                            #'COMPRESSOR INLET PRESSURE', 'GT SPEED', 'COMPRESSOR DISCHARGE PRESSURE', 'REACTIVE POWER', 'EXHAUST SPREAD 2', 'DEW POINT', 'COALESCING FILTER OUTLET TEMPERATURE',
                            #'COALESCING FILTER OUTLET PRESSURE', 'FUEL GAS PRESSURE1', 'SCRUBBER INTLET PRESSURE', 'IGV POSITION', 'MAXIMUM VIBRATION', 'FUEL GAS TEMPERATURE'], errors='ignore').columns
# ==========================================
# SAVE FEATURES
# ==========================================

# pickle.dump(
#     features,
#     open("saved_models/features.pkl", "wb")
# )

target_power = df['OUTPUT POWER']
target_sfc = df['FUEL GAS FLOW']
target_efficiency = df['EXHAUST TEMPERATURE']

# def new_plot(features, target):
#     for col in df[features].columns:
#         plt.figure(figsize=(8, 6))
#         plt.scatter(df[col], target, color='blue', marker='o', linestyle='-', alpha=0.7)
#         plt.title(f"{target.name} vs {df[col].name}")
#         plt.xlabel(df[col].name)
#         plt.ylabel(target.name)
#         plt.show()


# # Plot all targets
# new_plot(features, target_power)
# new_plot(features, target_sfc)
# new_plot(features, target_efficiency)


# Select only numerical columns
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Create correlation matrix
corr_matrix = numerical_df.corr()

# Plot heatmap
plt.figure(figsize=(15, 10))

sns.heatmap(
    corr_matrix,
    annot=True,          # show correlation values
    cmap='coolwarm',     # color style
    fmt=".2f",           # decimal format
    linewidths=0.5
)

plt.title("Correlation Heatmap of Gas Turbine Dataset")

plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure(figsize=(10, 8))

# Add 3D axis
ax = fig.add_subplot(111, projection='3d')

# Plot
ax.scatter(
    df['FUEL GAS FLOW'],     # X-axis
    df['EXHAUST TEMPERATURE'],  # Y-axis
    df['OUTPUT POWER'],      # Z-axis
    alpha=0.7
)

# Labels
ax.set_xlabel('FUEL GAS FLOW')
ax.set_ylabel('EXHAUST TEMPERATURE')
ax.set_zlabel('OUTPUT POWER')

# Title
ax.set_title('3D Visualization of Gas Turbine Performance')

plt.show()



# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(
    df['FUEL GAS FLOW'],
    df['EXHAUST TEMPERATURE'],
    df['OUTPUT POWER'],
    c=df['OUTPUT POWER'],
    alpha=0.7
)

# Labels
ax.set_xlabel('FUEL GAS FLOW')
ax.set_ylabel('EXHAUST TEMPERATURE')
ax.set_zlabel('OUTPUT POWER')

# Title
ax.set_title('Animated 3D Gas Turbine Visualization')

# Animation function
def rotate(angle):
    ax.view_init(elev=30, azim=angle)

# Animate
ani = FuncAnimation(
    fig,
    rotate,
    frames=360,
    interval=50
)

plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df['EXHAUST TEMPERATURE'],
    df['COMPRESSOR DISCHARGE TEMPERATURE'],
    df['OUTPUT POWER'],
    c=df['OUTPUT POWER'],
    alpha=0.7
)

ax.set_xlabel('EXHAUST TEMPERATURE')
ax.set_ylabel('COMPRESSOR DISCHARGE TEMPERATURE')
ax.set_zlabel('OUTPUT POWER')

ax.set_title('Turbine Performance Space')

plt.show()


# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(
#     df['FUEL GAS FLOW'],
#     df['OUTPUT POWER'],
#     df['EFFICIENCY'],
#     c=df['EFFICIENCY'],
#     alpha=0.7
# )

# ax.set_xlabel('FUEL GAS FLOW')
# ax.set_ylabel('OUTPUT POWER')
# ax.set_zlabel('EFFICIENCY')

# # Animation function
# def rotate(angle):
#     ax.view_init(elev=30, azim=angle)

# # Animate
# ani2 = FuncAnimation(
#     fig,
#     rotate,
#     frames=360,
#     interval=50
# )

# ax.set_title('Energy Conversion Space')

# plt.show()


# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(
#     df['FUEL GAS FLOW'],
#     df['EXHAUST TEMPERATURE'],
#     df['SFC'],
#     c=df['SFC'],
#     alpha=0.7
# )

# ax.set_xlabel('FUEL GAS FLOW')
# ax.set_ylabel('EXHAUST TEMPERATURE')
# ax.set_zlabel('SFC')

# ax.set_title('Animated 3D Turbine Visualization')

# def rotate(angle):
#     ax.view_init(elev=30, azim=angle)

# ani2 = FuncAnimation(
#     fig,
#     rotate,
#     frames=360,
#     interval=50
# )

# plt.show()



# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# surf = ax.plot_trisurf(
#     df['FUEL GAS FLOW'],
#     df['EXHAUST TEMPERATURE'],
#     df['OUTPUT POWER'],
#     cmap='viridis',
#     edgecolor='none'
# )

# ax.set_xlabel('FUEL GAS FLOW')
# ax.set_ylabel('EXHAUST TEMPERATURE')
# ax.set_zlabel('OUTPUT POWER')

# ax.set_title('3D Surface Plot')

# plt.show()



from sklearn.model_selection import train_test_split

# ==========================================
# FEATURES
# ==========================================

X = df[features]

# ==========================================
# TARGET VARIABLES
# ==========================================

y_power = target_power
y_sfc = target_sfc
y_efficiency = target_efficiency

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# SAVE SCALER
# ==========================================

# import pickle
# import os

# os.makedirs("saved_models", exist_ok=True)

# pickle.dump(
#     scaler,
#     open("saved_models/scaler.pkl", "wb")
# )

# ==========================================
# SPLIT FOR OUTPUT POWER
# ==========================================


X_train_power, X_test_power, y_train_power, y_test_power = train_test_split(
    X_scaled,
    y_power,
    test_size=0.20,
    random_state=400
)

# ==========================================
# SPLIT FOR SFC
# ==========================================

X_train_sfc, X_test_sfc, y_train_sfc, y_test_sfc = train_test_split(
    X_scaled,
    y_sfc,
    test_size=0.20,
    random_state=42
)

# ==========================================
# SPLIT FOR EFFICIENCY
# ==========================================

X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(
    X_scaled,
    y_efficiency,
    test_size=0.25,
    random_state=42
)

# ==========================================
# SAVE TEST DATA
# ==========================================

import pickle
import os

os.makedirs("saved_models", exist_ok=True)

# OUTPUT POWER
pickle.dump(
    X_test_power,
    open(
        "saved_models/X_test_power.pkl",
        "wb"
    )
)

pickle.dump(
    y_test_power,
    open(
        "saved_models/y_test_power.pkl",
        "wb"
    )
)

# FUEL GAS FLOW
pickle.dump(
    X_test_sfc,
    open(
        "saved_models/X_test_sfc.pkl",
        "wb"
    )
)

pickle.dump(
    y_test_sfc,
    open(
        "saved_models/y_test_sfc.pkl",
        "wb"
    )
)

# EXHAUST TEMPERATURE
pickle.dump(
    X_test_eff,
    open(
        "saved_models/X_test_eff.pkl",
        "wb"
    )
)

pickle.dump(
    y_test_eff,
    open(
        "saved_models/y_test_eff.pkl",
        "wb"
    )
)

# ==========================================
# DISPLAY SHAPES
# ==========================================

# print("OUTPUT POWER TRAINING SHAPE:")
# print(X_train_power.shape, y_train_power.shape)

# print("\nOUTPUT POWER TESTING SHAPE:")
# print(X_test_power.shape, y_test_power.shape)

# print("\nSFC TRAINING SHAPE:")
# print(X_train_sfc.shape, y_train_sfc.shape)

# print("\nEFFICIENCY TRAINING SHAPE:")
# print(X_train_eff.shape, y_train_eff.shape)

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score
# )



# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# DEFINE MODELS
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
    
models = {

    # "ANN": MLPRegressor(
    #     learning_rate=0.001,
    #     max_iter=500,
    #     random_state=70
    # ),
    "ANN": "PyTorch_ANN",

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42, criterion='squared_error'
    ),

    "SVM": SVR(
        kernel='rbf'
    ),

    "XGBoost": XGBRegressor(
        n_estimators=400,
        random_state=42
    )
}


# # ==========================================
# FUNCTION FOR TRAINING & EVALUATION
# ==========================================

def evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    target_name
):

    results = []

    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")

    for name, model in models.items():

        # ==========================================
        # PYTORCH ANN
        # ==========================================

        if name == "ANN":

            # Convert to tensors
            X_train_tensor = torch.tensor(
                X_train,
                dtype=torch.float32
            )

            X_test_tensor = torch.tensor(
                X_test,
                dtype=torch.float32
            )

            y_train_tensor = torch.tensor(
                y_train.values,
                dtype=torch.float32
            ).view(-1, 1)

            # DataLoader
            train_dataset = TensorDataset(
                X_train_tensor,
                y_train_tensor
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=True
            )

            # Initialize ANN
            input_size = X_train.shape[1]

            ann_model = ANNModel(input_size)

            criterion = nn.MSELoss()

            optimizer = optim.Adam(
                ann_model.parameters(),
                lr=0.001
            )

            # Training Loop
            epochs = 1000

            losses = []

            for epoch in range(epochs):

                ann_model.train()

                epoch_loss = 0

                for batch_X, batch_y in train_loader:

                    predictions = ann_model(batch_X)

                    loss = criterion(
                        predictions,
                        batch_y
                    )

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)

                losses.append(avg_loss)

                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Loss: {avg_loss:.6f}"
                    )

            # Prediction
            ann_model.eval()

            with torch.no_grad():

                y_pred = ann_model(
                    X_test_tensor
                ).numpy().flatten()
            
            # # ==========================================
            # # SAVE ANN MODEL
            # # ==========================================

            # torch.save(

            #     ann_model.state_dict(),

            #     f"saved_models/{target_name}_ANN.pth"
            # )

        # ==========================================
        # SCIKIT-LEARN MODELS
        # ==========================================

        else:

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # # ==========================================
            # # SAVE SKLEARN MODEL
            # # ==========================================

            # pickle.dump(

            #     model,

            #     open(

            #         f"saved_models/{target_name}_{name}.pkl",

            #         "wb"
            #     )
            # )

        # ==========================================
        # METRICS
        # ==========================================

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred
        )

        # # ==========================================
        # # PERMUTATION IMPORTANCE
        # # ==========================================

        # if name != "ANN":

        #     permutation_importance_plot(

        #         trained_model=model,

        #         X_test=X_test,

        #         y_test=y_test,

        #         feature_names=X.columns,

        #         model_name=name,

        #         target_name=target_name
        #     )

        # Save Results
        results.append({
            'Model': name,
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'R2 Score': round(r2, 3)
        })

        # Plot
        sns.set_style("darkgrid")
        plt.plot(
            y_pred[:100],
            label="predicted"
        )

        # Actual Values
        plt.plot(
            y_test.values[:100],
            label='Actual',
            linewidth=2
        )

        plt.title(
            f'{name} (Actual vs Predicted) - {target_name}'
        )

        plt.xlabel('Samples')

        plt.ylabel(target_name)

        plt.legend()

        plt.grid(True)

        plt.show()

       # ==========================================
        # FUTURE PREDICTION VISUALIZATION
        # ==========================================

        sns.set_style("darkgrid")

        plt.figure(figsize=(14, 6))

        # ------------------------------------------
        # EXISTING TEST DATA
        # ------------------------------------------

        actual_values = y_test.values[:100]

        predicted_values = y_pred[:100]

        x_existing = np.arange(len(actual_values))

        # ------------------------------------------
        # FUTURE DATA
        # ------------------------------------------

        future_steps = 30

        # Last samples from test data
        last_data = X_test[-future_steps:]

        # ------------------------------------------
        # FUTURE PREDICTIONS
        # ------------------------------------------

        if name == "ANN":

            future_tensor = torch.tensor(
                last_data,
                dtype=torch.float32
            )

            ann_model.eval()

            with torch.no_grad():

                future_predictions = ann_model(
                    future_tensor
                ).numpy().flatten()

        else:

            future_predictions = model.predict(last_data)

        # ------------------------------------------
        # FUTURE X-AXIS
        # ------------------------------------------

        x_future = np.arange(
            len(actual_values),
            len(actual_values) + future_steps
        )

        # ------------------------------------------
        # PLOTS
        # ------------------------------------------

        # Actual Values
        plt.plot(
            x_existing,
            actual_values,
            label='Actual',
            linewidth=3
        )

        # Predicted Existing
        plt.plot(
            x_existing,
            predicted_values,
            label='Predicted Existing',
            linestyle='--',
            linewidth=2
        )

        # Future Predictions
        plt.plot(
            x_future,
            future_predictions,
            label='Future Prediction',
            linewidth=3
        )

        # Separator Line
        plt.axvline(
            x=len(actual_values)-1,
            linestyle=':',
            linewidth=2
        )

        # Labels
        plt.title(
            f'{name} Future Prediction - {target_name}'
        )

        plt.xlabel('Samples / Future Cycles')

        plt.ylabel(target_name)

        plt.legend()

        plt.grid(True)

        plt.show()

    # for col in models:
    #     # Plot
    #     sns.set_style("darkgrid")
    #     plt.plot(
    #         y_pred[:100],
    #         label=name
    #     )

    #     # Actual Values
    #     plt.plot(
    #         y_test.values[:100],
    #         label='Actual',
    #         linewidth=3
    #     )

    #     plt.title(
    #         f'Actual vs Predicted - {target_name}'
    #     )

    #     plt.xlabel('Samples')

    #     plt.ylabel(target_name)

    #     plt.legend()

    #     plt.grid(True)

    #     plt.show()

    # Results DataFrame
    results_df = pd.DataFrame(results)

    # # ==========================================
    # # SAVE METRICS
    # # ==========================================

    # results_df.to_csv(

    #     f"saved_models/{target_name}_metrics.csv",

    #     index=False
    # )

    print(f"\nMODEL PERFORMANCE FOR {target_name}")

    print(results_df)

    return results_df


# ==========================================
# OUTPUT POWER MODELS
# ==========================================

results_power = evaluate_models(
    X_train_power,
    X_test_power,
    y_train_power,
    y_test_power,
    "OUTPUT POWER"
)

# ==========================================
# SFC MODELS
# ==========================================

results_sfc = evaluate_models(
    X_train_sfc,
    X_test_sfc,
    y_train_sfc,
    y_test_sfc,
    "SFC"
)

# ==========================================
# EFFICIENCY MODELS
# ==========================================

results_efficiency = evaluate_models(
    X_train_eff,
    X_test_eff,
    y_train_eff,
    y_test_eff,
    "EXHAUST TEMPERATURE"
)

# ==========================================
# VISUALIZE MODEL COMPARISON
# ==========================================

def compare_models(results_df, title):

    model_names = results_df['Model']

    r2_scores = results_df['R2 Score']

    rmse_scores = results_df['RMSE']

    # R2 SCORE
    plt.figure(figsize=(10, 5))
    sns.set_style("darkgrid")

    plt.bar(model_names, r2_scores)

    plt.title(f"{title} - R2 Score")

    plt.xlabel("Models")

    plt.ylabel("R2 Score")

    plt.grid(True)

    plt.show()

    # RMSE
    plt.figure(figsize=(10, 5))

    plt.bar(model_names, rmse_scores)

    plt.title(f"{title} - RMSE")

    plt.xlabel("Models")

    plt.ylabel("RMSE")

    plt.grid(True)

    plt.show()

# ==========================================
# COMPARE ALL TARGETS
# ==========================================

compare_models(results_power, "OUTPUT POWER")

compare_models(results_sfc, "SFC")

compare_models(results_efficiency, "EXHAUST TEMPERATURE")



# ==========================================
# TRAIN BEST MODEL FUNCTION
# ==========================================

def train_best_model(
    results_df,
    X_train,
    y_train
):

    # --------------------------------------
    # BEST MODEL NAME
    # --------------------------------------

    best_model_name = results_df.loc[
        results_df['R2 Score'].idxmax(),
        'Model'
    ]

    print(f"\nBest Model: {best_model_name}")

    # --------------------------------------
    # GET MODEL
    # --------------------------------------

    if best_model_name == "ANN":

        input_size = X_train.shape[1]

        best_model = ANNModel(input_size)

        criterion = nn.MSELoss()

        optimizer = optim.Adam(
            best_model.parameters(),
            lr=0.001
        )

        # Convert to tensors
        X_tensor = torch.tensor(
            X_train,
            dtype=torch.float32
        )

        y_tensor = torch.tensor(
            y_train.values,
            dtype=torch.float32
        ).view(-1,1)

        dataset = TensorDataset(
            X_tensor,
            y_tensor
        )

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True
        )

        # Train ANN
        epochs = 1000

        for epoch in range(epochs):

            best_model.train()

            for batch_X, batch_y in loader:

                pred = best_model(batch_X)

                loss = criterion(
                    pred,
                    batch_y
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

    else:

        best_model = models[best_model_name]

        best_model.fit(
            X_train,
            y_train
        )

    return best_model, best_model_name

# ==========================================
# BEST MODELS
# ==========================================

best_power_model, best_power_name = train_best_model(

    results_power,

    X_train_power,

    y_train_power
)

best_sfc_model, best_sfc_name = train_best_model(

    results_sfc,

    X_train_sfc,

    y_train_sfc
)

best_eff_model, best_eff_name = train_best_model(

    results_efficiency,

    X_train_eff,

    y_train_eff
)

# ==========================================
# USER INPUT FEATURES
# ==========================================

print("\nENTER FEATURE VALUES")

user_inputs = []

for feature in features:

    value = float(
        input(f"Enter value for {feature}: ")
    )

    user_inputs.append(value)

# Convert to array
new_data = np.array([user_inputs])

# Scale input
new_data_scaled = scaler.transform(new_data)

# ==========================================
# OUTPUT POWER PREDICTION
# ==========================================

if best_power_name == "ANN":

    tensor_data = torch.tensor(
        new_data_scaled,
        dtype=torch.float32
    )

    best_power_model.eval()

    with torch.no_grad():

        power_prediction = best_power_model(
            tensor_data
        ).numpy().flatten()[0]

else:

    power_prediction = best_power_model.predict(
        new_data_scaled
    )[0]

# ==========================================
# FUEL GAS FLOW PREDICTION
# ==========================================

if best_sfc_name == "ANN":

    tensor_data = torch.tensor(
        new_data_scaled,
        dtype=torch.float32
    )

    best_sfc_model.eval()

    with torch.no_grad():

        sfc_prediction = best_sfc_model(
            tensor_data
        ).numpy().flatten()[0]

else:

    sfc_prediction = best_sfc_model.predict(
        new_data_scaled
    )[0]

# ==========================================
# EXHAUST TEMPERATURE PREDICTION
# ==========================================

if best_eff_name == "ANN":

    tensor_data = torch.tensor(
        new_data_scaled,
        dtype=torch.float32
    )

    best_eff_model.eval()

    with torch.no_grad():

        eff_prediction = best_eff_model(
            tensor_data
        ).numpy().flatten()[0]

else:

    eff_prediction = best_eff_model.predict(
        new_data_scaled
    )[0]

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nPREDICTIONS USING BEST MODELS")

print(f"\nBest OUTPUT POWER Model: {best_power_name}")

print(f"Predicted OUTPUT POWER: {power_prediction:.3f}")

print(f"\nBest FUEL GAS FLOW Model: {best_sfc_name}")

print(f"Predicted FUEL GAS FLOW: {sfc_prediction:.3f}")

print(f"\nBest EXHAUST TEMPERATURE Model: {best_eff_name}")

print(f"Predicted EXHAUST TEMPERATURE: {eff_prediction:.3f}")

