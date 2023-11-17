import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib  # Import joblib for model serialization

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


class HealthcareResourceModel:
    def __init__(self):
        """
        Initializes the HealthcareResourceModel class.
        """
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess_data(self, data_source):
        """
        Loads the data and performs initial preprocessing steps.
        """        
        if isinstance(data_source, str):
            # If data_source is a string, assume it's a file path and load data from the file
            self.data = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            # If data_source is a DataFrame, use it directly
            self.data = data_source
            
        # Standardizing date formats
        date_columns = ['dob', 'admitted_at', 'discharged_at', 'inserted_at', 'updated_at']
        for col in date_columns:
            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')

        # Standardizing categorical data
        categorical_columns = ['sex', 'state', 'visit_type', 'type']
        for col in categorical_columns:
            self.data[col] = self.data[col].str.lower()

        # Removing rows with missing values
        self.data = self.data.dropna(subset=['state', 'admitted_at', 'discharged_at'])
        
        
    def dataset_summary(self):
        """
        Returns a comprehensive summary of the dataset, including missing value percentages.
        """
        if self.data is None:
            raise ValueError("Dataset is not loaded. Please run load_and_preprocess_data method.")

        num_rows, num_columns = self.data.shape
        column_data_types = self.data.dtypes
        missing_values = self.data.isnull().sum()
        missing_percentages = (missing_values / num_rows) * 100  # Calculate missing value percentages
        numeric_columns = self.data.select_dtypes(include=['number'])

        summary = {
            "Number of Rows": num_rows,
            # "Number of Columns": num_columns,
            "Data Types": column_data_types,
            "Missing Values": missing_values,
            "Missing Value Percentages": missing_percentages,
        }
        
        summary_df = pd.DataFrame(summary)

        formatted_summary = summary_df.to_string()
        
        print(formatted_summary)

        return summary_df


    def feature_engineering(self):
        """
        Performs feature engineering on the dataset.
        """
        self.data['admission_month'] = self.data['admitted_at'].dt.month
        self.data['admission_dayofweek'] = self.data['admitted_at'].dt.dayofweek
        current_year = datetime.now().year
        self.data['age'] = current_year - self.data['dob'].dt.year
        self.data['length_of_stay'] = (self.data['discharged_at'] - self.data['admitted_at']).dt.total_seconds() / (24 * 3600)

        # One-hot encoding for categorical variables
        categorical_cols = ['sex', 'state', 'visit_type']

        label_encoder = LabelEncoder()

        for col in categorical_cols:
            self.data[col] = label_encoder.fit_transform(self.data[col].astype(str))

        self.data.drop(['dob', 'admitted_at', 'discharged_at', 'inserted_at', 'updated_at'], axis=1, inplace=True)

    def split_data(self):
        """
        Splits the data into training and testing sets.
        """
        X = self.data.drop(['length_of_stay', 'institution_id', 'patient_id', 'visit_id', 'type'], axis=1)
        y = self.data['length_of_stay']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def model_selection(self):
        """
        Selects the best regression model among different options.
        """
        models = {
          "Linear Regression": LinearRegression(),
          "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
          "Random Forest Regressor": RandomForestRegressor(random_state=42),
          "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        }

        best_model = None
        best_mae = float('inf')

        for model_name, model in models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)

            print(f"Model: {model_name}, MAE: {mae}")

            if mae < best_mae:
                best_mae = mae
                best_model = model_name

        self.model = models[best_model]

        self.save_best_model(best_model)

        return best_model, best_mae


    def save_best_model(self, best_model):
        """
        Saves the best-trained model to a file.
        """
        # Train the selected best model on the full dataset
        self.model.fit(self.X_train, self.y_train)

        # Save the trained model to a file using joblib
        joblib.dump(self.model, 'best_model.pkl')

    def evaluate_model(self):
        """
        Evaluates the model using MAE and RMSE.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Please run model_selection method.")
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        return mae, rmse

    def plot_feature_importance(self, top_n=10):
        """
        Generates a feature importance plot for the best model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Please run model_selection method.")

        # Load the saved model
        saved_model = joblib.load('best_model.pkl')

        # Getting feature importances from the model
        feature_importances = saved_model.feature_importances_

        # Creating a DataFrame for feature importances
        features = pd.DataFrame({'Feature': self.X_train.columns, 'Importance': feature_importances})

        # Sorting the features by importance
        features_sorted = features.sort_values(by='Importance', ascending=False)

        # Plotting the top N most important features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=features_sorted.head(top_n))
        plt.title(f'Top {top_n} Feature Importances in {self.model}')
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature')
        plt.show()