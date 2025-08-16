# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
# from sklearn.svm import SVR, SVC
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# import uuid
# import os
# import smtplib
# from email.mime.text import MIMEText
# import re

# # Function to send email with unique code
# def send_email(to_email, unique_code):
#     try:
#         smtp_server = "smtp.gmail.com"
#         smtp_port = 587
#         smtp_email = st.secrets["smtp"]["email"]
#         smtp_password = st.secrets["smtp"]["password"]

#         msg = MIMEText(f"Your unique code for AlgoLab is: {unique_code}\n\nKeep this code safe and use it to log in.")
#         msg["Subject"] = "Your AlgoLab Unique Code"
#         msg["From"] = smtp_email
#         msg["To"] = to_email

#         with smtplib.SMTP(smtp_server, smtp_port) as server:
#             server.starttls()
#             server.login(smtp_email, smtp_password)
#             server.sendmail(smtp_email, to_email, msg.as_string())
#         return True
#     except Exception as e:
#         st.error(f"Failed to send email: {str(e)}. Please check your email address or try again later.")
#         return False

# # Initialize user storage
# if 'users' not in st.session_state:
#     st.session_state.users = pd.DataFrame(columns=["Email", "Code"])
#     if os.path.exists("users.csv"):
#         try:
#             st.session_state.users = pd.read_csv("users.csv")
#         except Exception:
#             pass
#     if st.session_state.users.empty:
#         st.session_state.users = pd.DataFrame(columns=["Email", "Code"])

# # Initialize session state
# if 'max_step' not in st.session_state:
#     st.session_state.max_step = 1
# if 'current_step' not in st.session_state:
#     st.session_state.current_step = 1
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'model' not in st.session_state:
#     st.session_state.model = None
# if 'y_test' not in st.session_state:
#     st.session_state.y_test = None
# if 'y_pred' not in st.session_state:
#     st.session_state.y_pred = None
# if 'task' not in st.session_state:
#     st.session_state.task = None
# if 'X' not in st.session_state:
#     st.session_state.X = None
# if 'y' not in st.session_state:
#     st.session_state.y = None
# if 'features' not in st.session_state:
#     st.session_state.features = None
# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False
# if 'email_submitted' not in st.session_state:
#     st.session_state.email_submitted = False
# if 'current_email' not in st.session_state:
#     st.session_state.current_email = None

# # Login / Register logic
# if not st.session_state.logged_in:
#     st.title("AlgoLab Login / Register")
#     st.markdown("""
#     Recommendation for your app:
    
#     Email + code combo is the most secure for a small educational app.
    
#     Each user gets a unique code tied to their email.
    
#     Only that email + code pair can log in.
    
#     Optional: track sessions to prevent multiple logins from different devices.
#     """)

#     # Step 1: Email input
#     if not st.session_state.email_submitted:
#         email = st.text_input("Enter your email (real email only)")
#         if st.button("Submit Email"):
#             email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
#             if not email or not re.match(email_regex, email):
#                 st.error("Please enter a valid email address (e.g., user@example.com).")
#             else:
#                 st.session_state.current_email = email
#                 st.session_state.email_submitted = True
#                 st.rerun()

#     # Step 2: Code input or registration
#     else:
#         email = st.session_state.current_email
#         users = st.session_state.users
#         user_emails = users['Email'].tolist()

#         if email in user_emails:
#             st.info(f"Email {email} is registered. Enter your unique code to log in.")
#             code_input = st.text_input("Enter your unique code", type="password")
#             if st.button("Login"):
#                 user = users[users['Email'] == email]
#                 if not user.empty and user.iloc[0]['Code'] == code_input:
#                     st.session_state.logged_in = True
#                     st.session_state.email_submitted = False
#                     st.session_state.current_email = None
#                     st.success("Logged in successfully!")
#                     st.rerun()
#                 else:
#                     st.error("Incorrect code.")
#         else:
#             st.info(f"Email {email} is not registered. Registering now...")
#             unique_code = str(uuid.uuid4())[:8].upper()
#             if send_email(email, unique_code):
#                 new_user = pd.DataFrame([[email, unique_code]], columns=["Email", "Code"])
#                 st.session_state.users = pd.concat([users, new_user], ignore_index=True)
#                 try:
#                     st.session_state.users.to_csv("users.csv", index=False)
#                 except Exception:
#                     pass
#                 st.success(f"Registered successfully! Check {email} (including spam/junk) for your code.")
#                 code_input = st.text_input("Enter your unique code", type="password")
#                 if st.button("Login"):
#                     if code_input == unique_code:
#                         st.session_state.logged_in = True
#                         st.session_state.email_submitted = False
#                         st.session_state.current_email = None
#                         st.success("Logged in successfully!")
#                         st.rerun()
#                     else:
#                         st.error("Incorrect code.")
# else:
#     # Sidebar navigation
#     st.sidebar.title("AlgoLab by Lotfullah Muslimwal")
#     st.sidebar.markdown("### Steps")
#     st.sidebar.markdown("""
#     [GitHub](https://github.com/lotfullahmsl) | [LinkedIn](https://www.linkedin.com/in/lotfullahmsl/)
#     """)
#     step_options = ["1. Upload Data", "2. Preprocessing", "3. Model Selection & Training", "4. Results & Visualization", "5. Build Predictor App"]
#     selected_step = st.sidebar.selectbox("Select Step", step_options, index=st.session_state.current_step - 1)
#     step = step_options.index(selected_step) + 1

#     if step > st.session_state.max_step:
#         st.sidebar.warning("Please complete the previous steps first.")
#         st.session_state.current_step = st.session_state.max_step
#     else:
#         st.session_state.current_step = step

#     # Modular functions
#     def load_data():
#         st.title("AlgoLab: Upload Data")
#         st.markdown("### Explanation\nUpload your dataset here. Supported formats: CSV, XLSX, TXT. Select encoding if needed for special characters.")
        
#         uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])
#         encoding = st.selectbox("File Encoding", ["utf-8", "latin-1", "ISO-8859-1", "cp1252"], help="Choose encoding if the file has special characters.")
        
#         if uploaded_file is not None:
#             try:
#                 if uploaded_file.name.endswith(".csv") or uploaded_file.type == "text/csv":
#                     df = pd.read_csv(uploaded_file, encoding=encoding)
#                 elif uploaded_file.name.endswith(".xlsx"):
#                     df = pd.read_excel(uploaded_file)
#                 elif uploaded_file.name.endswith(".txt"):
#                     df = pd.read_csv(uploaded_file, sep="\t", encoding=encoding)
#                 st.session_state.df = df
#                 st.dataframe(df.head())
#                 st.success("Data loaded successfully!")
#             except Exception as e:
#                 st.error(f"Error loading file: {e}")
        
#         if st.session_state.df is not None:
#             if st.button("Next: Preprocessing"):
#                 st.session_state.max_step = max(st.session_state.max_step, 2)
#                 st.session_state.current_step = 2

#     def preprocessing():
#         st.title("AlgoLab: Data Cleaning & Transformation")
#         st.markdown("### Explanation\nClean and transform your data. Drop unnecessary columns, handle missing values, and encode categorical variables.")
        
#         if st.session_state.df is None:
#             st.warning("Please upload data in Step 1.")
#             return
        
#         df = st.session_state.df
        
#         # Drop columns
#         st.subheader("Drop Columns")
#         cols_to_drop = st.multiselect("Select columns to drop", df.columns, help="Dropped columns will be removed permanently from the dataset.")
#         if st.button("Drop Selected Columns") and cols_to_drop:
#             df.drop(columns=cols_to_drop, inplace=True)
#             st.session_state.df = df
#             st.success("Columns dropped!")
        
#         st.dataframe(df.head())
        
#         # Change datatypes
#         st.subheader("Change Column Datatypes")
#         for col in df.columns:
#             current_type = str(df[col].dtype)
#             new_type = st.selectbox(f"{col} (current: {current_type})", ["keep", "float", "int", "str", "category", "datetime"], key=f"type_{col}")
#             if new_type != "keep" and st.button(f"Apply Change to {col}"):
#                 try:
#                     if new_type == "float":
#                         df[col] = df[col].astype(float)
#                     elif new_type == "int":
#                         df[col] = df[col].astype(int)
#                     elif new_type == "str":
#                         df[col] = df[col].astype(str)
#                     elif new_type == "category":
#                         df[col] = df[col].astype("category")
#                     elif new_type == "datetime":
#                         df[col] = pd.to_datetime(df[col])
#                     st.success(f"{col} changed to {new_type}")
#                 except Exception as e:
#                     st.error(f"Failed to change {col}: {e}")
        
#         # Handle missing values
#         missing_cols = df.columns[df.isnull().any()].tolist()
#         if missing_cols:
#             st.subheader("Handle Missing Values")
#             for col in missing_cols:
#                 st.write(f"Column: {col} (missing: {df[col].isnull().sum()})")
#                 is_numeric = pd.api.types.is_numeric_dtype(df[col])
#                 methods = ["Mean", "Median", "Mode", "Constant", "Forward Fill", "Backward Fill"] if is_numeric else ["Mode", "Constant", "Forward Fill", "Backward Fill"]
#                 if is_numeric:
#                     methods.append("KNN Imputer")
#                 method = st.selectbox(f"Method for {col}", methods, key=f"method_{col}")
#                 value = None
#                 if method == "Constant":
#                     value = st.text_input(f"Constant value for {col}", key=f"const_{col}")
#                 if st.button(f"Impute {col}"):
#                     if method == "Mean":
#                         df[col].fillna(df[col].mean(), inplace=True)
#                     elif method == "Median":
#                         df[col].fillna(df[col].median(), inplace=True)
#                     elif method == "Mode":
#                         df[col].fillna(df[col].mode()[0], inplace=True)
#                     elif method == "Constant":
#                         df[col].fillna(value, inplace=True)
#                     elif method == "Forward Fill":
#                         df[col].fillna(method="ffill", inplace=True)
#                     elif method == "Backward Fill":
#                         df[col].fillna(method="bfill", inplace=True)
#                     elif method == "KNN Imputer":
#                         imputer = KNNImputer(n_neighbors=5)
#                         df_numeric = df.select_dtypes(include=np.number)
#                         df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
#                         df[col] = df_numeric[col]
#                     st.success(f"Missing values in {col} imputed!")
        
#         # Encoding
#         st.subheader("Encoding Categorical Columns")
#         cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
#         if cat_cols:
#             col_to_encode = st.selectbox("Select column to encode", cat_cols)
#             encode_method = st.selectbox("Encoding Method", ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"])
#             if encode_method == "Ordinal Encoding":
#                 categories = st.text_input("Categories (comma-separated, in order)", help="e.g., low,medium,high")
#             if st.button("Apply Encoding"):
#                 if encode_method == "Label Encoding":
#                     le = LabelEncoder()
#                     df[col_to_encode] = le.fit_transform(df[col_to_encode])
#                 elif encode_method == "One-Hot Encoding":
#                     df = pd.get_dummies(df, columns=[col_to_encode])
#                 elif encode_method == "Ordinal Encoding":
#                     if categories:
#                         cats = [c.strip() for c in categories.split(",")]
#                         oe = OrdinalEncoder(categories=[cats])
#                         df[col_to_encode] = oe.fit_transform(df[[col_to_encode]])
#                 st.session_state.df = df
#                 st.success(f"{col_to_encode} encoded!")
        
#         st.dataframe(df.head())
        
#         if st.button("Next: Model Selection & Training"):
#             st.session_state.max_step = max(st.session_state.max_step, 3)
#             st.session_state.current_step = 3

#     def model_training():
#         st.title("AlgoLab: Model Selection & Training")
#         st.markdown("### Explanation\nSelect your ML task, target variable, model, and hyperparameters. Train the model and view metrics and code.")
        
#         if st.session_state.df is None:
#             st.warning("Please complete previous steps.")
#             return
        
#         df = st.session_state.df
        
#         task = st.selectbox("Task Type", ["Regression", "Classification"])
#         target = st.selectbox("Target Column", df.columns)
#         features = [col for col in df.columns if col != target]
#         selected_features = st.multiselect("Select Features (default: all)", features, default=features)
        
#         X = df[selected_features]
#         y = df[target]
        
#         st.session_state.task = task
#         st.session_state.X = X
#         st.session_state.y = y
#         st.session_state.features = selected_features
        
#         if task == "Regression":
#             model_options = ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest", "Gradient Boosting", "SVR", "KNN"]
#         else:
#             model_options = ["Logistic Regression", "SVC", "Decision Tree", "Random Forest", "KNN", "Naive Bayes", "Gradient Boosting"]
        
#         selected_model = st.selectbox("Select Model", model_options)
        
#         # Hyperparameters
#         st.subheader("Hyperparameters")
#         params = {}
#         if selected_model in ["Ridge", "Lasso"]:
#             params["alpha"] = st.slider("Alpha", 0.1, 10.0, 1.0)
#         if selected_model in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
#             params["max_depth"] = st.slider("Max Depth", 1, 20, 3, help="None for unlimited")
#             if params["max_depth"] == "None":
#                 params["max_depth"] = None
#         if selected_model in ["Random Forest", "Gradient Boosting"]:
#             params["n_estimators"] = st.slider("Number of Estimators", 50, 200, 100)
#         if "KNN" in selected_model:
#             params["n_neighbors"] = st.slider("Number of Neighbors", 1, 20, 5)
#         if selected_model == "SVR" or selected_model == "SVC":
#             params["C"] = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
#             params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly"])
#         if selected_model == "Logistic Regression":
#             params["C"] = st.slider("C (Inverse Regularization)", 0.1, 10.0, 1.0)
        
#         test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
        
#         if st.button("Fit Model"):
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
#             # Instantiate model
#             if task == "Regression":
#                 if selected_model == "Linear Regression":
#                     model = LinearRegression(**params)
#                 elif selected_model == "Ridge":
#                     model = Ridge(**params)
#                 elif selected_model == "Lasso":
#                     model = Lasso(**params)
#                 elif selected_model == "Decision Tree":
#                     model = DecisionTreeRegressor(**params)
#                 elif selected_model == "Random Forest":
#                     model = RandomForestRegressor(**params)
#                 elif selected_model == "Gradient Boosting":
#                     model = GradientBoostingRegressor(**params)
#                 elif selected_model == "SVR":
#                     model = SVR(**params)
#                 elif selected_model == "KNN":
#                     model = KNeighborsRegressor(**params)
#             else:
#                 if selected_model == "Logistic Regression":
#                     model = LogisticRegression(**params)
#                 elif selected_model == "SVC":
#                     model = SVC(**params)
#                 elif selected_model == "Decision Tree":
#                     model = DecisionTreeClassifier(**params)
#                 elif selected_model == "Random Forest":
#                     model = RandomForestClassifier(**params)
#                 elif selected_model == "KNN":
#                     model = KNeighborsClassifier(**params)
#                 elif selected_model == "Naive Bayes":
#                     model = GaussianNB(**params)
#                 elif selected_model == "Gradient Boosting":
#                     model = GradientBoostingClassifier(**params)
            
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
            
#             st.session_state.model = model
#             st.session_state.y_test = y_test
#             st.session_state.y_pred = y_pred
            
#             # Evaluation metrics
#             st.subheader("Evaluation Metrics")
#             if task == "Regression":
#                 mae = mean_absolute_error(y_test, y_pred)
#                 mse = mean_squared_error(y_test, y_pred)
#                 rmse = np.sqrt(mse)
#                 r2 = r2_score(y_test, y_pred)
#                 st.write(f"MAE: {mae:.4f}")
#                 st.write(f"MSE: {mse:.4f}")
#                 st.write(f"RMSE: {rmse:.4f}")
#                 st.write(f"R²: {r2:.4f}")
#             else:
#                 acc = accuracy_score(y_test, y_pred)
#                 prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
#                 rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
#                 f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
#                 st.write(f"Accuracy: {acc:.4f}")
#                 st.write(f"Precision: {prec:.4f}")
#                 st.write(f"Recall: {rec:.4f}")
#                 st.write(f"F1 Score: {f1:.4f}")
            
#             # Show code
#             st.subheader("Python Code for Training")
#             model_class_name = model.__class__.__name__
#             param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
#             code = f"""
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.{model_class_name.lower().split('regressor')[0] if 'Regressor' in model_class_name else model_class_name.lower().split('classifier')[0]} import {model_class_name}
#     from sklearn.metrics import *  # Import specific metrics as needed
    
#     # Assume df is your DataFrame
#     X = df[{selected_features}]
#     y = df['{target}']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
    
#     model = {model_class_name}({param_str})
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     # Evaluation
#     # Add metrics code here
#     """
#             st.code(code, language="python")
        
#         if st.button("Next: Results & Visualization"):
#             st.session_state.max_step = max(st.session_state.max_step, 4)
#             st.session_state.current_step = 4

#     def visualization():
#         st.title("AlgoLab: Results & Visualization")
#         st.markdown("### Explanation\nVisualize your data and model results. Download preprocessed data and trained model.")
        
#         if st.session_state.model is None:
#             st.warning("Please train a model in Step 3.")
#             return
        
#         df = st.session_state.df
#         task = st.session_state.task
#         model = st.session_state.model
#         y_test = st.session_state.y_test
#         y_pred = st.session_state.y_pred
#         features = st.session_state.features
        
#         vis_options = ["Correlation Heatmap", "Pairplot", "Distribution Plot", "Feature Importance"]
#         if task == "Classification":
#             vis_options.append("Confusion Matrix")
#         if task == "Regression":
#             vis_options.append("Predicted vs Actual")
        
#         vis_type = st.selectbox("Select Visualization", vis_options)
        
#         if vis_type == "Correlation Heatmap":
#             fig, ax = plt.subplots(figsize=(10, 8))
#             sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
#             st.pyplot(fig)
#         elif vis_type == "Pairplot":
#             sns.pairplot(df.sample(min(100, len(df))))
#             st.pyplot(plt)
#         elif vis_type == "Distribution Plot":
#             col = st.selectbox("Select Column", df.columns)
#             fig, ax = plt.subplots()
#             sns.histplot(df[col], kde=True, ax=ax)
#             st.pyplot(fig)
#         elif vis_type == "Feature Importance":
#             if hasattr(model, "feature_importances_"):
#                 imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
#                 fig, ax = plt.subplots()
#                 sns.barplot(x=imp.values, y=imp.index, ax=ax)
#                 ax.set_title("Feature Importance")
#                 st.pyplot(fig)
#             else:
#                 st.info("This model does not support feature importance.")
#         elif vis_type == "Confusion Matrix":
#             cm = confusion_matrix(y_test, y_pred)
#             fig, ax = plt.subplots()
#             sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
#             st.pyplot(fig)
#         elif vis_type == "Predicted vs Actual":
#             fig, ax = plt.subplots()
#             ax.scatter(y_test, y_pred)
#             ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
#             ax.set_xlabel("Actual")
#             ax.set_ylabel("Predicted")
#             st.pyplot(fig)
        
#         # Downloads
#         st.subheader("Downloads")
#         csv_buffer = io.StringIO()
#         df.to_csv(csv_buffer, index=False)
#         st.download_button("Download Preprocessed Dataset", csv_buffer.getvalue(), "preprocessed.csv", "text/csv")
        
#         model_buffer = io.BytesIO()
#         joblib.dump(model, model_buffer)
#         model_buffer.seek(0)
#         st.download_button("Download Trained Model", model_buffer, "model.pkl", "application/octet-stream")

#         if st.button("Next: Build Predictor App"):
#             st.session_state.max_step = max(st.session_state.max_step, 5)
#             st.session_state.current_step = 5

#     def build_predictor():
#         st.title("AlgoLab: Build Predictor App")
#         st.markdown("### Predictor App\nUse the trained model to make predictions by entering feature values.")
#         st.markdown("**Note**: For categorical features, ensure inputs match the encoded values used during training (e.g., numbers for LabelEncoder, exact categories for OneHotEncoder).")

#         if st.session_state.model is None:
#             st.warning("Please train a model in Step 3.")
#             return

#         model = st.session_state.model
#         features = st.session_state.features
#         task = st.session_state.task
#         X = st.session_state.X

#         st.subheader("Enter Feature Values")
        
#         # Create input fields based on feature types
#         input_data = {}
#         for feature in features:
#             dtype = X[feature].dtype
#             if pd.api.types.is_numeric_dtype(dtype):
#                 input_data[feature] = st.number_input(f"Enter {feature}", value=0.0, key=f"input_{feature}")
#             else:
#                 input_data[feature] = st.text_input(f"Enter {feature}", key=f"input_{feature}")

#         if st.button("Predict"):
#             try:
#                 input_df = pd.DataFrame([input_data])
#                 prediction = model.predict(input_df)
#                 st.success(f"Prediction: {prediction[0]}")
#             except Exception as e:
#                 st.error(f"Prediction failed: {e}. Ensure inputs match the expected format (e.g., encoded values for categorical features).")

#     # Main app logic
#     if st.session_state.current_step == 1:
#         load_data()
#     elif st.session_state.current_step == 2:
#         preprocessing()
#     elif st.session_state.current_step == 3:
#         model_training()
#     elif st.session_state.current_step == 4:
#         visualization()
#     elif st.session_state.current_step == 5:
#         build_predictor()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import uuid
import smtplib
from email.mime.text import MIMEText
import re
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Function to send email with unique code
def send_email(to_email, unique_code):
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_email = st.secrets["smtp"]["email"]
        smtp_password = st.secrets["smtp"]["password"]

        msg = MIMEText(f"Your unique code for AlgoLab is: {unique_code}\n\nKeep this code safe and use it to log in.")
        msg["Subject"] = "Your AlgoLab Unique Code"
        msg["From"] = smtp_email
        msg["To"] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}. Please check your email address or try again later.")
        return False

# Initialize MongoDB client
def init_mongo_client():
    try:
        uri = st.secrets["mongodb"]["uri"]
        client = MongoClient(uri)
        db = client[st.secrets["mongodb"]["database"]]
        collection = db[st.secrets["mongodb"]["collection"]]
        # Create index on Email to ensure uniqueness
        collection.create_index("Email", unique=True)
        # Test connection
        client.server_info()  # Raises an error if connection fails
        return collection
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

# Load users from MongoDB
def load_users_from_mongo():
    try:
        collection = init_mongo_client()
        if collection is None:
            st.error("MongoDB client initialization failed.")
            return pd.DataFrame(columns=["Email", "Code"])
        users = list(collection.find({}, {"_id": 0, "Email": 1, "Code": 1}))
        if not users:
            return pd.DataFrame(columns=["Email", "Code"])
        return pd.DataFrame(users)
    except Exception as e:
        st.error(f"Failed to load users from MongoDB: {str(e)}")
        return pd.DataFrame(columns=["Email", "Code"])

# Save user to MongoDB
def save_user_to_mongo(email, code):
    try:
        collection = init_mongo_client()
        if collection is None:
            st.error("MongoDB client initialization failed.")
            return False
        collection.insert_one({"Email": email, "Code": code})
        return True
    except DuplicateKeyError:
        st.error("Email already exists in the database.")
        return False
    except Exception as e:
        st.error(f"Failed to save user to MongoDB: {str(e)}")
        return False

# Initialize user storage
if 'users' not in st.session_state:
    st.session_state.users = load_users_from_mongo()
    # Ensure DataFrame has Email and Code columns
    if st.session_state.users.empty or 'Email' not in st.session_state.users.columns or 'Code' not in st.session_state.users.columns:
        st.session_state.users = pd.DataFrame(columns=["Email", "Code"])

# Initialize session state
if 'max_step' not in st.session_state:
    st.session_state.max_step = 1
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'task' not in st.session_state:
    st.session_state.task = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'email_submitted' not in st.session_state:
    st.session_state.email_submitted = False
if 'current_email' not in st.session_state:
    st.session_state.current_email = None

# Login / Register logic
if not st.session_state.logged_in:
    st.title("AlgoLab Login / Register")
    st.markdown("""
    Recommendation for your app:
    
    Email + code combo is the most secure for a small educational app.
    
    Each user gets a unique code tied to their email.
    
    Only that email + code pair can log in.
    
    Optional: track sessions to prevent multiple logins from different devices.
    """)

    # Step 1: Email input
    if not st.session_state.email_submitted:
        email = st.text_input("Enter your email (real email only)")
        if st.button("Submit Email"):
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not email or not re.match(email_regex, email):
                st.error("Please enter a valid email address (e.g., user@example.com).")
            else:
                st.session_state.current_email = email
                st.session_state.email_submitted = True
                st.rerun()

    # Step 2: Code input or registration
    else:
        email = st.session_state.current_email
        users = st.session_state.users
        user_emails = users['Email'].tolist() if not users.empty and 'Email' in users.columns else []

        if email in user_emails:
            st.info(f"Email {email} is registered. Enter your unique code to log in.")
            code_input = st.text_input("Enter your unique code", type="password")
            if st.button("Login"):
                user = users[users['Email'] == email]
                if not user.empty and user.iloc[0]['Code'] == code_input:
                    st.session_state.logged_in = True
                    st.session_state.email_submitted = False
                    st.session_state.current_email = None
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Incorrect code.")
        else:
            st.info(f"Email {email} is not registered. Registering now...")
            unique_code = str(uuid.uuid4())[:8].upper()
            if send_email(email, unique_code):
                if save_user_to_mongo(email, unique_code):
                    st.session_state.users = load_users_from_mongo()  # Reload users
                    st.success(f"Registered successfully! Check {email} (including spam/junk) for your code.")
                else:
                    st.error("Failed to save user to MongoDB. Please try again.")
                code_input = st.text_input("Enter your unique code", type="password")
                if st.button("Login"):
                    user = st.session_state.users[st.session_state.users['Email'] == email]
                    if not user.empty and user.iloc[0]['Code'] == code_input:
                        st.session_state.logged_in = True
                        st.session_state.email_submitted = False
                        st.session_state.current_email = None
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Incorrect code.")
else:
    # Sidebar navigation
    st.sidebar.title("AlgoLab by Lotfullah Muslimwal")
    st.sidebar.markdown("### Steps")
    st.sidebar.markdown("""
    [GitHub](https://github.com/lotfullahmsl) | [LinkedIn](https://www.linkedin.com/in/lotfullahmsl/)
    """)
    step_options = ["1. Upload Data", "2. Preprocessing", "3. Model Selection & Training", "4. Results & Visualization", "5. Build Predictor App"]
    selected_step = st.sidebar.selectbox("Select Step", step_options, index=st.session_state.current_step - 1)
    step = step_options.index(selected_step) + 1

    if step > st.session_state.max_step:
        st.sidebar.warning("Please complete the previous steps first.")
        st.session_state.current_step = st.session_state.max_step
    else:
        st.session_state.current_step = step

    # Modular functions
    def load_data():
        st.title("AlgoLab: Upload Data")
        st.markdown("### Explanation\nUpload your dataset here. Supported formats: CSV, XLSX, TXT. Select encoding if needed for special characters.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])
        encoding = st.selectbox("File Encoding", ["utf-8", "latin-1", "ISO-8859-1", "cp1252"], help="Choose encoding if the file has special characters.")
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv") or uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith(".txt"):
                    df = pd.read_csv(uploaded_file, sep="\t", encoding=encoding)
                st.session_state.df = df
                st.dataframe(df.head())
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        if st.session_state.df is not None:
            if st.button("Next: Preprocessing"):
                st.session_state.max_step = max(st.session_state.max_step, 2)
                st.session_state.current_step = 2

    def preprocessing():
        st.title("AlgoLab: Data Cleaning & Transformation")
        st.markdown("### Explanation\nClean and transform your data. Drop unnecessary columns, handle missing values, and encode categorical variables.")
        
        if st.session_state.df is None:
            st.warning("Please upload data in Step 1.")
            return
        
        df = st.session_state.df
        
        # Drop columns
        st.subheader("Drop Columns")
        cols_to_drop = st.multiselect("Select columns to drop", df.columns, help="Dropped columns will be removed permanently from the dataset.")
        if st.button("Drop Selected Columns") and cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            st.session_state.df = df
            st.success("Columns dropped!")
        
        st.dataframe(df.head())
        
        # Change datatypes
        st.subheader("Change Column Datatypes")
        for col in df.columns:
            current_type = str(df[col].dtype)
            new_type = st.selectbox(f"{col} (current: {current_type})", ["keep", "float", "int", "str", "category", "datetime"], key=f"type_{col}")
            if new_type != "keep" and st.button(f"Apply Change to {col}"):
                try:
                    if new_type == "float":
                        df[col] = df[col].astype(float)
                    elif new_type == "int":
                        df[col] = df[col].astype(int)
                    elif new_type == "str":
                        df[col] = df[col].astype(str)
                    elif new_type == "category":
                        df[col] = df[col].astype("category")
                    elif new_type == "datetime":
                        df[col] = pd.to_datetime(df[col])
                    st.success(f"{col} changed to {new_type}")
                except Exception as e:
                    st.error(f"Failed to change {col}: {e}")
        
        # Handle missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            st.subheader("Handle Missing Values")
            for col in missing_cols:
                st.write(f"Column: {col} (missing: {df[col].isnull().sum()})")
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                methods = ["Mean", "Median", "Mode", "Constant", "Forward Fill", "Backward Fill"] if is_numeric else ["Mode", "Constant", "Forward Fill", "Backward Fill"]
                if is_numeric:
                    methods.append("KNN Imputer")
                method = st.selectbox(f"Method for {col}", methods, key=f"method_{col}")
                value = None
                if method == "Constant":
                    value = st.text_input(f"Constant value for {col}", key=f"const_{col}")
                if st.button(f"Impute {col}"):
                    if method == "Mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == "Median":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif method == "Mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == "Constant":
                        df[col].fillna(value, inplace=True)
                    elif method == "Forward Fill":
                        df[col].fillna(method="ffill", inplace=True)
                    elif method == "Backward Fill":
                        df[col].fillna(method="bfill", inplace=True)
                    elif method == "KNN Imputer":
                        imputer = KNNImputer(n_neighbors=5)
                        df_numeric = df.select_dtypes(include=np.number)
                        df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
                        df[col] = df_numeric[col]
                    st.success(f"Missing values in {col} imputed!")
        
        # Encoding
        st.subheader("Encoding Categorical Columns")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            col_to_encode = st.selectbox("Select column to encode", cat_cols)
            encode_method = st.selectbox("Encoding Method", ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"])
            if encode_method == "Ordinal Encoding":
                categories = st.text_input("Categories (comma-separated, in order)", help="e.g., low,medium,high")
            if st.button("Apply Encoding"):
                if encode_method == "Label Encoding":
                    le = LabelEncoder()
                    df[col_to_encode] = le.fit_transform(df[col_to_encode])
                elif encode_method == "One-Hot Encoding":
                    df = pd.get_dummies(df, columns=[col_to_encode])
                elif encode_method == "Ordinal Encoding":
                    if categories:
                        cats = [c.strip() for c in categories.split(",")]
                        oe = OrdinalEncoder(categories=[cats])
                        df[col_to_encode] = oe.fit_transform(df[[col_to_encode]])
                st.session_state.df = df
                st.success(f"{col_to_encode} encoded!")
        
        st.dataframe(df.head())
        
        if st.button("Next: Model Selection & Training"):
            st.session_state.max_step = max(st.session_state.max_step, 3)
            st.session_state.current_step = 3

    def model_training():
        st.title("AlgoLab: Model Selection & Training")
        st.markdown("### Explanation\nSelect your ML task, target variable, model, and hyperparameters. Train the model and view metrics and code.")
        
        if st.session_state.df is None:
            st.warning("Please complete previous steps.")
            return
        
        df = st.session_state.df
        
        task = st.selectbox("Task Type", ["Regression", "Classification"])
        target = st.selectbox("Target Column", df.columns)
        features = [col for col in df.columns if col != target]
        selected_features = st.multiselect("Select Features (default: all)", features, default=features)
        
        X = df[selected_features]
        y = df[target]
        
        st.session_state.task = task
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.features = selected_features
        
        if task == "Regression":
            model_options = ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest", "Gradient Boosting", "SVR", "KNN"]
        else:
            model_options = ["Logistic Regression", "SVC", "Decision Tree", "Random Forest", "KNN", "Naive Bayes", "Gradient Boosting"]
        
        selected_model = st.selectbox("Select Model", model_options)
        
        # Hyperparameters
        st.subheader("Hyperparameters")
        params = {}
        if selected_model in ["Ridge", "Lasso"]:
            params["alpha"] = st.slider("Alpha", 0.1, 10.0, 1.0)
        if selected_model in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
            params["max_depth"] = st.slider("Max Depth", 1, 20, 3, help="None for unlimited")
            if params["max_depth"] == "None":
                params["max_depth"] = None
        if selected_model in ["Random Forest", "Gradient Boosting"]:
            params["n_estimators"] = st.slider("Number of Estimators", 50, 200, 100)
        if "KNN" in selected_model:
            params["n_neighbors"] = st.slider("Number of Neighbors", 1, 20, 5)
        if selected_model == "SVR" or selected_model == "SVC":
            params["C"] = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
            params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly"])
        if selected_model == "Logistic Regression":
            params["C"] = st.slider("C (Inverse Regularization)", 0.1, 10.0, 1.0)
        
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
        
        if st.button("Fit Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Instantiate model
            if task == "Regression":
                if selected_model == "Linear Regression":
                    model = LinearRegression(**params)
                elif selected_model == "Ridge":
                    model = Ridge(**params)
                elif selected_model == "Lasso":
                    model = Lasso(**params)
                elif selected_model == "Decision Tree":
                    model = DecisionTreeRegressor(**params)
                elif selected_model == "Random Forest":
                    model = RandomForestRegressor(**params)
                elif selected_model == "Gradient Boosting":
                    model = GradientBoostingRegressor(**params)
                elif selected_model == "SVR":
                    model = SVR(**params)
                elif selected_model == "KNN":
                    model = KNeighborsRegressor(**params)
            else:
                if selected_model == "Logistic Regression":
                    model = LogisticRegression(**params)
                elif selected_model == "SVC":
                    model = SVC(**params)
                elif selected_model == "Decision Tree":
                    model = DecisionTreeClassifier(**params)
                elif selected_model == "Random Forest":
                    model = RandomForestClassifier(**params)
                elif selected_model == "KNN":
                    model = KNeighborsClassifier(**params)
                elif selected_model == "Naive Bayes":
                    model = GaussianNB(**params)
                elif selected_model == "Gradient Boosting":
                    model = GradientBoostingClassifier(**params)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.session_state.model = model
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            
            # Evaluation metrics
            st.subheader("Evaluation Metrics")
            if task == "Regression":
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"MAE: {mae:.4f}")
                st.write(f"MSE: {mse:.4f}")
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"R²: {r2:.4f}")
            else:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                st.write(f"Accuracy: {acc:.4f}")
                st.write(f"Precision: {prec:.4f}")
                st.write(f"Recall: {rec:.4f}")
                st.write(f"F1 Score: {f1:.4f}")
            
            # Show code
            st.subheader("Python Code for Training")
            model_class_name = model.__class__.__name__
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            code = f"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.{model_class_name.lower().split('regressor')[0] if 'Regressor' in model_class_name else model_class_name.lower().split('classifier')[0]} import {model_class_name}
    from sklearn.metrics import *  # Import specific metrics as needed
    
    # Assume df is your DataFrame
    X = df[{selected_features}]
    y = df['{target}']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
    
    model = {model_class_name}({param_str})
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    # Add metrics code here
    """
            st.code(code, language="python")
        
        if st.button("Next: Results & Visualization"):
            st.session_state.max_step = max(st.session_state.max_step, 4)
            st.session_state.current_step = 4

    def visualization():
        st.title("AlgoLab: Results & Visualization")
        st.markdown("### Explanation\nVisualize your data and model results. Download preprocessed data and trained model.")
        
        if st.session_state.model is None:
            st.warning("Please train a model in Step 3.")
            return
        
        df = st.session_state.df
        task = st.session_state.task
        model = st.session_state.model
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        features = st.session_state.features
        
        vis_options = ["Correlation Heatmap", "Pairplot", "Distribution Plot", "Feature Importance"]
        if task == "Classification":
            vis_options.append("Confusion Matrix")
        if task == "Regression":
            vis_options.append("Predicted vs Actual")
        
        vis_type = st.selectbox("Select Visualization", vis_options)
        
        if vis_type == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        elif vis_type == "Pairplot":
            sns.pairplot(df.sample(min(100, len(df))))
            st.pyplot(plt)
        elif vis_type == "Distribution Plot":
            col = st.selectbox("Select Column", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)
        elif vis_type == "Feature Importance":
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
                fig, ax = plt.subplots()
                sns.barplot(x=imp.values, y=imp.index, ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)
            else:
                st.info("This model does not support feature importance.")
        elif vis_type == "Confusion Matrix":
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
        elif vis_type == "Predicted vs Actual":
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)
        
        # Downloads
        st.subheader("Downloads")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("Download Preprocessed Dataset", csv_buffer.getvalue(), "preprocessed.csv", "text/csv")
        
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)
        st.download_button("Download Trained Model", model_buffer, "model.pkl", "application/octet-stream")

        if st.button("Next: Build Predictor App"):
            st.session_state.max_step = max(st.session_state.max_step, 5)
            st.session_state.current_step = 5

    def build_predictor():
        st.title("AlgoLab: Build Predictor App")
        st.markdown("### Predictor App\nUse the trained model to make predictions by entering feature values.")
        st.markdown("**Note**: For categorical features, ensure inputs match the encoded values used during training (e.g., numbers for LabelEncoder, exact categories for OneHotEncoder).")

        if st.session_state.model is None:
            st.warning("Please train a model in Step 3.")
            return

        model = st.session_state.model
        features = st.session_state.features
        task = st.session_state.task
        X = st.session_state.X

        st.subheader("Enter Feature Values")
        
        # Create input fields based on feature types
        input_data = {}
        for feature in features:
            dtype = X[feature].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                input_data[feature] = st.number_input(f"Enter {feature}", value=0.0, key=f"input_{feature}")
            else:
                input_data[feature] = st.text_input(f"Enter {feature}", key=f"input_{feature}")

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                st.success(f"Prediction: {prediction[0]}")
            except Exception as e:
                st.error(f"Prediction failed: {e}. Ensure inputs match the expected format (e.g., encoded values for categorical features).")

    # Main app logic
    if st.session_state.current_step == 1:
        load_data()
    elif st.session_state.current_step == 2:
        preprocessing()
    elif st.session_state.current_step == 3:
        model_training()
    elif st.session_state.current_step == 4:
        visualization()
    elif st.session_state.current_step == 5:
        build_predictor()