
# EXAMPLE: HOW TO LOAD THE MODEL
import pickle
import joblib
import pandas as pd

# Method 1: Load with joblib (faster for sklearn models)
model_package = joblib.load('../models/regression/best_life_expectancy_model.joblib')


# Make predictions (example)
# new_data = pd.DataFrame(your_data, columns=feature_names)
# predictions = model.predict(new_data)
