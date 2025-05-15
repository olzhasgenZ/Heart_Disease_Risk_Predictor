Heart Disease Risk Predictor

A Python desktop application that predicts the risk of heart disease using machine learning and a user-friendly graphical interface.

-Description
The Heart Disease Risk Predictor helps users assess their risk of heart disease by entering medical data such as age, blood pressure, cholesterol levels, and more. It uses a Random Forest model trained on a heart disease dataset to provide a percentage risk score along with actionable recommendations. The app features an intuitive interface with tooltips, light/dark theme support, and a resizable window, making it accessible to both technical and non-technical users.
This project showcases skills in Python, machine learning (Scikit-learn), data processing (Pandas), and GUI development (Customtkinter).

-Features
Interactive GUI: Input medical data through fields with dropdowns for categorical options and text entries for numerical values.
Tooltips: Hover over "?" icons for guidance on each input (e.g., "Example: 120 (systolic blood pressure in mmHg)").
Machine Learning: Random Forest model predicts heart disease risk based on 11 medical features.
Theme Support: Switch between light and dark themes for better user experience.
Resizable Window: Adjust the window size with a minimum of 600x500 pixels.
Error Handling: Validates inputs and displays user-friendly error messages.

-Installation
To run the Heart Disease Risk Predictor locally, follow these steps:

Download or clone the repository

Install dependencies:Ensure you have Python 3.8 or higher installed. Then, install the required libraries:
pip install pandas numpy scikit-learn customtkinter


Obtain the dataset:

The application requires the heart_disease_prediction.csv dataset, based on the UCI Heart Disease Dataset.
Download the dataset from Kaggle or another compatible source.
Place the heart_disease_prediction.csv file in the project’s root directory.


Run the application:
python main.py



Alternatively, you can install dependencies from the provided requirements file:
pip install -r requirements.txt

Usage

Launch the application by running python main.py.
Enter your medical data in the input fields (e.g., Age, Sex, Blood Pressure).
Hover over the "?" icons next to each field for help with input formats.
Click the "Assess Risk" button to view your heart disease risk percentage and a recommendation:
Low Risk (<20%): No immediate concern.
Moderate Risk (20–50%): Consult a doctor.
High Risk (>50%): Urgently consult a cardiologist.


Toggle the theme (light/dark) using the switch at the bottom.

Screenshots
Main window showing input fields, tooltips, and risk assessment result.
Requirements

Python: 3.8 or higher
Libraries:
pandas
numpy
scikit-learn
customtkinter



See requirements.txt for a complete list of dependencies.
Dataset
The application uses the heart_disease_prediction.csv dataset, which contains medical data for training the Random Forest model. Key features include:

Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Maximum Heart Rate, Exercise-Induced Angina, ST Depression, ST Slope.
Target: HeartDisease (0 = no disease, 1 = disease).


Note: Place heart_disease_prediction.csv in the root directory to run the application.
Limitations

This tool is for informational purposes only and should not replace professional medical advice.
The model’s accuracy depends on the quality of the dataset and may not generalize to all populations.
Limited input validation (e.g., no range checks for age or blood pressure).

Future Improvements

Add input validation for realistic ranges (e.g., age 18–100).
Include model performance metrics (e.g., accuracy, F1-score).
Add a "Reset" button to clear input fields.
Support saving risk assessment results to a file.

License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code with attribution.
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request with a description of your changes.

Please report bugs or suggest features via GitHub Issues.
Contact
For questions, suggestions, or feedback, please:

Open an issue on GitHub.
Contact me at olzhassar08@gmail.com.

Acknowledgments

Dataset: Based on the UCI Heart Disease Dataset, available on Kaggle.
Libraries: Thanks to the developers of Pandas, Scikit-learn, and Customtkinter.

