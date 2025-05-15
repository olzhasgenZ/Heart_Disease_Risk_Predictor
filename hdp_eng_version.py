import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import customtkinter as ctk
from tkinter import messagebox

class Tooltip(ctk.CTkToplevel):
    def __init__(self, parent, text, **kwargs):
        super().__init__(parent, **kwargs)
        self.withdraw()
        self.overrideredirect(True)

        # Set colors based on the current theme
        self.update_colors()

        self.label = ctk.CTkLabel(self, text=text, justify="left",
                                corner_radius=6, fg_color=self.bg_color,
                                text_color=self.text_color,
                                padx=10, pady=5, wraplength=250)
        self.label.pack()

    def update_colors(self):
        current_theme = ctk.get_appearance_mode()
        self.bg_color = "#FFFFFF" if current_theme == "Light" else "#353638"
        self.text_color = "#000000" if current_theme == "Light" else "#FFFFFF"

    def show(self, x, y):
        self.geometry(f"+{x}+{y}")
        self.deiconify()

    def hide(self):
        self.withdraw()

    def update_theme(self):
        self.update_colors()
        self.label.configure(fg_color=self.bg_color, text_color=self.text_color)

class HoverTooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if not self.tooltip:
            self.tooltip = Tooltip(self.widget, self.text)
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        self.tooltip.show(x, y)

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.hide()

    def update_theme(self):
        if self.tooltip:
            self.tooltip.update_theme()

class HeartDiseaseApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Heart Disease Risk Prediction")
        self.geometry("900x700")
        self.resizable(True, True)
        self.minsize(600, 500)

        # Theme setup
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Load data and train model
        self.df = self.load_data()
        self.model, self.scaler, self.feature_columns = self.train_model()

        # Lists to store HoverTooltip and help buttons
        self.tooltips_list = []
        self.help_buttons = []

        # Create interface
        self.create_widgets()

    def load_data(self):
        df = pd.read_csv('heart_disease_prediction.csv')
        df = df[(df['RestingBP'] != 0) & (df['Cholesterol'] != 0)]
        return df

    def train_model(self):
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        df_encoded = pd.get_dummies(self.df, columns=categorical_cols)

        X = df_encoded.drop('HeartDisease', axis=1)
        y = df_encoded['HeartDisease']
        feature_columns = X.columns.tolist()

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        return model, scaler, feature_columns

    def create_widgets(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(self.main_frame,
                                 text="Heart Disease Risk Assessment",
                                 font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=(0, 20))

        # Input frame
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.pack(fill="x", padx=10, pady=10)

        # Input fields with tooltips
        self.entries = {}
        self.tooltips = {
            'Age': "Example: 45 (age in years)",
            'Sex': "M - male, F - female",
            'ChestPainType': "ATA - atypical angina\nNAP - non-anginal pain\nASY - asymptomatic\nTA - typical angina",
            'RestingBP': "Example: 120 (systolic blood pressure in mmHg)",
            'Cholesterol': "Example: 200 (cholesterol level in mg/dl)",
            'FastingBS': "0 - blood sugar <120 mg/dl\n1 - blood sugar >120 mg/dl",
            'RestingECG': "Normal - normal\nST - ST segment abnormality\nLVH - left ventricular hypertrophy",
            'MaxHR': "Example: 150 (maximum heart rate)",
            'ExerciseAngina': "Y - yes, N - no",
            'Oldpeak': "Example: 1.5 (ST segment depression)",
            'ST_Slope': "Up - upsloping\nFlat - flat\nDown - downsloping"
        }

        # Set initial tooltip button color based on theme
        current_theme = ctk.get_appearance_mode()
        button_color = "#D3D3D3" if current_theme == "Light" else "#2b2b2b"

        for i, (field, hint) in enumerate(self.tooltips.items(), start=1):
            row_frame = ctk.CTkFrame(self.input_frame)
            row_frame.pack(fill="x", pady=5)

            label = ctk.CTkLabel(row_frame, text=f"{i}. {self.get_field_label(field)}:", width=180, anchor="w")
            label.pack(side="left", padx=(0, 5))

            if field in ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
                values = self.get_combobox_values(field)
                self.entries[field] = ctk.CTkComboBox(row_frame, values=values, width=180)
            else:
                self.entries[field] = ctk.CTkEntry(row_frame, width=180)
            self.entries[field].pack(side="left", padx=(0, 5))

            # Help icon with dynamic color
            help_btn = ctk.CTkLabel(row_frame, text="?", width=20,
                                   cursor="hand2", fg_color=button_color, corner_radius=10)
            help_btn.pack(side="left")
            tooltip = HoverTooltip(help_btn, hint)
            self.tooltips_list.append(tooltip)
            self.help_buttons.append(help_btn)

        self.evaluate_btn = ctk.CTkButton(self.main_frame, text="Assess Risk",
                                        command=self.assess_risk, height=40)
        self.evaluate_btn.pack(pady=20)

        self.result_frame = ctk.CTkFrame(self.main_frame, height=80)
        self.result_frame.pack(fill="x", pady=(0, 10))

        self.result_label = ctk.CTkLabel(self.result_frame, text="",
                                       font=ctk.CTkFont(size=14),
                                       wraplength=600, justify="left")
        self.result_label.pack(pady=10, padx=10, fill="both", expand=True)

        self.theme_switch = ctk.CTkSwitch(self.main_frame,
                                        text="Dark Theme",
                                        command=self.toggle_theme)
        self.theme_switch.pack(pady=10)
        self.theme_switch.select()

    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode()
        new_mode = "Light" if current_mode == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.theme_switch.configure(text=f"{new_mode} Theme")

        # Update theme for tooltips and buttons
        button_color = "#D3D3D3" if new_mode == "Light" else "#2b2b2b"
        for tooltip in self.tooltips_list:
            tooltip.update_theme()
        for button in self.help_buttons:
            button.configure(fg_color=button_color)

    def get_field_label(self, field):
        labels = {
            'Age': "Age (years)",
            'Sex': "Sex",
            'ChestPainType': "Chest Pain Type",
            'RestingBP': "Blood Pressure (mmHg)",
            'Cholesterol': "Cholesterol (mg/dl)",
            'FastingBS': "Fasting Blood Sugar",
            'RestingECG': "Resting ECG Result",
            'MaxHR': "Maximum Heart Rate",
            'ExerciseAngina': "Exercise-Induced Angina",
            'Oldpeak': "ST Depression",
            'ST_Slope': "ST Slope"
        }
        return labels.get(field, field)

    def get_combobox_values(self, field):
        values = {
            'Sex': ["M", "F"],
            'ChestPainType': ["ATA", "NAP", "ASY", "TA"],
            'FastingBS': ["0", "1"],
            'RestingECG': ["Normal", "ST", "LVH"],
            'ExerciseAngina': ["Y", "N"],
            'ST_Slope': ["Up", "Flat", "Down"]
        }
        return values.get(field, [])

    def prepare_input(self, answers):
        data = pd.DataFrame([answers])
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        data = pd.get_dummies(data, columns=categorical_cols)

        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0

        data = data[self.feature_columns]
        scaled_data = self.scaler.transform(data)
        return scaled_data

    def assess_risk(self):
        try:
            answers = {}
            for field, entry in self.entries.items():
                value = entry.get()
                if not value:
                    raise ValueError(f"The '{self.get_field_label(field)}' field is empty")

                if field in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']:
                    answers[field] = int(value)
                elif field == 'Oldpeak':
                    answers[field] = float(value)
                else:
                    answers[field] = value

            input_data = self.prepare_input(answers)
            proba = self.model.predict_proba(input_data)[0][1]
            risk_percent = round(proba * 100, 1)

            result_text = f"Heart Disease Risk: {risk_percent}%\n"
            if risk_percent < 20:
                result_text += "✅ Low Risk"
            elif 20 <= risk_percent < 50:
                result_text += "⚠️ Moderate Risk - Consult a doctor"
            else:
                result_text += "🚨 High Risk - Urgently consult a cardiologist"

            self.result_label.configure(text=result_text)

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = HeartDiseaseApp()
    app.mainloop()