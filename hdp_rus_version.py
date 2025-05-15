#Предиктор сердечных рисков


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

        # Определяем цвета в зависимости от текущей темы
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

        self.title("Прогноз сердечных заболеваний")
        self.geometry("900x700")
        self.resizable(True, True)
        self.minsize(600, 500)

        # Настройка темы
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Загрузка данных и обучение модели
        self.df = self.load_data()
        self.model, self.scaler, self.feature_columns = self.train_model()

        # Списки для хранения HoverTooltip и кнопок подсказок
        self.tooltips_list = []
        self.help_buttons = []

        # Создание интерфейса
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
        # Основной фрейм
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Заголовок
        title_label = ctk.CTkLabel(self.main_frame,
                                 text="Оценка риска сердечных заболеваний",
                                 font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=(0, 20))

        # Фрейм для полей ввода
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.pack(fill="x", padx=10, pady=10)

        # Поля ввода с подсказками
        self.entries = {}
        self.tooltips = {
            'Age': "Пример: 45 (возраст в годах)",
            'Sex': "M - мужской, F - женский",
            'ChestPainType': "ATA - атипичная стенокардия\nNAP - неангинозная боль\nASY - бессимптомно\nTA - типичная стенокардия",
            'RestingBP': "Пример: 120 (систолическое давление в мм рт.ст.)",
            'Cholesterol': "Пример: 200 (уровень холестерина в mg/dl)",
            'FastingBS': "0 - сахар <120 мг/дл\n1 - сахар >120 мг/дл",
            'RestingECG': "Normal - норма\nST - отклонения сегмента ST\nLVH - гипертрофия левого желудочка",
            'MaxHR': "Пример: 150 (максимальная ЧСС)",
            'ExerciseAngina': "Y - да, N - нет",
            'Oldpeak': "Пример: 1.5 (депрессия ST сегмента)",
            'ST_Slope': "Up - восходящий\nFlat - плоский\nDown - нисходящий"
        }

        # Определяем начальный цвет кнопки подсказки в зависимости от темы
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

            # Значок подсказки с динамическим цветом
            help_btn = ctk.CTkLabel(row_frame, text="?", width=20,
                                   cursor="hand2", fg_color=button_color, corner_radius=10)
            help_btn.pack(side="left")
            tooltip = HoverTooltip(help_btn, hint)
            self.tooltips_list.append(tooltip)
            self.help_buttons.append(help_btn)  # Сохраняем кнопку в список

        self.evaluate_btn = ctk.CTkButton(self.main_frame, text="Оценить риск",
                                        command=self.assess_risk, height=40)
        self.evaluate_btn.pack(pady=20)

        self.result_frame = ctk.CTkFrame(self.main_frame, height=80)
        self.result_frame.pack(fill="x", pady=(0, 10))

        self.result_label = ctk.CTkLabel(self.result_frame, text="",
                                       font=ctk.CTkFont(size=14),
                                       wraplength=600, justify="left")
        self.result_label.pack(pady=10, padx=10, fill="both", expand=True)

        self.theme_switch = ctk.CTkSwitch(self.main_frame,
                                        text="Темная тема",
                                        command=self.toggle_theme)
        self.theme_switch.pack(pady=10)
        self.theme_switch.select()

    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode()
        new_mode = "Light" if current_mode == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.theme_switch.configure(text=f"{new_mode} тема")

        # Обновляем тему всех подсказок и кнопок
        button_color = "#D3D3D3" if new_mode == "Light" else "#2b2b2b"
        for tooltip in self.tooltips_list:
            tooltip.update_theme()
        for button in self.help_buttons:
            button.configure(fg_color=button_color)

    def get_field_label(self, field):
        labels = {
            'Age': "Возраст (лет)",
            'Sex': "Пол",
            'ChestPainType': "Тип боли в груди",
            'RestingBP': "Артериальное давление (мм рт.ст.)",
            'Cholesterol': "Уровень холестерина (mg/dl)",
            'FastingBS': "Уровень сахара натощак",
            'RestingECG': "Результат ЭКГ в покое",
            'MaxHR': "Максимальная ЧСС",
            'ExerciseAngina': "Стенокардия при нагрузке",
            'Oldpeak': "Депрессия ST сегмента",
            'ST_Slope': "Наклон ST сегмента"
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
                    raise ValueError(f"Поле '{self.get_field_label(field)}' не заполнено")

                if field in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']:
                    answers[field] = int(value)
                elif field == 'Oldpeak':
                    answers[field] = float(value)
                else:
                    answers[field] = value

            input_data = self.prepare_input(answers)
            proba = self.model.predict_proba(input_data)[0][1]
            risk_percent = round(proba * 100, 1)

            result_text = f"Риск сердечного заболевания: {risk_percent}%\n"
            if risk_percent < 20:
                result_text += "✅ Низкий риск"
            elif 20 <= risk_percent < 50:
                result_text += "⚠️ Умеренный риск - рекомендуется консультация врача"
            else:
                result_text += "🚨 Высокий риск - настоятельно рекомендуется обратиться к кардиологу"

            self.result_label.configure(text=result_text)

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные данные: {str(e)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    app = HeartDiseaseApp()
    app.mainloop()