import pandas as pd





column_number = 14


data = pd.read_csv('z_1_days.csv', encoding='ISO-8859-1')


if column_number < 0 or column_number >= len(data.columns):
    print("Invalid")
else:
    print(f" {column_number}:", data.columns[column_number])

    selected_column = data.iloc[:, column_number].values
    selected_column = selected_column.reshape(-1, 1)
    print("val:\n", selected_column[:5])
