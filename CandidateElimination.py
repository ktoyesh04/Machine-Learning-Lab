import pandas as pd

df = pd.read_csv('data.csv')

c = df.shape[1]
specific = ['Φ']*(c-1)
general = [['?']*(c-1) for _ in range(c-1)]

for _, row in df.iterrows():
    if row.EnjoySport == 'Yes':
        for i, value in enumerate(row[:-1]):
            if specific[i] == 'Φ':
                specific[i] = value
            elif specific[i] != value:
                specific[i] = '?'
                general[i][i] = '?'
    else:
        for i, value in enumerate(row[:-1]):
            if specific[i] != value:
                general[i][i] = specific[i]

general = [l for l in general if l.count('?') != (c-1)]
print('Specific Hypothesis\n',specific)
print('General Hypothesis\n', general)
