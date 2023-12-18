import pandas as pd

df = pd.read_csv('data.csv')
c = df.shape[1]
h = ['Φ']*(c-1)

rows = df[df.EnjoySport == 'Yes'].iloc[:, :-1]

for _, row in rows.iterrows():
    for i, value in enumerate(row):
        if h[i] == 'Φ':
            h[i] = value
        elif h[i] != value:
            h[i] = '?'

print('The final hypothesis is:', h)
