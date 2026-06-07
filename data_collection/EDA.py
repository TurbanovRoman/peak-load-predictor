import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r"C:\Users\qwesd\Desktop\Obshaya\Vuz 4 kurs\Diplome\proactive-scaler\data_collection\synth_azure.csv")
df.head()
plt.figure(figsize=(10,7))
plt.plot(df['ds'],df['rps'])
plt.show()

