import pandas as pd

url = "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2241339&reg=3&lang=1"
tables = pd.read_html(url)

# Annexure-I is typically the first table
state_wise_df = tables[0]
state_wise_df.to_csv("ncrp_state_wise_2021_2023.csv", index=False)