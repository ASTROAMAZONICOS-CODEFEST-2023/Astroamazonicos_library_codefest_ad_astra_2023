from NewsAnalyzer import NewsAnalyzer
import pandas as pd

instance = NewsAnalyzer()
instance.ner_from_dataset(pd.read_excel("noticias.xlsx"),".",summary=False)