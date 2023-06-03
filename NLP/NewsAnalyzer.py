# Importing dependencies

import pickle
import spacy
import es_core_news_lg
import re
import pandas as pd
import unidecode
import unicodedata as uc
from string import punctuation
from spacy.lang.es import STOP_WORDS
from spacy import displacy
import os
import string
from unidecode import unidecode
from heapq import nlargest

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Connection libraries.
import urllib.request
import urllib.parse
import urllib.error
import requests
from bs4 import BeautifulSoup
import json
import sys

#import ML models

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

sys.setrecursionlimit(1500000)
class NewsAnalyzer:
    """
    buenas
    """

    #Class attributes
    __impact_tags:list = ["mineria", "contaminacion", "deforestacion", "narcotrafico", "infraestructura"]
    
    __nlp = spacy.load("es_core_news_lg")

    def ner_from_str(self, text:str):
        """
        """
        d = pd.DataFrame(text)
        
        datos_preprocess:pd.DataFrame = self.__main_preprocess(d, self.__impact_tags, summary)
        datos_new_columns:pd.DataFrame = self.__main_ner(datos_preprocess)
        datos_new_columns["ImpactPrediction"] =  self.__identify_factor(datos_new_columns["text_preprocess"])
        json_file = open(output_path+"/result.json","w")
        json_dict = datos_new_columns[["TEXTO","ImpactPrediction","NAMES"]].iloc[0].to_json(orient = 'records')
        print(json_dict)
        # print(type(json_dict))
        
        json_file.write(json_dict)
        #json_object = json.dumps(json_dict)
        
        # json_file.write(json_object)
        # df2 = df.to_json(orient = 'records')
        
        json_file.close

        pass

    def ner_from_file(self):
        """
        """

        pass

    def ner_from_url(self):
        """
        """

        pass

    def ner_from_dataset(self,df:pd.DataFrame, output_path:str, summary = False)->None:
        """
        Return a dataframe with a cleaned tags and cleaned text.
        """

        processed = 

        # X = datos_new_columns.text_preprocess
        # y = datos_new_columns.etiqueta_preprocess  
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
        datos_new_columns["ImpactPrediction"] =  self.__identify_factor(datos_new_columns["text_preprocess"])
        # datos_new_columns["summary"] = 
        json_file = open(output_path+"/result.json","w")
        json_dict = datos_new_columns[["TEXTO","ImpactPrediction","NAMES"]].iloc[0].to_json(orient = 'records')
        print(json_dict)
        # print(type(json_dict))
        
        json_file.write(json_dict)
        #json_object = json.dumps(json_dict)
        
        # json_file.write(json_object)
        # df2 = df.to_json(orient = 'records')
        
        json_file.close
        
        
        print(datos_preprocess)
        

    def __identify_factor(self, X, model_name:str = "SGD"):
        """
        """
        loaded_model = pickle.load(open("models/model"+model_name+".sav", 'rb'))
        return loaded_model.predict(X)



    __default_path = "https://www.wwf.org.co/_donde_trabajamos_/amazonas/las_seis_grandes_amenazas_de_la_amazonia/#:~:text=Desde%20el%20a%C3%B1o%202000%20hasta,la%20deforestaci%C3%B3n%20en%20la%20regi%C3%B3n"


    def web_scrapping(self, url:string = __default_path)->BeautifulSoup:
        """
        """
        # URL
        r = requests.get(url)
        data = r.text
        soup = BeautifulSoup(data, "html.parser")
        #Extract content
        for s in soup.select('style'):
            s.extract()

        return soup

    # NLP
    def __find_organizations(self, text, nlp) -> list:
        """
        This function uses of entity labels from spacy to find organizations
        """
        doc = nlp(text)
        list = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # print(ent.text, ent.label_)
                list.append(ent)

        return list

    def __find_dates(self, text, nlp) -> list:
        """
        This function uses of entity labels from spacy to find dates. It also use the re library to find patterns in the text
        that could lead in to a date.
        """
        months = ["enero", "ene", "january", "jan", "febrero", "february", "feb", "marzo", "march", "mar", "abril", "april", "mayo", "may", 'junio', 'june', "jun", "julio", "july", "jul",
                  "agosto", "ago", "august", "aug", "septiembre", 'september', "sep", 'octubre', 'october', "oct", "noviembre", 'november', "nov", "diciembre", "december", "dec"]
        doc = nlp(text)
        lista = []
        for ent in doc.ents:
            if ent.label_ == "DATE":
                # print(ent.text, ent.label_)
                lista.append(ent)
        for m in months:
            if (len(re.findall("([0-9]{2}\s"+m+"\s[0-9]{4})", text)) > 0):
                lista.append(re.findall("([0-9]{2}\s"+m+"\s[0-9]{4})", text))
            if (len(re.findall("([0-9]{2}\s"+m+"\s)", text)) > 0):
                lista.append(re.findall("([0-9]{2}\s"+m+"\s)", text))
            if (len(re.findall("(\s"+m+"\s[0-9]{4})", text)) > 0):
                lista.append(re.findall("(\s"+m+"\s[0-9]{4})", text))
            if (len(re.findall("([0-9]{4}\s"+m+"\s)", text)) > 0):
                lista.append(re.findall("([0-9]{4}\s"+m+"\s)", text))
            if (len(re.findall("(\s"+m+"\s[0-9]{2})", text)) > 0):
                lista.append(re.findall("(\s"+m+"\s[0-9]{2})", text))
            if (len(re.findall("([0-9]{1}\s"+m+"\s[0-9]{4})", text)) > 0):
                lista.append(re.findall("([0-9]{1}\s"+m+"\s[0-9]{4})", text))
            if (len(re.findall("([0-9]{1}\s"+m+"\s)", text)) > 0):
                lista.append(re.findall("([0-9]{1}\s"+m+"\s)", text))
            if (len(re.findall("(\s"+m+"\s[0-9]{1})", text)) > 0):
                lista.append(re.findall("(\s"+m+"\s[0-9]{1})", text))

        return lista

    """
    This function uses the entity labels from spacy to find locations. It also use the re library to find patterns in the text
    that could lead in to a location or address
    """
    def __find_locations(self, text, nlp) -> list:
        municipios = ["victoria", "miriti-parana", "puerto santander", "pedrera", "tarapaca",
                      "leticia", "puerto nariño", "puerto arica", "encanto", "chorrera", "puerto alegria"]
        cardinales = ["Norte", "Sur", "Este", "Oeste", "Occidente", "Oriente"]
        direccion = ["Calle", "Avenida", "Carrera", "Diagonal"]
        doc = nlp(text)
        lista = []
        for ent in doc.ents:
            if ent.label_ == "LOC":
                lista.append(ent)
        for l in municipios:
            if (len(re.findall("("+l+")", text)) > 0):
                lista.append(re.findall("("+l+")", text))
        for c in cardinales:
            if (len(re.findall("("+c+")", text)) > 0):
                lista.append(re.findall("("+c+")", text))
        for d in direccion:
            if (len(re.findall("("+d+"[0-9]{2}\s)", text)) > 0):
                lista.append(re.findall("("+d+"[0-9]{2}\s)", text))
        return lista

    """
    This function makes an attemp of finding person names.
    """

    def __find_names(self, text, nlp) -> list:
        doc = nlp(text)
        person = []
        for ent in doc.ents:
            if ent.label_ == "PER":
                person.append(ent)
        return person
    
    # Feature adicional de resumen
    def __summarize(self, text, per=0.5):
        # nlp = spacy.load('en_core_web_sm')
        nlp = spacy.load('es_core_news_lg')
        doc= nlp(text)
        tokens=[token.text for token in doc]
        word_frequencies={}
        for word in doc:
            if word.text.lower() not in list(STOP_WORDS):
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        max_frequency=max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word]=word_frequencies[word]/max_frequency
        sentence_tokens= [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():                            
                        sentence_scores[sent]=word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent]+=word_frequencies[word.text.lower()]
        select_length=int(len(sentence_tokens)*per)
        summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
        final_summary=[word.text for word in summary]
        summary=''.join(final_summary)
        return summary 

    def __preprocess_text(self, text):
        s = ""
        text = unidecode(text)
        # print(text)
        for char in text:
            s += char.lower()
        # print(s)
        words = ""
        for word in s.split():
            if word not in STOP_WORDS:
                words=words+" "+ word
        return words
    
    def __replace_tildes(self,text):
        text = text.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("▪", "").replace("ü", "u")
        return text

    def __deEmojify(self, text):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags = re.UNICODE)
        s = regrex_pattern.sub(r'',text)
        val_aux = re.sub(r'[\u200b|\u2010|\u1967]+', '', s)  
        s = uc.normalize('NFC',val_aux)
        return s

    def __remove_special_characters(self, text):
        s = ""
        for char in text:
            if char.isalnum() or char.isspace() or char in punctuation:
                s+=char
        s_ = " ".join(s.split())
        val_aux = re.sub(r'[\u200b|\u2010]+', '', s_)  
        s = uc.normalize('NFC',val_aux)
        return s

    def __clean_string(self, string):
        # string = string.replace("  ","")
        string = string.replace("\n","")
        return string
    
    # Metodo que reune todos los metodos de preprocesamiento
    def __process_text_pipeline(self, text:str)->str:
        text = self.__preprocess_text(text)
        text = self.__replace_tildes(text)
        text = self.__deEmojify(text)
        text = self.__remove_special_characters(text)
        text = self.__clean_string(text)
        text = self.__delete_punctuation(text)
        return text
    
    def __delete_punctuation(self, text):
        new_string = text.translate(str.maketrans('', '', string.punctuation))
        return new_string
    
    def __main_preprocess(self, df, tags:list, summary = False):
        final_list = []
        target_list = []
        summary_list = []
        
        for index, row in df.iterrows():
            text = str(row[df.columns.get_loc("TEXTO")])
            text2 = str(row[df.columns.get_loc("ETIQUETA")])
            
            final_list.append(self.__process_text_pipeline(text))
            if summary:
                text3 = str(row[df.columns.get_loc("TEXTO")])
                summary_list.append(self.__summarize(text3))
            tag_parts:list = self.__process_text_pipeline(text2).split(" ") 
            
            tag:str = "" 
            for t in tag_parts:
                if t in tags:    
                    tag = t
                    break
            if tag == "":
                tag = "other"
            target_list.append(tag)
            
        df["text_preprocess"] = final_list
        df["etiqueta_preprocess"] = target_list
        if summary:
            df["sumarize"] = summary_list
        return df
    
    def __main_ner(self, df:pd.DataFrame)-> pd.DataFrame:
        names_list:list = []
        locs_list:list = []
        orgs_list:list = []
        dates_list:list = []
        
        for index, row in df.iterrows():
            text_pre_process = str(row[df.columns.get_loc("text_preprocess")])
            names:str = self.__find_names(text_pre_process,self.__nlp)
            locs:str = self.__find_locations(text_pre_process,self. __nlp)
            orgs:str = self.__find_organizations(text_pre_process, self.__nlp)
            dates:str = self.__find_dates(text_pre_process, self.__nlp)
            
            names_list.append(names)
            locs_list.append(locs)
            orgs_list.append(orgs)
            dates_list.append(dates)
            
            
        df["NAMES"] = names_list
        df["LOCS"] = locs_list
        df["ORGS"] = orgs_list
        df["DATES"] = dates_list
        return df


    # Impact tag ML model methods

    # datos_new_columns
    #X = data_new2.text_preprocess
    #y = data_new2.etiqueta_preprocess

    #__X_train, __X_test, __y_train, __y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    def __multinomial_NB(self):
        nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
        nb.fit(self.X_train, self.y_train)
        my_tags = ["mineria", "contaminacion", "deforestacion", "narcotrafico", "infraestructura"]

        y_pred = nb.predict(self.X_test)

        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred,target_names=my_tags))

    def __SGD_Classifier(self):
        sgd = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
        sgd.fit(self.X_train, self.y_train)
        y_pred = sgd.predict(self.X_test)

        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred,target_names=self.__impact_tags))       

    def __log_reg(self):
        logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
        logreg.fit(self.X_train, self.y_train)
        y_pred = logreg.predict(self.X_test)

        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred,target_names=self.__impact_tags))

    def __lineal_SVC(self):
        svc = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC()),
               ])
        svc.fit(self.X_train, self.y_train)

        y_pred = svc.predict(self.X_test)

        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred,target_names=self.__impact_tags))

    def __random_forest(self):
        rf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)),
               ])
        rf.fit(self.X_train, self.y_train)

        y_pred = rf.predict(self.X_test)

        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred,target_names=self.__impact_tags))


    def __KNeighbors_classifier(self):
        kn = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', KNeighborsClassifier(n_neighbors=5)),
               ])
        kn.fit(self.X_train, self.y_train)

        y_pred = kn.predict(self.X_test)

        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred,target_names=self.__impact_tags))
