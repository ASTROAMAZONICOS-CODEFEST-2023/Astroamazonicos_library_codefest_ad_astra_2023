# Importing dependencies

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

#import ML models

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

class NewsAnalyzer:
    """
    buenas
    """

    #Class attributes
    __impact_tags:list = ["mineria", "contaminacion", "deforestacion", "narcotrafico", "infraestructura"]
    
    __nlp = spacy.load("es_core_news_lg")

    def ner_from_str(self):
        """
        """

        pass

    def ner_from_file(self):
        """
        """

        pass

    def ner_from_url(self):
        """
        """

        pass

    def ner_from_dataset(self,df:pd.DataFrame, output_path:str)->None:
        """Return a dataframe with a cleaned tags and cleaned text.
        """

        datos_preprocess:pd.DataFrame = self.__main_preprocess(df, self.__impact_tags)
        datos_new_columns:pd.DataFrame = self.__main_ner(datos_preprocess)
        print(datos_preprocess.head())


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
    def __summarize(self, text, per):
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
    
    def __replace_tildes(text):
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
        text = self.preprocess_text(text)
        text = self.replace_tildes(text)
        text = self.deEmojify(text)
        text = self.remove_special_characters(text)
        text = self.clean_string(text)
        text = self.delete_punctuation(text)
        return text
    
    def __delete_punctuation(self, text):
        new_string = text.translate(str.maketrans('', '', string.punctuation))
        return new_string
    
    def __main_preprocess(self, df, tags:list):
        final_list = []
        target_list = []
        for index, row in df.iterrows():
            text = str(row[df.columns.get_loc("TEXTO")])
            text2 = str(row[df.columns.get_loc("ETIQUETA")])
            final_list.append(self.process_text_pipeline(text))
            
            tag_parts:list = self.process_text_pipeline(text2).split(" ") 
            
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
        return df
    
    def __main_ner(self, df:pd.DataFrame)-> pd.DataFrame:
        names_list:list = []
        locs_list:list = []
        orgs_list:list = []
        dates_list:list = []
        
        for index, row in df.iterrows():
            text_pre_process = str(row[df.columns.get_loc("text_preprocess")])
            names:str = self.find_names(text_pre_process,nlp)
            locs:str = self.find_locations(text_pre_process, nlp)
            orgs:str = self.find_organizations(text_pre_process, nlp)
            dates:str = self.find_dates(text_pre_process, nlp)
            
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
    X = data_new2.text_preprocess
    y = data_new2.etiqueta_preprocess

    __X_train, __X_test, __y_train, __y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

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
