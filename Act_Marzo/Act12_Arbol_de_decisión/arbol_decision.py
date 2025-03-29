# Imports needed for the script
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn.tree import export_graphviz
from graphviz import Source

artists_billboard = pd.read_csv("artists_billboard_fix3.csv")
print("Shape:")
print(artists_billboard.shape)
print("Head")
print(artists_billboard.head())
print(artists_billboard.groupby('top').size())

#gráficas por columna
sb.catplot(x='artist_type',data=artists_billboard,kind="count")
plt.show()
sb.catplot(x='mood',data=artists_billboard,kind="count", aspect=3)
plt.show()
sb.catplot(x='tempo',data=artists_billboard,hue='top',kind="count")
plt.show()
sb.catplot(x='genre',data=artists_billboard,kind="count", aspect=3)
plt.show()
sb.catplot(x='anioNacimiento',data=artists_billboard,kind="count", aspect=3)
plt.show()


colores = ['orange', 'blue']
tamanios = [60, 40]

asignar = []
asignar2 = []

for index, row in artists_billboard.iterrows():
    asignar.append(colores[row['top'] % 2])  
    asignar2.append(tamanios[row['top'] % 2]) 

f1 = artists_billboard['chart_date'].values
f2 = artists_billboard['durationSeg'].values

print(len(f1), len(f2), len(asignar), len(asignar2))
plt.scatter(f1, f2, c=asignar, s=asignar2)
plt.axis([20030101, 20160101, 0, 600])
plt.show()

def edad_fix(anio):
   if anio==0:
     return None
   return anio

artists_billboard['anioNacimiento']=artists_billboard.apply(lambda x:edad_fix(x['anioNacimiento']),axis=1)

def calcula_edad(anio,cuando):
   cad=str(cuando)
   momento=cad[:4]
   if anio==0.0:
     return None
   return int(momento)-anio

artists_billboard['edad_en_billboard']=artists_billboard.apply(lambda x:calcula_edad(x['anioNacimiento'],x['chart_date']),axis=1)

age_avg=artists_billboard['edad_en_billboard'].mean()
age_std=artists_billboard['edad_en_billboard'].std()
age_null_count=artists_billboard['edad_en_billboard'].isnull().sum()
age_null_random_list=np.random.randint(age_avg-age_std,age_avg+age_std,size=age_null_count)
conValoresNulos=np.isnan(artists_billboard['edad_en_billboard'])
artists_billboard.loc[np.isnan(artists_billboard['edad_en_billboard']),'edad_en_billboard']=age_null_random_list
artists_billboard['edad_en_billboard']=artists_billboard['edad_en_billboard'].astype(int)
print("EdadPromedio:"+str(age_avg))
print("DesvióStdEdad:"+str(age_std))
print("Intervaloparaasignaredadaleatoria:"+str(int(age_avg-age_std))+"a"+str(int(age_avg+age_std)))

f1=artists_billboard['edad_en_billboard'].values
f2=artists_billboard.index
colores=['orange','blue','green']
asignar=[]
for index, row in artists_billboard.iterrows():
   if(conValoresNulos[index]):
      asignar.append(colores[2])#verde
   else:
      asignar.append(colores[row['top']])

plt.scatter(f1,f2,c=asignar,s=30)
plt.axis([15,50,0,650])
plt.show()

#MAPPING

#MoodMapping
artists_billboard['moodEncoded']=artists_billboard['mood'].map({
   'Energizing':6,
   'Empowering':6,
   'Cool':5,
   'Yearning':4,#anhelo,deseo,ansia
   'Excited':5,#emocionado
   'Defiant':3,
   'Sensual':2,
   'Gritty':3,#coraje
   'Sophisticated':4,
   'Aggressive':4,#provocativo
   'Fiery':4,#caracterfuerte
   'Urgent':3,
   'Rowdy':4,#ruidosoalboroto
   'Sentimental':4,
   'Easygoing':1,#sencillo
   'Melancholy':4,
   'Romantic':2,
   'Peaceful':1,
   'Brooding':4,#melancolico
   'Upbeat':5,#optimistaalegre
   'Stirring':5,#emocionante
   'Lively':5,#animado
   'Other':0,'':0}).astype(int)

#TempoMapping
artists_billboard['tempoEncoded']=artists_billboard['tempo'].map({
   'FastTempo':0,
   'MediumTempo':1,
   'SlowTempo':2}).fillna(0).astype(int)

#genre
# Mapeo de la columna 'genre' con manejo de NaN o vacíos
artists_billboard['genreEncoded'] = artists_billboard['genre'].map({
    'Urban': 4,
    'Pop': 3,
    'Traditional': 2,
    'Alternative&Punk': 1,
    'Electronica': 1,
    'Rock': 1,
    'Soundtrack': 0,
    'Jazz': 0,
    'Other': 0,
    '': 0  # Se maneja el valor vacío
}).fillna(0).astype(int)  # Reemplazar NaN con 0

# Mapeo de la columna 'artist_type' con manejo de NaN o vacíos
artists_billboard['artist_typeEncoded'] = artists_billboard['artist_type'].map({
    'Female': 2,
    'Male': 3,
    'Mixed': 1,
    '': 0  # Se maneja el valor vacío
}).fillna(0).astype(int)  # Reemplazar NaN con 0


#Mappingedadenlaque llegaron albillboard
artists_billboard.loc[artists_billboard['edad_en_billboard']<=21,'edadEncoded']=0
artists_billboard.loc[(artists_billboard['edad_en_billboard']>21)&(artists_billboard['edad_en_billboard']<=26),'edadEncoded']=1
artists_billboard.loc[(artists_billboard['edad_en_billboard']>26)&(artists_billboard['edad_en_billboard']<=30),'edadEncoded']=2
artists_billboard.loc[(artists_billboard['edad_en_billboard']>30)&(artists_billboard['edad_en_billboard']<=40),'edadEncoded']=3
artists_billboard.loc[artists_billboard['edad_en_billboard']>40,'edadEncoded']=4

#MappingSongDuration
artists_billboard.loc[artists_billboard['durationSeg']<=150,'durationEncoded']=0
artists_billboard.loc[(artists_billboard['durationSeg'] > 150) & (artists_billboard['durationSeg'] <= 180), 'durationEncoded'] = 1
artists_billboard.loc[(artists_billboard['durationSeg'] > 180) & (artists_billboard['durationSeg'] <= 210), 'durationEncoded'] = 2
artists_billboard.loc[(artists_billboard['durationSeg'] > 210) & (artists_billboard['durationSeg'] <= 240), 'durationEncoded'] = 3
artists_billboard.loc[(artists_billboard['durationSeg'] > 240) & (artists_billboard['durationSeg'] <= 270), 'durationEncoded'] = 4
artists_billboard.loc[(artists_billboard['durationSeg'] > 270) & (artists_billboard['durationSeg'] <= 300), 'durationEncoded'] = 5
artists_billboard.loc[ artists_billboard['durationSeg'] > 300, 'durationEncoded'] = 6

drop_elements = ['id','title','artist','mood','tempo','genre','artist_type','chart_date','anioNacimiento','durationSeg','edad_en_billboard']
artists_encoded = artists_billboard.drop(drop_elements, axis = 1)

mood_table=artists_encoded[['moodEncoded', 'top']].groupby(['moodEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#print(mood_table)
artist_table= artists_encoded[['artist_typeEncoded', 'top']].groupby(['artist_typeEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#print(artist_table)
genere_table=artists_encoded[['genreEncoded', 'top']].groupby(['genreEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#print(genere_table)
tempo_table= artists_encoded[['tempoEncoded', 'top']].groupby(['tempoEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#print(tempo_table)
duration_table= artists_encoded[['durationEncoded', 'top']].groupby(['durationEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#print(duration_table)
edad_table=artists_encoded[['edadEncoded', 'top']].groupby(['edadEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#print(edad_table)

cv=KFold(n_splits=10)#Numero deseado de "folds" que haremos
accuracies=list()
max_attributes=len(list(artists_encoded))
depth_range=range(1,max_attributes+1)

#Testearemos la profundidad de 1 acantidad de atributos + 1
for depth in depth_range:
   fold_accuracy=[]
   tree_model=tree.DecisionTreeClassifier(
      criterion='entropy',
      min_samples_split=20,
      min_samples_leaf=5,
      max_depth=depth,
      class_weight={1:3.5})
   for train_fold, valid_fold in cv.split(artists_encoded):
     f_train=artists_encoded.loc[train_fold]
     f_valid=artists_encoded.loc[valid_fold]
     
     model=tree_model.fit(X=f_train.drop(['top'],axis=1), y=f_train["top"])
     valid_acc=model.score(X=f_valid.drop(['top'],axis=1),y=f_valid["top"])#calculamoslaprecisionconelsegmentode validacion
     fold_accuracy.append(valid_acc)
   
   
   avg=sum(fold_accuracy)/len(fold_accuracy)
   accuracies.append(avg)
   


#Mostramoslosresultadosobtenidos
df=pd.DataFrame({"MaxDepth":depth_range,"AverageAccuracy":accuracies})
df=df[["MaxDepth","AverageAccuracy"]]
#print(df.to_string(index=False))

#Creararraysdeentrenamientoy lasetiquetasqueindicansillegóatopono
y_train=artists_encoded['top']
x_train=artists_encoded.drop(['top'],axis=1).values

#CrearArboldedecisionconprofundidad=4
decision_tree=tree.DecisionTreeClassifier(
   criterion='entropy',
   min_samples_split=20,
   min_samples_leaf=5,
   max_depth=4,
   class_weight={1:3.5})
decision_tree.fit(x_train,y_train)

#exportar elmodeloa archivo.dot
with open("tree1.dot", "w") as f:
    export_graphviz(
        decision_tree,
        out_file=f,
        max_depth=7,
        impurity=True,
        feature_names=list(artists_encoded.drop(['top'], axis=1)),
        class_names=['No', 'N1Billboard'],
        rounded=True,
        filled=True
    )

# Usar graphviz para convertir el archivo .dot a una imagen .png
Source.from_file("tree1.dot").render("tree1", format="png")

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
print("Precisión del árbol: ", acc_decision_tree)

#predecir artista CAMILA CABELLO featuring YOUNG THUG
# con su canción Havana llego a numero 1 Billboard US en 2017

x_test = pd.DataFrame(columns=('top','moodEncoded', 'tempoEncoded', 'genreEncoded','artist_typeEncoded','edadEncoded','durationEncoded'))
x_test.loc[0] = (1,5,2,4,1,0,3)
y_pred = decision_tree.predict(x_test.drop(['top'], axis = 1))

y_proba = decision_tree.predict_proba(x_test.drop(['top'], axis = 1))
proba=y_proba[0][y_pred]*100
print("Prediccion: " + str(y_pred))
print("Probabilidad de Acierto Havana: " + str(round(proba[0], 2))+"%")

x_test = pd.DataFrame(columns=('top','moodEncoded', 'tempoEncoded', 'genreEncoded','artist_typeEncoded','edadEncoded','durationEncoded'))
x_test.loc[0] = (0,4,2,1,3,2,3)
y_pred = decision_tree.predict(x_test.drop(['top'], axis = 1))

y_proba = decision_tree.predict_proba(x_test.drop(['top'], axis = 1))
proba2 = y_proba[0][y_pred]* 100
print("Prediccion: " + str(y_pred))
print("Probabilidad de Acierto Believer: " + str(round(proba2[0], 2))+"%")