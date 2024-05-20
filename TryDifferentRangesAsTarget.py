import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data_only_mie_features = pd.read_csv('Extracted_Features_2.csv')
data_only_mie_features.dropna(inplace=True)

features = ['PUNTI_NEL_QUADRO',
            'FIXATIONS',
            'mean_diam_pupilla',
            'min_diam_pupilla',
            'max_diam_pupilla',
            'std_diam_pupilla',
            'diff_max_quadro_blank',
            'max_similar_fixations',
            'max_durata_similar_fixations',
            'n_aree_interesse']
X = data_only_mie_features[features]
y = data_only_mie_features['voto']

# Funzione per classificare il target in base ai range forniti
def classify_target(y, lower_range, upper_range):
    return y.apply(lambda x: 1 if x >= upper_range[0] and x <= upper_range[1] else (0 if x >= lower_range[0] and x <= lower_range[1] else -1))

def balance_dataset(X, y):
    # 1. Identifica la classe minore e conta il numero di istanze
    class_counts = y.value_counts()
    min_class = class_counts.idxmin()
    min_class_count = class_counts.min()

    # 2. Ottieni gli indici di entrambe le classi
    min_class_indices = y[y == min_class].index
    max_class_indices = y[y != min_class].index

    # 3. Sotto-campiona la classe maggiore
    max_class_sampled_indices = max_class_indices[:min_class_count]

    # 4. Combina gli indici
    balanced_indices = min_class_indices.union(max_class_sampled_indices)

    # 5. Filtra X e y per ottenere i dataset bilanciati
    X_balanced = X.loc[balanced_indices].reset_index(drop=True)
    y_balanced = y.loc[balanced_indices].reset_index(drop=True)
    
    return X_balanced, y_balanced

# Funzione per training, cross-validation in 5-folds 
def train_and_evaluate(models, X, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {name: {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0} for name in models.keys()}
    num_folds = skf.get_n_splits(X, y)
    
    for train_index, valid_index in skf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]
        y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]
        
        # Bilanciare i dati del fold
        X_train_balanced, y_train_balanced = balance_dataset(X_train_fold, y_train_fold)
        
        # Dividere i dati in set di addestramento e di test
        X_train, X_test, y_train, y_test = train_test_split(X_train_balanced, y_train_balanced, test_size=0.2, random_state=seed)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name]['accuracy'] += accuracy_score(y_test, y_pred)
            results[name]['precision'] += precision_score(y_test, y_pred)
            results[name]['recall'] += recall_score(y_test, y_pred)
            results[name]['f1_score'] += f1_score(y_test, y_pred)
    
    # Calcolo della media per ogni metrica
    for name in results:
        results[name]['accuracy'] /= num_folds
        results[name]['precision'] /= num_folds
        results[name]['recall'] /= num_folds
        results[name]['f1_score'] /= num_folds
    
    return results

# Definire i modelli
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC()
}

# Range di finestre mobili
window_sizes = range(5,25,2)

# Tenere traccia dei risultati
all_results = []

seed = 1
# Iterare sui window_sizes
for window_size1 in window_sizes:
    for window_size2 in window_sizes:
        for lower_start in range(0, 25-window_size1, 2):  # Spostare la finestra dello 0
            lower_range = (lower_start, lower_start + window_size1 - 1)
            for upper_start in range(50, 25+window_size2, -2):  # Spostare la finestra dell'1
                try:
                    upper_range = (upper_start - window_size2 + 1, upper_start)

                    # Classificare il target in base ai range
                    y_classified = classify_target(y, lower_range, upper_range)

                    # Filtrare i dati con target valido (0 o 1)
                    valid_indices = y_classified[y_classified != -1].index
                    X_valid = X.loc[valid_indices]
                    y_valid = y_classified.loc[valid_indices]

                    # Verifico che nel y_classified vi siano almeno 500 esempi per classe
                    class_counts = y_valid.value_counts()
                    min_class = class_counts.idxmin()
                    min_class_count = class_counts.min()

                    if min_class_count < 500:
                        continue

                    #Bilancio il dataset in base alla classe minore in modo da tenere allineati X e y
                    # Ottieni gli indici di entrambe le classi
                    min_class_indices = y_valid[y_valid == min_class].index
                    max_class_indices = y_valid[y_valid != min_class].index
                    # Sotto-campiona la classe maggiore
                    max_class_sampled_indices = max_class_indices[:min_class_count]
                    # Combina gli indici
                    balanced_indices = min_class_indices.union(max_class_sampled_indices)
                    # Filtra X e y per ottenere i dataset bilanciati
                    X_balanced = X_valid.loc[balanced_indices].reset_index(drop=True)
                    y_balanced = y_valid.loc[balanced_indices].reset_index(drop=True)

                    # Dividere i dati in set di addestramento e di test
                    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=seed)

                    # Addestrare e valutare i modelli
                    results = train_and_evaluate(models, X_balanced, y_balanced, seed=seed)

                    all_results.append({
                        'window_size1': window_size1,
                        'window_size2': window_size2,
                        'lower_range': lower_range,
                        'upper_range': upper_range,
                        'Logistic Regression': results["Logistic Regression"],
                        'Decision Tree': results["Decision Tree"],
                        'Random Forest': results["Random Forest"],
                        'XGBoost': results["XGBoost"],
                        'SVM': results["SVM"]
                    })
                    seed = (seed + 1) % 42
                except:
                    print(f'{lower_range} {upper_range}')

def getMax(x):
    el = []
    for m in models:
        el.append(x[m]['accuracy'])
    return max(el)

# Creare una lista ordinata dei risultati in base all'accuratezza dei modelli
sorted_results = sorted(all_results, key=lambda x: getMax(x), reverse=True)

try:
    df_result = pd.DataFrame.from_dict(sorted_results)
    df_result.to_csv('Results.csv', index=False)
except:
    print('Errore Dataframe')



