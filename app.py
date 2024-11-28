from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer 
import io
import json
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

CLASSIFIERS = {
    'naive_bayes': ('Naive Bayes', GaussianNB()),
    'decision_tree': ('Árvore de Decisão', DecisionTreeClassifier(random_state=42)),
    'svm': ('SVM', SVC(random_state=42)),
    'knn': ('KNN', KNeighborsClassifier()),
    'random_forest': ('Random Forest', RandomForestClassifier(random_state=42)),
    'logistic_regression': ('Regressão Logística', LogisticRegression(random_state=42))
}

def create_confusion_matrix_plot(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    # Carregar dados do CSV do localStorage (enviado como string base64)
    csv_data = base64.b64decode(data['csv_data'].split(',')[1]).decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_data))

    # Parâmetros da análise
    target_column = data['target_column']
    validation_method = data['validation_method']
    test_size = float(data['test_size'])
    n_iterations = int(data['iterations'])
    n_folds = int(data.get('n_folds', 5))
    selected_classifiers = data['classifiers']
    nan_handling = data['nan_handling']
    
    # Obter a opção de deletar ou não as linhas com NaN
    delete_nan_option = data.get('delete_nan_option', 'yes')  # Padrão é 'yes' caso não enviado
    
    if target_column not in df.columns:
        return jsonify({'error': 'Coluna alvo não encontrada'}), 400
    
    # Preparar dados
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Remover colunas completamente vazias
    X = X.dropna(how='all', axis=1)
    
    if delete_nan_option == 'yes':
        # Remover linhas com qualquer NaN (confirmado pelo usuário)
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
    else:
        # Deixar as linhas com NaN intactas (não removê-las)
        pass
    
    if nan_handling == 'drop':
        # Caso o nan_handling seja 'drop', aplicar a mesma lógica de remover NaNs
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
    else:
        # Imputação de valores ausentes
        imputer = SimpleImputer(strategy=nan_handling)
        X_imputed = imputer.fit_transform(X)
        
        # Garantir que o número de colunas seja consistente
        X = pd.DataFrame(X_imputed, columns=X.columns)
        
        # Remover linhas onde o alvo é NaN
        mask = y.notna()
        X = X[mask]
        y = y[mask]
    
    # Verificar se há dados suficientes após o tratamento de NaN
    if len(X) == 0:
        return jsonify({'error': 'Não há dados suficientes após o tratamento de valores ausentes'}), 400
    

    
    class_names = sorted(y.unique())
    
    # Inicializar dicionário de resultados
    results = {}
    confusion_matrices = {}
    
    for clf_name in selected_classifiers:
        clf_display_name, clf = CLASSIFIERS[clf_name]
        results[clf_name] = {
            'name': clf_display_name,
            'metrics': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        }
        
        if validation_method == 'train_test_split':
            for _ in range(n_iterations):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=None
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                results[clf_name]['metrics']['accuracy'].append(accuracy_score(y_test, y_pred))
                results[clf_name]['metrics']['precision'].append(
                    precision_score(y_test, y_pred, average='weighted'))
                results[clf_name]['metrics']['recall'].append(
                    recall_score(y_test, y_pred, average='weighted'))
                results[clf_name]['metrics']['f1'].append(
                    f1_score(y_test, y_pred, average='weighted'))
                
                if _ == 0:  # Salvar matriz de confusão da primeira iteração
                    confusion_matrices[clf_name] = create_confusion_matrix_plot(
                        y_test, y_pred, class_names)
        
        else:  # Cross-validation
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                if metric == 'accuracy':
                    scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
                else:
                    scores = cross_val_score(clf, X, y, cv=kf, 
                                          scoring=f'{metric}_weighted')
                results[clf_name]['metrics'][metric] = scores.tolist()
            
            # Criar matriz de confusão usando todo o dataset
            clf.fit(X, y)
            y_pred = clf.predict(X)
            confusion_matrices[clf_name] = create_confusion_matrix_plot(
                y, y_pred, class_names)
    
    # Calcular estatísticas finais
    final_results = []
    for clf_name, result in results.items():
        metrics = result['metrics']
        final_results.append({
            'classifier': result['name'],
            'accuracy_mean': np.mean(metrics['accuracy']),
            'accuracy_std': np.std(metrics['accuracy']),
            'precision_mean': np.mean(metrics['precision']),
            'precision_std': np.std(metrics['precision']),
            'recall_mean': np.mean(metrics['recall']),
            'recall_std': np.std(metrics['recall']),
            'f1_mean': np.mean(metrics['f1']),
            'f1_std': np.std(metrics['f1']),
            'confusion_matrix': confusion_matrices[clf_name]
        })
    
    # Adicionar informações sobre o dataset
    dataset_info = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'class_distribution': y.value_counts().to_dict(),
        'features': list(X.columns),
        'classes': list(class_names)
    }
    
    return jsonify({
        'results': final_results,
        'dataset_info': dataset_info,
        'validation_method': validation_method
    })

if __name__ == '__main__':
    app.run(debug=True)
