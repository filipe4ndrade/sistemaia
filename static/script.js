document.addEventListener('DOMContentLoaded', function() {
    // Carregar configurações salvas
    loadSavedSettings();
    
    // Gerenciar visibilidade das opções de validação
    document.querySelectorAll('input[name="validation_method"]').forEach(radio => {
        radio.addEventListener('change', function() {
            document.getElementById('trainTestOptions').style.display = 
                this.value === 'train_test_split' ? 'block' : 'none';
            document.getElementById('crossValOptions').style.display = 
                this.value === 'cross_validation' ? 'block' : 'none';
        });
    });
    
    // Gerenciar upload de arquivo CSV
    document.getElementById('file').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                // Salvar dados do CSV no localStorage
                localStorage.setItem('csv_data', e.target.result);
                
                // Atualizar select de colunas
                updateColumnSelect(e.target.result);
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Gerenciar formulário
    document.getElementById('analysisForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Validar seleções
        const selectedClassifiers = Array.from(document.querySelectorAll('input[name="classifiers"]:checked'))
            .map(cb => cb.value);
            
        if (selectedClassifiers.length === 0) {
            alert('Selecione pelo menos um classificador.');
            return;
        }
        
        // Mostrar spinner
        document.getElementById('loadingSpinner').style.display = 'block';
        document.getElementById('results').style.display = 'none';
        
        // Preparar dados
        const formData = {
            csv_data: localStorage.getItem('csv_data'),
            target_column: document.getElementById('target_column').value,
            validation_method: document.querySelector('input[name="validation_method"]:checked').value,
            test_size: document.getElementById('test_size').value,
            iterations: document.getElementById('iterations').value,
            n_folds: document.getElementById('n_folds').value,
            nan_handling: document.querySelector('input[name="nan_handling"]:checked').value,
            classifiers: selectedClassifiers
        };
        
        // Salvar configurações
        saveSettings(formData);
        
        try {
            // Enviar requisição
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                displayResults(data);
            } else {
                alert(data.error || 'Erro ao processar análise');
            }
        } catch (error) {
            alert('Erro ao conectar com o servidor');
        } finally {
            document.getElementById('loadingSpinner').style.display = 'none';
        }
    });
});

function updateColumnSelect(csvData) {
    const targetSelect = document.getElementById('target_column');
    
    // Limpar opções existentes
    targetSelect.innerHTML = '<option value="">Selecione a coluna...</option>';
    
    try {
        // Decodificar a string base64 para texto
        const decodedData = atob(csvData.split(',')[1]);
        
        // Usar uma regex para lidar com campos que podem conter vírgulas dentro de aspas
        const firstLine = decodedData.split('\n')[0];
        const headers = firstLine.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g) || [];
        
        // Adicionar novas opções
        headers.forEach(header => {
            const option = document.createElement('option');
            // Remover aspas se existirem e fazer trim
            const cleanHeader = header.replace(/^"(.*)"$/, '$1').trim();
            option.value = cleanHeader;
            option.textContent = cleanHeader;
            targetSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Erro ao processar cabeçalhos do CSV:', error);
        alert('Erro ao processar o arquivo CSV. Verifique se o formato está correto.');
    }
}

function displayResults(data) {
    document.getElementById('results').style.display = 'block';
    
    // Exibir informações do dataset
    const datasetStats = document.getElementById('datasetStats');
    datasetStats.innerHTML = `
        <p><strong>Amostras:</strong> ${data.dataset_info.n_samples}</p>
        <p><strong>Features:</strong> ${data.dataset_info.n_features}</p>
        <p><strong>Features utilizadas:</strong> ${data.dataset_info.features.join(', ')}</p>
        <p><strong>Classes:</strong> ${data.dataset_info.classes.join(', ')}</p>
    `;
    
    // Criar gráfico de distribuição de classes
    const distribution = data.dataset_info.class_distribution;
    const ctx = document.createElement('canvas');
    document.getElementById('classDistribution').innerHTML = '';
    document.getElementById('classDistribution').appendChild(ctx);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(distribution),
            datasets: [{
                label: 'Distribuição de Classes',
                data: Object.values(distribution),
                backgroundColor: 'rgba(0, 123, 255, 0.5)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // Exibir resultados
    const resultsTable = document.getElementById('resultsTable');
    resultsTable.innerHTML = data.results.map(result => `
        <tr>
            <td>${result.classifier}</td>
            <td>${(result.accuracy_mean * 100).toFixed(2)}% ± ${(result.accuracy_std * 100).toFixed(2)}%</td>
            <td>${(result.precision_mean * 100).toFixed(2)}% ± ${(result.precision_std * 100).toFixed(2)}%</td>
            <td>${(result.recall_mean * 100).toFixed(2)}% ± ${(result.recall_std * 100).toFixed(2)}%</td>
            <td>${(result.f1_mean * 100).toFixed(2)}% ± ${(result.f1_std * 100).toFixed(2)}%</td>
        </tr>
    `).join('');
    
    // Exibir matrizes de confusão
    const matricesContainer = document.querySelector('#confusionMatrices .matrices-grid');
    matricesContainer.innerHTML = data.results.map(result => `
        <div class="confusion-matrix">
            <h4>${result.classifier}</h4>
            <img src="data:image/png;base64,${result.confusion_matrix}" alt="Matriz de Confusão">
        </div>
    `).join('');
}

function saveSettings(settings) {
    localStorage.setItem('classifier_settings', JSON.stringify({
        validation_method: settings.validation_method,
        test_size: settings.test_size,
        iterations: settings.iterations,
        n_folds: settings.n_folds,
        nan_handling: settings.nan_handling,
        classifiers: settings.classifiers
    }));
}

function loadSavedSettings() {
    const settings = JSON.parse(localStorage.getItem('classifier_settings'));
    if (settings) {
        // Restaurar método de validação
        document.querySelector(`input[name="validation_method"][value="${settings.validation_method}"]`).click();
        
        // Restaurar valores numéricos
        document.getElementById('test_size').value = settings.test_size;
        document.getElementById('iterations').value = settings.iterations;
        document.getElementById('n_folds').value = settings.n_folds;
        // Restaurar classificadores selecionados
        settings.classifiers.forEach(clf => {
            document.querySelector(`input[name="classifiers"][value="${clf}"]`).checked = true;
        });
        if (settings.nan_handling) {
            document.querySelector(`input[name="nan_handling"][value="${settings.nan_handling}"]`).checked = true;
        }
    }
}