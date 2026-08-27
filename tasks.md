# Implementation Plan: PoC IA Triagem de Radiografias de Tórax

## Overview

Implementação incremental do pipeline de IA para triagem de radiografias de tórax da KD Diagnósticos. O plano segue a ordem: estrutura do projeto → pré-processamento DICOM → extração de labels → divisão de dados → treinamento SageMaker → inferência batch → métricas e relatórios. Cada componente é implementado, testado e integrado antes de avançar.

## Tasks

- [ ] 1. Estrutura do projeto e interfaces base
  - [ ] 1.1 Criar estrutura de diretórios e arquivos de configuração
    - Criar diretórios `src/`, `src/preprocessing/`, `src/labeling/`, `src/splitting/`, `src/training/`, `src/inference/`, `src/utils/`, `tests/`, `tests/unit/`, `tests/property/`, `notebooks/`
    - Criar `requirements.txt` com todas as dependências (PyTorch ≥2.0, torchvision ≥0.15, pydicom ≥2.4, opencv-python ≥4.8, numpy ≥1.24, pandas ≥2.0, scikit-learn ≥1.3, matplotlib ≥3.7, boto3 ≥1.28, sagemaker ≥2.170, hypothesis ≥6.80)
    - Criar `setup.py` ou `pyproject.toml` para o pacote
    - _Requirements: 9.1, 9.4, 9.5_

  - [ ] 1.2 Definir dataclasses e tipos base
    - Implementar em `src/models.py` as dataclasses: `PreprocessConfig`, `ProcessResult`, `BatchResult`, `RelatorioPre`, `ClassificacaoResult`, `ClassificacaoExame`, `LaudoInput`, `ExtractionBatchResult`, `SplitConfig`, `DataSplit`, `ImagemLabel`, `ImagemProcessada`, `Prediction`, `MetricasCompletas`, `TrainingConfig`, `TrainingResult`, `ComparacaoModelos`, `InferenceConfig`, `BatchInferenceResult`, `RelatorioComparativo`, `KeywordsConfig`, `LeakageReport`
    - Definir Literals para classes: `Literal["Normal", "Suspeito", "Grave"]`
    - Definir Literals para status: `Literal["classificado", "revisao_pendente", "rejeitado"]`
    - Incluir regras de validação conforme Data Models do design
    - _Requirements: 5.1, 2.1, 3.1_

  - [ ] 1.3 Implementar utilitários AWS (S3 e CloudWatch)
    - Criar `src/utils/aws_helpers.py` com funções para upload/download S3, listagem de arquivos por prefixo, escrita de métricas no CloudWatch
    - Configurar região us-east-1 como padrão
    - Implementar wrappers para operações S3 nos prefixos raw/, processed/, models/, results/
    - _Requirements: 9.2, 9.4, 9.5, 10.1_

- [ ] 2. Pipeline de pré-processamento de imagens DICOM
  - [ ] 2.1 Implementar conversão DICOM → PNG com CLAHE e normalização
    - Criar `src/preprocessing/pipeline.py` com classe `PipelinePreprocessamento`
    - Implementar `processar_imagem()`: decodificação pydicom → escala de cinza 8-bit → resize 512x512 com padding preto → CLAHE (clip_limit=2.0, tile_grid=8x8) → normalização [0,1] float32
    - Implementar `normalize_to_uint8()` auxiliar
    - Implementar tratamento de exceções para DICOM corrompido/inválido com logging
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ] 2.2 Implementar processamento em lote e geração de relatório
    - Implementar `processar_lote()` com processamento paralelo (multiprocessing, pool de 4 workers)
    - Implementar `gerar_relatorio()` retornando contagem de sucessos e falhas
    - Garantir continuidade do lote em caso de falha individual
    - _Requirements: 1.5, 1.6_

  - [ ]* 2.3 Write property test for integridade do pré-processamento
    - **Property 1: Integridade do Pré-processamento**
    - Testar que para toda imagem processada com sucesso: shape == (512, 512), dtype == float32, valores ∈ [0.0, 1.0]
    - Usar Hypothesis para gerar arrays numpy de dimensões variadas como entrada
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**

  - [ ]* 2.4 Write unit tests for pipeline de pré-processamento
    - Testar DICOM válido → PNG 512x512 correto
    - Testar DICOM corrompido → erro gracioso sem interrupção
    - Testar imagem não-quadrada → padding preto correto
    - Testar normalização: valores de saída em [0, 1]
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [ ] 3. Extrator de labels a partir de laudos estruturados
  - [ ] 3.1 Implementar classificação por palavras-chave
    - Criar `src/labeling/extrator.py` com classe `ExtratorLabels`
    - Definir listas de palavras-chave por severidade (KEYWORDS_GRAVE, KEYWORDS_SUSPEITO, KEYWORDS_NORMAL) conforme design
    - Implementar `classificar_laudo()`: busca case-insensitive, classificação por severidade máxima
    - Implementar rejeição de laudos com < 10 caracteres
    - Implementar marcação "revisão pendente" quando nenhuma keyword encontrada
    - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.6_

  - [ ] 3.2 Implementar agregação no nível do exame e processamento em lote
    - Implementar `agregar_exame()`: estratégia max-severity entre imagens do mesmo exame
    - Implementar `processar_lote()`: processamento de múltiplos laudos com estatísticas de distribuição
    - _Requirements: 2.3_

  - [ ]* 3.3 Write property test for severidade máxima na extração de labels
    - **Property 4: Severidade Máxima na Extração de Labels**
    - Testar que para laudos com múltiplas keywords de categorias diferentes, a classe resultante é a de maior severidade
    - Usar Hypothesis para gerar combinações de keywords de múltiplas categorias
    - **Validates: Requirements 2.1, 2.5**

  - [ ]* 3.4 Write property test for exclusividade de classificação do extrator
    - **Property 2 (parcial): Exclusividade de Classificação**
    - Testar que `classificar_laudo()` nunca retorna classe inválida: resultado ∈ {"Normal", "Suspeito", "Grave", None} e status ∈ {"classificado", "revisao_pendente", "rejeitado"}
    - Usar Hypothesis com `st.text()` para gerar strings arbitrárias
    - **Validates: Requirements 5.1, 2.1**

  - [ ]* 3.5 Write unit tests for extrator de labels
    - Testar laudo com keyword grave → classe Grave
    - Testar laudo vazio/curto → rejeitado
    - Testar múltiplas categorias → max severity
    - Testar laudo sem keywords → revisão pendente
    - Testar case-insensitivity
    - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.6_

- [ ] 4. Checkpoint - Validar componentes de dados
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Pipeline de divisão de dados com prevenção de data leakage
  - [ ] 5.1 Implementar divisão estratificada com group-based split
    - Criar `src/splitting/pipeline.py` com classe `PipelineDivisaoDados`
    - Implementar `dividir()`: agrupamento por exam_id, estratificação por classe, divisão 80/10/10 com seed fixa
    - Garantir todas as imagens de um exame no mesmo conjunto
    - Garantir proporção de cada classe ±5pp da proporção global
    - Implementar alertas para classe < 20 exemplos em val/test
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ] 5.2 Implementar verificação de leakage e exportação de manifesto
    - Implementar `verificar_leakage()`: checar interseção vazia entre conjuntos, interromper com erro se leakage detectado
    - Implementar `exportar_manifesto()`: CSV com caminho do arquivo e conjunto atribuído
    - _Requirements: 8.1, 8.2, 8.3, 3.6_

  - [ ]* 5.3 Write property test for prevenção de data leakage
    - **Property 3: Prevenção de Data Leakage**
    - Testar que para qualquer seed a interseção de exam_ids entre train/val/test é vazia
    - Usar Hypothesis com `st.integers()` para gerar seeds arbitrárias
    - **Validates: Requirements 8.1, 8.2, 8.3**

  - [ ]* 5.4 Write property test for reprodutibilidade da divisão
    - **Property 5: Reprodutibilidade da Divisão**
    - Testar que a mesma seed sempre produz exatamente a mesma divisão
    - Executar divisão duas vezes com mesma seed e verificar igualdade
    - **Validates: Requirements 3.1**

  - [ ]* 5.5 Write unit tests for pipeline de divisão
    - Testar divisão 80/10/10 com dataset pequeno
    - Testar que exames com múltiplas imagens ficam no mesmo conjunto
    - Testar estratificação dentro da tolerância ±5pp
    - Testar que leakage é detectado e execução interrompida
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 8.1, 8.2, 8.3_

- [ ] 6. Pipeline de treinamento SageMaker
  - [ ] 6.1 Implementar script de treinamento com fine-tuning em duas etapas
    - Criar `src/training/train.py` como entry point do SageMaker Training Job
    - Implementar carregamento de modelos pré-treinados (ResNet50, EfficientNet-B0 do torchvision)
    - Implementar Fase 1: backbone congelado (ResNet50: 10 épocas, EfficientNet: 15 épocas)
    - Implementar Fase 2: fine-tuning completo com LR ÷ 10
    - Implementar detecção de divergência (loss NaN × 3 épocas → interromper)
    - Implementar logging de métricas por época (val_f1, val_loss, train_loss) via print para metric_definitions
    - Implementar salvamento de model.tar.gz com pesos, config e métricas
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7_

  - [ ] 6.2 Implementar data augmentation exclusiva para treinamento
    - Criar `src/training/augmentation.py` com função `aplicar_augmentation()`
    - Implementar augmentation estocástica para conjunto train: flip horizontal, rotação (±15°), ajuste de brilho
    - Implementar transformação determinística para val/test: apenas resize e normalização
    - Implementar DataLoader com prefetch (4 workers) e batch_size=32
    - _Requirements: 8.4, 8.5_

  - [ ] 6.3 Implementar orquestração de treinamento e comparação de modelos
    - Criar `src/training/orchestrator.py` com classe `PipelineTreinamento`
    - Implementar `treinar_modelo()`: configura e lança SageMaker Training Job com hiperparâmetros
    - Implementar `comparar_modelos()`: comparação por F1-score de validação
    - Implementar `registrar_melhor_modelo()`: registro no SageMaker Experiments
    - Implementar integração com CloudWatch (metric_definitions) e SageMaker Experiments
    - _Requirements: 4.4, 4.6, 10.1, 10.3_

  - [ ]* 6.4 Write property test for augmentation exclusiva para treinamento
    - **Property 6: Augmentation Exclusiva para Treinamento**
    - Testar que `aplicar_augmentation(image, "validation")` e `aplicar_augmentation(image, "test")` são determinísticas (aplicar duas vezes produz resultado idêntico)
    - Usar Hypothesis para gerar arrays numpy 512x512 com valores [0,1]
    - **Validates: Requirements 8.4, 8.5**

  - [ ]* 6.5 Write unit tests for pipeline de treinamento
    - Testar congelamento/descongelamento de backbone
    - Testar detecção de divergência de loss (NaN × 3 épocas)
    - Testar que LR fase 2 = LR fase 1 / 10
    - Testar formato de saída model.tar.gz
    - _Requirements: 4.2, 4.3, 4.5, 4.7_

- [ ] 7. Checkpoint - Validar pipeline de treinamento
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Pipeline de inferência em lote (Batch Transform)
  - [ ] 8.1 Implementar execução de Batch Transform e geração de resultados
    - Criar `src/inference/pipeline.py` com classe `PipelineInferencia`
    - Implementar `executar_batch()`: configuração de Transformer (SingleRecord, max_payload=6MB, ml.g4dn.xlarge, content_type image/png)
    - Implementar script de inferência (`src/inference/inference.py`) que carrega modelo e gera JSON por imagem: {predicted_class, confidence, probabilities}
    - Implementar validação de formato de entrada (PNG 512x512 normalizado)
    - Implementar tratamento de imagens > 6MB (ignorar + log CloudWatch)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 5.6_

  - [ ] 8.2 Implementar cálculo de métricas de avaliação
    - Criar `src/inference/metrics.py` com função `calcular_metricas_completas()`
    - Implementar AUC-ROC por classe (One-vs-Rest) e macro-average
    - Implementar F1-Score weighted e macro
    - Implementar Precision e Recall por classe
    - Implementar geração de matriz de confusão 3x3
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ] 8.3 Implementar relatório comparativo e visualizações
    - Implementar `gerar_relatorio_comparativo()`: tabela com métricas lado a lado (ResNet50 vs EfficientNet-B0)
    - Implementar salvamento de relatório em JSON e visualização PNG (matplotlib)
    - Implementar salvamento no bucket S3 results/
    - _Requirements: 7.5, 7.6_

  - [ ]* 8.4 Write property test for exclusividade de classificação
    - **Property 2: Exclusividade de Classificação**
    - Testar que para toda predição: classe ∈ {Normal, Suspeito, Grave}, sum(probabilities) == 1.0 (±1e-6), predicted_class == argmax(probabilities)
    - Usar Hypothesis para gerar distribuições de probabilidade válidas
    - **Validates: Requirements 5.1, 5.2, 5.3**

  - [ ]* 8.5 Write property test for consistência da matriz de confusão
    - **Property 7: Consistência da Matriz de Confusão**
    - Testar que para toda matriz gerada, soma de cada linha == total de amostras da classe real
    - Usar Hypothesis para gerar listas de predictions e ground_truth
    - **Validates: Requirements 7.4**

  - [ ]* 8.6 Write unit tests for pipeline de inferência
    - Testar formato JSON de saída (classe, confiança, probabilidades)
    - Testar rejeição de imagem com formato inválido
    - Testar cálculo de métricas com dataset pequeno conhecido
    - Testar que matriz de confusão soma corretamente
    - _Requirements: 6.2, 5.6, 7.1, 7.2, 7.3, 7.4_

- [ ] 9. Infraestrutura AWS e monitoramento
  - [ ] 9.1 Implementar configuração IAM e políticas de acesso
    - Criar `infrastructure/iam_policy.json` com permissões mínimas: S3 (GetObject, PutObject nos buckets do projeto), SageMaker (CreateTrainingJob, CreateTransformJob, CreateNotebookInstance), CloudWatch (PutMetricData, PutLogEvents, GetLogEvents)
    - Criar script `infrastructure/setup_iam.py` para criação do usuário IAM dedicado com política restrita
    - Documentar requisito de MFA habilitado
    - _Requirements: 9.1, 9.6_

  - [ ] 9.2 Implementar setup de buckets S3 e logging estruturado
    - Criar script `infrastructure/setup_s3.py` para criação de bucket com SSE-S3, Block Public Access e estrutura de prefixos (raw/, processed/, models/, results/)
    - Implementar `src/utils/logging_config.py` com logging estruturado para CloudWatch
    - Configurar log groups para training jobs e batch transform jobs
    - Implementar registro de stack trace completo em falhas
    - _Requirements: 9.2, 9.5, 10.1, 10.2, 10.4, 10.5_

  - [ ] 9.3 Implementar integração com SageMaker Experiments
    - Criar `src/utils/experiments.py` com funções para criar/atualizar experiments, registrar hiperparâmetros e métricas, e rastrear artefatos
    - Integrar com pipeline de treinamento e comparação de modelos
    - _Requirements: 10.3_

- [ ] 10. Validação de escopo de modalidades e integração final
  - [ ] 10.1 Implementar validação de modalidade suportada
    - Criar `src/utils/validators.py` com função de validação de modalidade
    - Aceitar apenas radiografia de tórax com 1-3 imagens por exame
    - Rejeitar tomografia, densitometria óssea e mamografia com mensagem adequada
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ] 10.2 Implementar notebook de execução end-to-end
    - Criar `notebooks/poc_pipeline.ipynb` orquestrando todo o pipeline: pré-processamento → extração labels → divisão → treinamento → inferência → métricas
    - Integrar todos os componentes conforme exemplo de uso do design
    - Garantir que cada etapa valida resultados antes de avançar
    - _Requirements: 1.1-1.6, 2.1-2.6, 3.1-3.6, 4.1-4.7, 5.1-5.6, 6.1-6.5, 7.1-7.6_

- [ ] 11. Final checkpoint - Validação completa
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- O projeto utiliza Python como linguagem de implementação conforme definido no design
- Infraestrutura AWS na região us-east-1 conforme premissas de custo
- Biblioteca Hypothesis utilizada para property-based testing

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "3.1", "9.1", "9.2"] },
    { "id": 3, "tasks": ["2.2", "3.2", "9.3"] },
    { "id": 4, "tasks": ["2.3", "2.4", "3.3", "3.4", "3.5"] },
    { "id": 5, "tasks": ["5.1", "10.1"] },
    { "id": 6, "tasks": ["5.2"] },
    { "id": 7, "tasks": ["5.3", "5.4", "5.5"] },
    { "id": 8, "tasks": ["6.1", "6.2"] },
    { "id": 9, "tasks": ["6.3"] },
    { "id": 10, "tasks": ["6.4", "6.5"] },
    { "id": 11, "tasks": ["8.1"] },
    { "id": 12, "tasks": ["8.2", "8.3"] },
    { "id": 13, "tasks": ["8.4", "8.5", "8.6"] },
    { "id": 14, "tasks": ["10.2"] }
  ]
}
```
