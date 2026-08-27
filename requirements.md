# Requirements Document

## Introduction

Este documento define os requisitos para a Prova de Conceito (PoC) de Inteligência Artificial aplicada à triagem automática de exames de imagem para a empresa KD Diagnósticos. A solução utiliza visão computacional para classificar radiografias de tórax por nível de prioridade (Normal, Suspeito, Grave), apoiando o fluxo de trabalho dos radiologistas na etapa de pré-laudo. A PoC será executada em infraestrutura AWS (S3, SageMaker, CloudWatch, IAM) com duração estimada de 6 semanas.

## Glossary

- **Sistema_Classificacao**: Modelo de visão computacional responsável por classificar imagens de radiografia de tórax em três níveis de prioridade
- **Pipeline_Preprocessamento**: Componente responsável por converter imagens DICOM em formato adequado para treinamento e inferência
- **Extrator_Labels**: Componente responsável por extrair rótulos de classificação a partir dos laudos estruturados existentes
- **Pipeline_Treinamento**: Componente que executa o treinamento e fine-tuning dos modelos de classificação no SageMaker
- **Pipeline_Inferencia**: Componente que executa inferência em lote (Batch Transform) para validação dos modelos
- **Normal**: Classificação indicando exame sem achados patológicos relevantes (cor Azul)
- **Suspeito**: Classificação indicando possíveis alterações que requerem atenção do especialista (cor Amarela)
- **Grave**: Classificação indicando achados que sugerem condição crítica ou prioritária (cor Vermelha)
- **Exame**: Conjunto de uma ou mais imagens radiológicas de um mesmo paciente e mesma incidência
- **Laudo_Estruturado**: Relatório médico pré-formatado que segue template padronizado (compêndio radiológico)
- **Fine_Tuning**: Técnica de treinamento que adapta um modelo pré-treinado para o domínio específico de radiologia
- **Batch_Transform**: Modo de inferência em lote do SageMaker, sem necessidade de endpoint real-time
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization — técnica de equalização de histograma para melhorar contraste em imagens médicas
- **Data_Leakage**: Vazamento de dados entre conjuntos de treinamento e validação que compromete a avaliação do modelo

## Requirements

### Requisito 1: Pré-processamento de Imagens DICOM

**User Story:** Como cientista de dados, eu quero que imagens DICOM sejam convertidas e padronizadas automaticamente, para que possam ser utilizadas como entrada nos modelos de classificação.

#### Acceptance Criteria

1. WHEN uma imagem DICOM de radiografia de tórax é fornecida, THE Pipeline_Preprocessamento SHALL converter a imagem para formato PNG em escala de cinza 8-bit
2. WHEN uma imagem é convertida, THE Pipeline_Preprocessamento SHALL redimensionar a imagem para resolução de 512x512 pixels, aplicando padding preto para preservar a proporção original caso a imagem não seja quadrada
3. WHEN uma imagem é redimensionada, THE Pipeline_Preprocessamento SHALL aplicar equalização CLAHE com clip limit de 2.0 e tile grid de 8x8 pixels
4. WHEN o CLAHE é aplicado, THE Pipeline_Preprocessamento SHALL normalizar os valores de pixel para o intervalo [0, 1] em formato float32
5. IF uma imagem DICOM não puder ser decodificada pela biblioteca pydicom (arquivo corrompido, cabeçalho DICOM ausente ou dados de pixel inválidos), THEN THE Pipeline_Preprocessamento SHALL registrar um erro em log contendo o nome do arquivo e a descrição da falha, e continuar o processamento das demais imagens
6. WHEN o pré-processamento de um lote é concluído, THE Pipeline_Preprocessamento SHALL gerar um relatório indicando a quantidade total de imagens processadas com sucesso e a quantidade de imagens que falharam

### Requisito 2: Extração de Rótulos a Partir de Laudos Estruturados

**User Story:** Como cientista de dados, eu quero extrair automaticamente rótulos de classificação dos laudos existentes, para que eu tenha dados rotulados para treinamento sem necessidade de anotação manual.

#### Acceptance Criteria

1. WHEN um laudo estruturado é fornecido como entrada de texto, THE Extrator_Labels SHALL classificar o laudo em exatamente uma das três categorias mutuamente exclusivas: Normal, Suspeito ou Grave, aplicando a categoria de maior severidade entre todas as palavras-chave encontradas (Grave > Suspeito > Normal)
2. THE Extrator_Labels SHALL utilizar correspondência de palavras-chave case-insensitive e por correspondência exata de termos contra as listas de palavras-chave definidas no documento de design para identificar achados no laudo estruturado
3. WHEN um exame possui múltiplas imagens com laudos individuais, THE Extrator_Labels SHALL agregar os rótulos no nível do exame utilizando a estratégia de severidade máxima (max-severity), atribuindo ao exame o rótulo de maior gravidade entre todas as imagens do exame
4. IF nenhuma palavra-chave de qualquer categoria (Normal, Suspeito ou Grave) for encontrada no laudo, THEN THE Extrator_Labels SHALL marcar o registro com o status "revisão pendente" e excluí-lo do conjunto de dados de treinamento até revisão manual
5. IF um laudo contiver palavras-chave de múltiplas categorias de severidade, THEN THE Extrator_Labels SHALL atribuir a categoria de maior gravidade (Grave > Suspeito > Normal) como rótulo final do laudo individual
6. IF o texto do laudo estiver vazio ou contiver menos de 10 caracteres, THEN THE Extrator_Labels SHALL rejeitar o registro com indicação de erro informando que o laudo é insuficiente para classificação

### Requisito 3: Separação de Dados para Treinamento e Validação

**User Story:** Como cientista de dados, eu quero garantir que a divisão dos dados entre treinamento e validação seja feita corretamente, para que a avaliação do modelo reflita performance real em dados não vistos.

#### Acceptance Criteria

1. THE Pipeline_Treinamento SHALL dividir o dataset nas proporções de 80% para treinamento, 10% para validação e 10% para teste, utilizando uma seed fixa configurável para garantir reprodutibilidade (mesma seed produz a mesma divisão)
2. WHEN a divisão de dados é realizada, THE Pipeline_Treinamento SHALL manter todas as imagens de um mesmo exame no mesmo conjunto (train, validation ou test)
3. WHEN a divisão de dados é realizada, THE Pipeline_Treinamento SHALL garantir que nenhuma imagem de um paciente presente no conjunto de treinamento apareça no conjunto de validação ou teste
4. WHEN a divisão de dados é realizada, THE Pipeline_Treinamento SHALL aplicar estratificação por classe de diagnóstico, de modo que a proporção de cada classe em cada conjunto não se desvie mais de 5 pontos percentuais da proporção original no dataset completo
5. IF o número de exemplos de uma classe for inferior a 20 no conjunto de validação ou teste, THEN THE Pipeline_Treinamento SHALL reportar um alerta contendo o nome da classe afetada, o conjunto (validação ou teste), e a quantidade de exemplos encontrada
6. WHEN a divisão é concluída, THE Pipeline_Treinamento SHALL exportar um manifesto em formato CSV contendo, para cada imagem, o caminho do arquivo e o conjunto atribuído (train, validation ou test)

### Requisito 4: Treinamento de Modelos com Fine-Tuning

**User Story:** Como cientista de dados, eu quero treinar modelos de classificação com fine-tuning em duas etapas, para que eu possa comparar arquiteturas e selecionar a melhor para o domínio de radiologia de tórax.

#### Acceptance Criteria

1. THE Pipeline_Treinamento SHALL treinar modelos utilizando as arquiteturas ResNet50 e EfficientNet-B0 pré-treinadas no ImageNet
2. WHEN o treinamento é iniciado, THE Pipeline_Treinamento SHALL executar a primeira etapa com o backbone congelado (frozen), treinando apenas as camadas de classificação durante o número de épocas configurado por arquitetura (ResNet50: 10 épocas, EfficientNet-B0: 15 épocas)
3. WHEN a primeira etapa é concluída, THE Pipeline_Treinamento SHALL executar a segunda etapa com fine-tuning completo (todas as camadas descongeladas) com learning rate reduzido por um fator de 10 em relação ao learning rate da primeira etapa
4. THE Pipeline_Treinamento SHALL registrar as métricas de treinamento e validação (loss, accuracy e F1-score por época) no CloudWatch e no SageMaker Experiments
5. WHEN o treinamento é concluído, THE Pipeline_Treinamento SHALL salvar os artefatos do modelo no S3 em formato model.tar.gz contendo o arquivo de pesos do modelo, o arquivo de configuração e o arquivo de métricas
6. WHEN o treinamento de todas as arquiteturas é concluído, THE Pipeline_Treinamento SHALL comparar os modelos com base no F1-score de validação e registrar o modelo com maior F1-score como melhor modelo no SageMaker Experiments
7. IF o treinamento de um modelo falhar por erro de memória ou divergência de loss (loss = NaN por 3 épocas consecutivas), THEN THE Pipeline_Treinamento SHALL interromper o treinamento do modelo afetado, registrar o erro no CloudWatch e prosseguir com o treinamento dos demais modelos

### Requisito 5: Classificação em Três Níveis de Prioridade

**User Story:** Como radiologista da KD, eu quero que as imagens sejam classificadas em três níveis de prioridade com cores, para que eu possa priorizar meu fluxo de trabalho.

#### Acceptance Criteria

1. WHEN uma imagem de radiografia de tórax é submetida ao modelo, THE Sistema_Classificacao SHALL retornar exatamente uma das três classificações mutuamente exclusivas: Normal (Azul), Suspeito (Amarelo) ou Grave (Vermelho)
2. WHEN uma classificação é realizada, THE Sistema_Classificacao SHALL retornar as probabilidades associadas a cada uma das três classes, onde a soma das probabilidades é igual a 1.0 (±1e-6)
3. WHEN uma classificação é realizada, THE Sistema_Classificacao SHALL selecionar a classe com maior probabilidade como classe predita (argmax da distribuição softmax)
4. THE Sistema_Classificacao SHALL operar como classificação de imagem inteira (whole-image classification), sem localização de achados
5. THE Sistema_Classificacao SHALL processar imagens de radiografia de tórax como modalidade primária
6. IF a imagem de entrada não estiver no formato esperado (PNG 512x512 normalizado), THEN THE Sistema_Classificacao SHALL rejeitar a imagem e retornar erro indicando o formato recebido versus o formato esperado

### Requisito 6: Inferência em Lote (Batch Transform)

**User Story:** Como cientista de dados, eu quero executar inferência em lote para validar o modelo contra o dataset de teste, para que eu possa avaliar a performance sem necessidade de endpoint real-time.

#### Acceptance Criteria

1. WHEN um job de Batch Transform é executado, THE Pipeline_Inferencia SHALL processar todas as imagens PNG do dataset de validação ou teste utilizando estratégia SingleRecord com content_type "image/png" e max_payload de 6 MB
2. WHEN a inferência de uma imagem é concluída, THE Pipeline_Inferencia SHALL gravar um resultado em formato JSON no bucket S3 de resultados contendo: classe predita (predicted_class), confiança da predição (confidence entre 0.0 e 1.0), e probabilidades por classe
3. THE Pipeline_Inferencia SHALL utilizar instância ml.g4dn.xlarge para execução do Batch Transform
4. IF um job de Batch Transform falhar, THEN THE Pipeline_Inferencia SHALL registrar o erro no CloudWatch com detalhes da falha incluindo identificação do job, timestamp e motivo da falha
5. IF uma imagem de entrada exceder 6 MB de tamanho, THEN THE Pipeline_Inferencia SHALL ignorar a imagem e registrar um aviso no CloudWatch indicando o nome do arquivo que excedeu o limite

### Requisito 7: Avaliação de Performance com Métricas Definidas

**User Story:** Como stakeholder do projeto, eu quero avaliar a performance dos modelos com métricas objetivas, para que eu possa decidir se a PoC demonstra viabilidade tecnológica.

#### Acceptance Criteria

1. WHEN a inferência em lote é concluída, THE Pipeline_Inferencia SHALL calcular AUC-ROC para cada uma das três classes (Normal, Suspeito, Grave) e para o modelo geral utilizando macro-average, com valores no intervalo [0.0, 1.0]
2. WHEN a inferência em lote é concluída, THE Pipeline_Inferencia SHALL calcular F1-Score ponderado (weighted) e macro para o modelo, com valores no intervalo [0.0, 1.0]
3. WHEN a inferência em lote é concluída, THE Pipeline_Inferencia SHALL calcular precisão (precision) e revocação (recall) para cada classe individualmente (Normal, Suspeito, Grave)
4. THE Pipeline_Inferencia SHALL gerar uma matriz de confusão de dimensão 3x3 onde a soma de cada linha é igual ao total de amostras da respectiva classe real
5. WHEN dois modelos são treinados (ResNet50 e EfficientNet-B0), THE Pipeline_Inferencia SHALL produzir um relatório comparativo em formato tabular com todas as métricas (AUC-ROC, F1, precision, recall por classe) lado a lado para cada modelo
6. THE Pipeline_Inferencia SHALL salvar o relatório de métricas e a matriz de confusão no bucket S3 de resultados em formato JSON e como visualização gráfica (PNG)

### Requisito 8: Prevenção de Data Leakage

**User Story:** Como cientista de dados, eu quero garantir que não haja vazamento de informação entre conjuntos de dados, para que as métricas de avaliação sejam confiáveis e representativas.

#### Acceptance Criteria

1. THE Pipeline_Treinamento SHALL utilizar o identificador do exame como unidade de agrupamento na divisão de dados (group-based split), garantindo que todas as imagens pertencentes ao mesmo identificador de exame sejam atribuídas ao mesmo conjunto
2. WHEN a divisão é realizada, THE Pipeline_Treinamento SHALL verificar que a interseção de identificadores de exame entre quaisquer dois conjuntos (treinamento, validação, teste) é vazia
3. IF a verificação de divisão detectar pelo menos um identificador de exame presente em mais de um conjunto, THEN THE Pipeline_Treinamento SHALL interromper a execução e emitir mensagem de erro indicando os identificadores duplicados e os conjuntos afetados
4. WHEN data augmentation é aplicada, THE Pipeline_Treinamento SHALL aplicar transformações de augmentation exclusivamente nas imagens do conjunto de treinamento
5. WHILE o pipeline processa imagens dos conjuntos de validação ou teste, THE Pipeline_Treinamento SHALL aplicar apenas transformações determinísticas (redimensionamento e normalização), sem qualquer transformação de augmentation

### Requisito 9: Infraestrutura AWS com Controle de Acesso

**User Story:** Como gerente de projeto, eu quero que a infraestrutura AWS seja configurada com controles de acesso adequados, para que o projeto seja seguro e os recursos possam ser desativados após a PoC.

#### Acceptance Criteria

1. THE Pipeline_Treinamento SHALL utilizar um usuário IAM dedicado com permissões restritas exclusivamente às ações necessárias nos serviços S3 (leitura e escrita nos buckets do projeto), SageMaker (criação e gerenciamento de notebooks, training jobs e batch transform) e CloudWatch (escrita e leitura de logs), sem acesso a outros serviços ou recursos da conta AWS
2. THE Pipeline_Treinamento SHALL armazenar todas as imagens e artefatos em buckets S3 com criptografia habilitada (SSE-S3) e com acesso restrito exclusivamente ao usuário IAM dedicado do projeto
3. WHEN a PoC é concluída, THE Pipeline_Treinamento SHALL permitir a desativação do usuário IAM (desabilitação de credenciais de acesso) sem impacto em outros recursos da conta AWS, mantendo os dados nos buckets S3 acessíveis por outros usuários autorizados da conta
4. THE Pipeline_Treinamento SHALL operar na região us-east-1 conforme definido nas premissas de custo, com todos os recursos (buckets S3, instâncias SageMaker, logs CloudWatch) provisionados nesta mesma região
5. THE Pipeline_Treinamento SHALL organizar os dados em buckets S3 utilizando os prefixos raw/, processed/, models/ e results/ para separação lógica de imagens originais, imagens pré-processadas, artefatos de modelo e resultados de inferência
6. THE Pipeline_Treinamento SHALL exigir autenticação multifator (MFA) habilitada na conta AWS utilizada para o projeto

### Requisito 10: Monitoramento e Logging

**User Story:** Como cientista de dados, eu quero que todos os jobs de treinamento e inferência gerem logs estruturados, para que eu possa debugar falhas e acompanhar o progresso dos experimentos.

#### Acceptance Criteria

1. WHEN um Training Job é executado, THE Pipeline_Treinamento SHALL enviar logs estruturados para o CloudWatch contendo, ao final de cada época, as métricas val_f1, val_loss e train_loss extraídas via metric_definitions
2. WHEN um Batch Transform Job é executado, THE Pipeline_Inferencia SHALL enviar logs de execução para o CloudWatch contendo timestamp de início e fim do job, quantidade total de registros processados e quantidade de registros com erro
3. THE Pipeline_Treinamento SHALL utilizar SageMaker Experiments para rastrear hiperparâmetros, métricas (val_f1, val_loss, train_loss) e artefatos (modelo treinado) de cada execução de treinamento
4. IF um job de treinamento falhar, THEN THE Pipeline_Treinamento SHALL registrar no CloudWatch o stack trace completo, o nome do job, o timestamp da falha e uma mensagem de erro indicando a causa da exceção
5. IF um Batch Transform Job falhar, THEN THE Pipeline_Inferencia SHALL registrar no CloudWatch o nome do job, o timestamp da falha, o stack trace completo e a quantidade de registros processados antes da falha

### Requisito 11: Escopo de Modalidades

**User Story:** Como gerente de projeto, eu quero que o escopo de modalidades seja claramente delimitado, para que a equipe foque nos exames prioritários e a PoC seja entregue no prazo.

#### Acceptance Criteria

1. THE Sistema_Classificacao SHALL aceitar e classificar exclusivamente exames de radiografia de tórax contendo entre 1 e 3 imagens por exame como modalidade suportada nesta PoC
2. IF um exame de modalidade não suportada (tomografia, densitometria óssea ou mamografia) for submetido, THEN THE Sistema_Classificacao SHALL rejeitar o exame e apresentar mensagem indicando que a modalidade não é suportada nesta versão
3. THE Sistema_Classificacao SHALL excluir tomografia do escopo desta PoC devido ao volume de imagens por exame (400+ imagens)
4. THE Sistema_Classificacao SHALL excluir densitometria óssea do escopo desta PoC
5. THE Sistema_Classificacao SHALL registrar mamografia como modalidade secundária planejada para fase posterior, sem implementar funcionalidade de classificação para esta modalidade na PoC atual
