# PoC — Arquitetura AWS para IA em Exames de Imagem
## KD Diagnósticos | Cenário Mediano Realista

---

## 1. Requisitos Extraídos das Transcrições

### 1.1 Objetivo da Solução
Desenvolver um modelo de visão computacional para **triagem automática (pré-laudo)** de exames de imagem, classificando-os por nível de prioridade (normal, suspeito, grave) para otimizar o fluxo de trabalho dos radiologistas.

### 1.2 Modalidades de Exames

| Modalidade | Imagens/Exame | Incluída na PoC? |
|---|---|---|
| Radiografia (Raio X) de Tórax | 2–3 | **Sim** |
| Mamografia | 4–8 | Sim (secundária) |
| Densitometria Óssea | 2–3 | Mencionada, mas não priorizada |
| Tomografia | 400+ | **Não** (explicitamente excluída) |

> **Decisão registrada na Reunião 2:** Tomografia NÃO entra neste primeiro momento. Foco será radiografia e mamografia.

### 1.3 Volume Atual e Futuro (mencionado por Danilo)

| Métrica | Valor | Fonte |
|---|---|---|
| Volume atual mensal (RX + Mamografia) | ~1.000 exames/mês | Confirmado (Transcrição 2, 8:51) |
| Volume futuro esperado (com novas negociações) | 3.000–4.000 exames/mês | Mencionado como projeção |

### 1.4 Dados Disponíveis para Treinamento

| Info | Valor | Fonte |
|---|---|---|
| Banco total de imagens | >10.000 imagens | Confirmado |
| % positivos estimado (tórax) | ~30% anormais (pneumonia etc.) | Estimativa do Danilo |
| % normais | ~50% | Confirmado |
| % graves (cardíacos etc.) | ~20% | Confirmado |
| Amostra mínima para iniciar | 50–100 casos positivos | Recomendação do Davi |
| Laudos estruturados? | Sim (templates/compêndio) | Confirmado |
| Rótulos de localização (bounding boxes)? | Não existem ainda | Confirmado |

### 1.5 Escopo da PoC

- **Classificação geral** da imagem (normal vs. patológica) — sem localização.
- Foco inicial em **radiografia de tórax**.
- ~100 casos positivos como ponto de partida.
- Modelo será de **classificação de imagem** (não detecção de objetos).
- Pode testar múltiplos modelos e comparar performance.
- PoC sem custo para a KD (subsidiada por AWS + Update).
- Objetivo: validar viabilidade tecnológica.

### 1.6 Fluxo Esperado de Inferência (produtivo)

```
Imagem DICOM → IA classifica → Resultado (Normal / Suspeito / Grave) → Médico valida → Laudo final assinado
```

Sistema de cores mencionado: Azul (normal), Amarelo (suspeito), Vermelho (grave).

### 1.7 Requisitos Técnicos Mencionados

- Acesso via plataforma online (imagens + laudos).
- Conta AWS será criada especificamente para o projeto.
- Usuário IAM dedicado, desativável após PoC.
- Laudos seguem template estruturado (facilita extração de labels).
- Variabilidade nas imagens (calibração, digital vs CR, dosagem de radiação).
- Histórico clínico pode ser utilizado como feature auxiliar.

---

## 2. Separação: Confirmado vs. Inferido vs. Desconhecido

### ✅ Informações Confirmadas

- Foco: radiografia de tórax (e mamografia secundariamente)
- Volume: ~1.000 exames/mês atualmente
- Dados: >10.000 imagens históricas disponíveis
- Laudos estruturados existem e estão associados às imagens
- 50–100 casos positivos como amostra mínima
- Classificação geral (sem bounding boxes na PoC)
- Imagens disponíveis em plataforma online
- PoC subsidiada (sem custo para KD)
- Conta AWS nova será criada

### 🔶 Valores Inferidos

- Formato das imagens: provavelmente DICOM (padrão radiológico)
- Tamanho médio por imagem: DICOM de RX tórax tipicamente 10–15 MB
- Para treinamento, imagens convertidas a PNG/JPEG ~1–3 MB
- Estratégia de PoC: classificação binária ou multiclasse simples
- Duração da PoC: 4–8 semanas (inferido do ritmo semanal de status reports)

### ❓ Valores Desconhecidos

- Tamanho exato das imagens em disco
- Formato de exportação da plataforma
- Se haverá necessidade de pré-processamento pesado (normalização, windowing)
- Resolução alvo para treinamento
- Quantidade exata de classes de classificação
- Métricas de sucesso (acurácia alvo)
- Se a inferência na PoC será batch ou real-time
- Região AWS desejada

---

## 3. Tabela de Premissas — PoC Cenário Mediano Realista

| Parâmetro | Valor Adotado | Justificativa |
|---|---|---|
| Exames para treinamento | 2.000 imagens (do banco de 10k+) | PREMISSA — subconjunto curado de tórax com labels verificados |
| Imagens/exame (RX tórax) | 2 | Confirmado |
| Total de imagens de treinamento | 4.000 | 2.000 exames × 2 imagens |
| Tamanho médio por imagem (pré-processada) | 2 MB | PREMISSA — PNG redimensionado ~512×512 ou 1024×1024 |
| Armazenamento de dados de treinamento | ~8 GB | 4.000 × 2 MB |
| Imagens originais (backup DICOM) | ~60 GB | PREMISSA — 4.000 × 15 MB |
| Armazenamento total S3 | ~70 GB | Originais + processados + artefatos |
| Instância de treinamento | ml.g4dn.xlarge (1 GPU T4, 16 GB GPU RAM) | PREMISSA — suficiente para fine-tuning de modelos como ResNet/EfficientNet com batch moderado |
| Número de treinamentos na PoC | 10 | PREMISSA — iterações de experimentação |
| Duração por treinamento | 2 horas | PREMISSA — fine-tuning com ~4k imagens, 30–50 epochs |
| Horas de treinamento total | 20 horas | 10 × 2h |
| Instância de notebook (desenvolvimento) | ml.t3.medium | PREMISSA — para EDA, pré-processamento, avaliação |
| Horas de notebook/mês | 160 horas (horário comercial, ~1 mês) | PREMISSA |
| Inferência na PoC | Batch (não real-time) | PREMISSA — PoC não precisa de endpoint persistente |
| Batch Transform (inferência) | ml.g4dn.xlarge, 2h total | PREMISSA — avaliar modelo em dataset de validação |
| Transferência de dados | ~80 GB (upload inicial) | Upload das imagens para S3 |
| Região AWS | us-east-1 | PREMISSA — menor custo, créditos AWS geralmente mais amplos |
| Duração da PoC | 6 semanas | PREMISSA |
| Logs (CloudWatch) | ~5 GB | PREMISSA |

---

## 4. Arquitetura AWS Proposta — PoC

### 4.1 Diagrama de Fluxo

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FLUXO DA PoC                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Plataforma KD] ──upload──► [S3 - Raw Bucket]                      │
│                                    │                                 │
│                                    ▼                                 │
│                         [SageMaker Notebook]                         │
│                         (EDA + Pré-processamento)                    │
│                                    │                                 │
│                                    ▼                                 │
│                          [S3 - Processed Bucket]                     │
│                          (imagens normalizadas + manifest)           │
│                                    │                                 │
│                                    ▼                                 │
│                      [SageMaker Training Job]                        │
│                      (ml.g4dn.xlarge × 1)                            │
│                                    │                                 │
│                                    ▼                                 │
│                        [S3 - Model Artifacts]                        │
│                                    │                                 │
│                                    ▼                                 │
│                    [SageMaker Batch Transform]                        │
│                    (Inferência em lote para validação)                │
│                                    │                                 │
│                                    ▼                                 │
│                         [S3 - Results Bucket]                        │
│                         (predições + métricas)                       │
│                                                                      │
│  [CloudWatch] ← logs de todos os jobs                                │
│  [IAM] ← controle de acesso                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Serviços AWS — Justificativa e Custos

---

#### 4.2.1 Amazon S3 (Simple Storage Service)

**Por que é necessário:**
Armazenamento central de todas as imagens (raw e processadas), artefatos de modelo e resultados de inferência.

**Como participa do fluxo:**
- Bucket `raw/`: recebe imagens originais da plataforma KD
- Bucket `processed/`: imagens normalizadas para treinamento
- Bucket `models/`: artefatos dos modelos treinados
- Bucket `results/`: saídas de inferência

**O que gera custo:**
- Armazenamento (GB-mês)
- Requisições PUT/GET
- Transferência de dados (egress)

**Parâmetros para a calculadora:**
| Parâmetro | Valor |
|---|---|
| Storage class | S3 Standard |
| Armazenamento | 70 GB |
| PUT requests/mês | 10.000 |
| GET requests/mês | 50.000 |
| Data Transfer Out | 5 GB (mínimo, apenas download de resultados) |

---

#### 4.2.2 Amazon SageMaker — Notebook Instance

**Por que é necessário:**
Ambiente de desenvolvimento para exploração de dados, pré-processamento, prototipação de modelos e avaliação de resultados.

**Como participa do fluxo:**
- Análise exploratória das imagens e labels
- Scripts de pré-processamento (resize, normalização)
- Criação do manifest de treinamento
- Análise de resultados pós-treinamento

**O que gera custo:**
- Horas de instância ligada

**Parâmetros para a calculadora:**
| Parâmetro | Valor |
|---|---|
| Tipo de instância | ml.t3.medium |
| Horas/mês | 160 h |
| Armazenamento EBS do notebook | 50 GB (gp3) |
| Duração | 2 meses |

---

#### 4.2.3 Amazon SageMaker — Training Jobs

**Por que é necessário:**
Execução dos treinamentos de modelos de classificação de imagem com GPU.

**Como participa do fluxo:**
- Recebe dados do S3 (processed)
- Executa treinamento (fine-tuning de modelos pré-treinados)
- Salva modelo treinado no S3

**O que gera custo:**
- Horas de instância de treinamento (cobrado por segundo)

**Parâmetros para a calculadora:**
| Parâmetro | Valor |
|---|---|
| Tipo de instância | ml.g4dn.xlarge |
| Total de horas de treinamento | 20 h |
| Número de jobs | 10 |
| Duração média por job | 2 h |

---

#### 4.2.4 Amazon SageMaker — Batch Transform

**Por que é necessário:**
Inferência em lote para avaliação do modelo contra o dataset de validação. Não há necessidade de endpoint real-time numa PoC.

**Como participa do fluxo:**
- Carrega modelo treinado do S3
- Processa batch de imagens de validação
- Grava predições no S3

**O que gera custo:**
- Horas de instância de inferência

**Parâmetros para a calculadora:**
| Parâmetro | Valor |
|---|---|
| Tipo de instância | ml.g4dn.xlarge |
| Total de horas | 4 h (2 sessões de validação de ~2h) |

---

#### 4.2.5 Amazon CloudWatch

**Por que é necessário:**
Captura de logs dos training jobs, do notebook e dos batch transforms. Necessário para debug e monitoramento durante a PoC.

**Como participa do fluxo:**
- Recebe logs automaticamente do SageMaker
- Permite troubleshooting de falhas de treinamento

**O que gera custo:**
- Ingestão de logs (GB)
- Armazenamento de logs (GB-mês)

**Parâmetros para a calculadora:**
| Parâmetro | Valor |
|---|---|
| Ingestão de logs | 5 GB total |
| Armazenamento de logs | 5 GB por 2 meses |

---

#### 4.2.6 AWS IAM + VPC (custo zero)

**Por que é necessário:**
Controle de acesso (requisito mencionado: usuário IAM dedicado, desativável). VPC padrão.

**Custo:** $0 (sem custo direto).

---

### 4.3 Serviços DESCARTADOS (e por quê)

| Serviço | Motivo para não incluir |
|---|---|
| SageMaker Real-time Endpoint | PoC não precisa de inferência real-time; batch é suficiente para validar acurácia |
| Amazon Rekognition | Serviço genérico, não otimizado para imagens médicas DICOM |
| AWS Lambda | Não há fluxo event-driven na PoC; processamento é manual/batch |
| Amazon ECR | Pode ser usado se container custom for necessário, mas SageMaker built-in containers cobrem a necessidade inicialmente |
| AWS Step Functions | Orquestração desnecessária para PoC com poucos steps manuais |
| Amazon RDS/DynamoDB | Não há necessidade de banco de dados; metadados ficam em manifests no S3 |
| AWS Glue/Athena | Volume insuficiente para justificar ETL automatizado |
| Amazon HealthLake | Fora do escopo de PoC; voltado para dados FHIR |

---

## 5. Parâmetros Consolidados para AWS Pricing Calculator

| # | Serviço | Configuração | Quantidade | Período |
|---|---|---|---|---|
| 1 | S3 Standard | 70 GB armazenamento | 70 GB | 2 meses |
| 2 | S3 Requests | PUT: 10k, GET: 50k | por mês | 2 meses |
| 3 | S3 Data Transfer Out | 5 GB | por mês | 2 meses |
| 4 | SageMaker Notebook | ml.t3.medium, 160h/mês | 320 h total | 2 meses |
| 5 | SageMaker Training | ml.g4dn.xlarge | 20 h total | — |
| 6 | SageMaker Batch Transform | ml.g4dn.xlarge | 4 h total | — |
| 7 | CloudWatch Logs Ingestion | 5 GB | 5 GB | — |
| 8 | CloudWatch Logs Storage | 5 GB | 2 meses | — |
| 9 | EBS (notebook) | gp3, 50 GB | 50 GB | 2 meses |

### Região: us-east-1 (N. Virginia)

---

## 6. Estimativa de Ordem de Grandeza (referência)

> ⚠️ Os preços abaixo são estimativas de referência com base em preços públicos on-demand da AWS (us-east-1). Devem ser validados na AWS Pricing Calculator.

| Serviço | Cálculo | Estimativa Mensal |
|---|---|---|
| S3 (70 GB) | 70 × $0.023 | ~$1.61 |
| S3 Requests | (10k PUT × $0.005/1k) + (50k GET × $0.0004/1k) | ~$0.07 |
| SageMaker Notebook (ml.t3.medium, 160h) | 160 × $0.0582 | ~$9.31 |
| SageMaker Training (ml.g4dn.xlarge, 20h total) | 20 × $0.7364 | ~$14.73* |
| SageMaker Batch Transform (ml.g4dn.xlarge, 4h) | 4 × $0.7364 | ~$2.95* |
| EBS 50 GB gp3 | 50 × $0.08 | ~$4.00 |
| CloudWatch Logs (5 GB ingest) | 5 × $0.50 | ~$2.50 |
| CloudWatch Logs Storage (5 GB) | 5 × $0.03 | ~$0.15 |
| **TOTAL estimado (2 meses de PoC)** | | **~$70–$90** |

*Treinamento e Batch Transform são custos únicos distribuídos ao longo da PoC.*

**Ordem de grandeza: < $100 para toda a PoC** (cenário mediano, sem endpoint real-time, sem spot instances).

> Com Spot Instances para treinamento (desconto típico de 60–70%), o custo de training cairia para ~$5–6. Custo total da PoC poderia ficar em torno de $55–$70.

---

## 7. Pontos de Maior Incerteza

| # | Ponto | Impacto Potencial |
|---|---|---|
| 1 | **Tamanho real das imagens** — se forem DICOM full-resolution (>20 MB cada), armazenamento S3 pode dobrar | Baixo impacto em custo ($1–2 a mais) |
| 2 | **Número de iterações de treinamento** — se a equipe precisar de 30+ experimentos em vez de 10 | Pode adicionar ~$15–$20 |
| 3 | **Necessidade de endpoint real-time** para demonstração ao cliente | Adiciona $0.74/h; se ficar ligado 8h/dia × 20 dias = ~$118/mês — SIGNIFICATIVO |
| 4 | **Pré-processamento pesado** (conversão DICOM→PNG com windowing) pode exigir instância maior no notebook | Impacto moderado ($10–20/mês) |
| 5 | **Região AWS** — se usar São Paulo (sa-east-1), preços são ~20–30% maiores | Impacto moderado |
| 6 | **Transferência de dados inicial** — upload das imagens pode demorar dependendo da conexão, mas não gera custo AWS (ingress é grátis) | Zero impacto financeiro |
| 7 | **Múltiplas modalidades** — se incluir mamografia na PoC, dobra o dataset e treinamentos | Pode dobrar custos de treinamento |

---

## 8. Recomendações

1. **Usar Spot Instances para training jobs** — reduz custo de GPU em ~70%.
2. **Desligar notebook fora do horário** — SageMaker cobra por hora ligada.
3. **Começar apenas com RX de tórax** — menor complexidade, valida pipeline inteiro.
4. **Evitar endpoint real-time na PoC** — é o item que mais impacta custo se mantido ligado. Batch Transform resolve a necessidade de validação.
5. **Lifecycle configuration no notebook** — configurar auto-stop após 1h de inatividade.
6. **Usar SageMaker Experiments** — para tracking de métricas sem custo adicional significativo.

---

## 9. Resumo Executivo

| Item | Valor |
|---|---|
| Custo estimado total da PoC (2 meses) | **$70–$90 (on-demand) / $55–$70 (com spot)** |
| Principal driver de custo | SageMaker Training (GPU) |
| Risco de custo | Endpoint real-time ligado continuamente |
| Complexidade da arquitetura | Baixa (4 serviços core) |
| Serviços AWS necessários | S3, SageMaker (Notebook + Training + Batch Transform), CloudWatch, IAM |

---

*Documento preparado em 21/08/2026 com base nas transcrições das reuniões de alinhamento e kickoff da KD Diagnósticos.*
