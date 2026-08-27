# Estimativa de Custos AWS — PoC IA Radiologia
## "KD Diagnósticos — PoC IA Radiologia — Cenário Inicial"

---

## 🔗 Link Compartilhável — AWS Pricing Calculator

> **https://calculator.aws/#/estimate?id=a51eac021402ae76ca33fb2cb657081f7e8bc911**

Estimativa gerada via AWS Pricing Calculator MCP Server em 21/08/2026.
Os custos são recalculados pela AWS ao abrir o link. Clique "Update estimate" para pricing atualizado.

---

## 1. Região AWS

| Decisão | Valor | Tipo |
|---|---|---|
| Região | **us-east-1 (N. Virginia)** | **PREMISSA** |

**Justificativa:** Não foi definida região nas transcrições. us-east-1 selecionada por ser a mais econômica para SageMaker e típica para PoCs subsidiadas pela AWS. Para produção, reavalisar sa-east-1 (São Paulo) por LGPD/latência (+20-30% custo).

---

## 2. Serviços Incluídos na Estimativa

| # | Serviço AWS | Grupo | Finalidade |
|---|---|---|---|
| 1 | Amazon S3 Standard | Armazenamento | Imagens raw/processadas, modelos, resultados |
| 2 | SageMaker Studio Notebooks | Desenvolvimento | EDA, pré-processamento, prototipação |
| 3 | SageMaker Training | Treinamento | Fine-tuning de modelos com GPU |
| 4 | SageMaker Batch Transform | Inferência | Inferência em lote para validação |
| 5 | Amazon CloudWatch | Monitoramento | Logs de jobs e debugging |

---

## 3. Configuração Detalhada por Serviço

### 3.1 Amazon S3 Standard
| Campo | Valor |
|---|---|
| Armazenamento | 70 GB/mês |
| Tamanho médio objeto | 2 MB |
| PUT/COPY/POST requests | 10.000/mês |
| GET requests | 50.000/mês |

### 3.2 SageMaker Studio Notebook
| Campo | Valor |
|---|---|
| Data Scientists | 1 |
| Notebooks/mês | 1 |
| Horas/dia | 8 |
| Dias/mês | 20 (dias úteis) |
| Instância | ml.t3.medium |

### 3.3 SageMaker Training
| Campo | Valor |
|---|---|
| Jobs/mês | 5 |
| Instâncias/job | 1 |
| Horas/job | 2 |
| Instância | ml.g4dn.xlarge (1× NVIDIA T4, 16 GB) |
| Storage/job | 30 GB gp2 |

### 3.4 SageMaker Batch Transform
| Campo | Valor |
|---|---|
| Jobs/mês | 2 |
| Instâncias/job | 1 |
| Horas/job | 2 |
| Instância | ml.g4dn.xlarge |

### 3.5 Amazon CloudWatch
| Campo | Valor |
|---|---|
| Standard Logs ingestão | 2.5 GB/mês |
| Retenção | 1 mês |

---

## 4. Custo Estimado

### Custo Mensal (baseado nos preços on-demand us-east-1)

| Serviço | Custo Mensal Estimado |
|---|---|
| Amazon S3 Standard | ~$1.70 |
| SageMaker Studio Notebook (ml.t3.medium, 160h) | ~$9.31 |
| SageMaker Training (ml.g4dn.xlarge, 10h/mês) | ~$7.36 |
| SageMaker Batch Transform (ml.g4dn.xlarge, 4h/mês) | ~$2.95 |
| CloudWatch Logs (2.5 GB ingest + storage) | ~$1.33 |
| **TOTAL MENSAL** | **~$22.65** |

### Projeções

| Período | Custo Estimado |
|---|---|
| **Mensal** | **~$22–$25/mês** |
| **Anual (projeção)** | **~$270–$300/ano** |
| **Total PoC (2 meses reais)** | **~$45–$50** |

> ⚠️ Os custos exatos são calculados pela AWS ao abrir o link. Os valores acima são referência baseada em preços públicos verificados.

---

## 5. Top 3 Maiores Componentes de Custo

| # | Componente | % Aprox. | Custo/mês |
|---|---|---|---|
| 🥇 | **SageMaker Notebook** (tempo de dev ativo) | ~41% | ~$9.31 |
| 🥈 | **SageMaker Training** (GPU) | ~33% | ~$7.36 |
| 🥉 | **SageMaker Batch Transform** (GPU) | ~13% | ~$2.95 |

O Notebook é o maior driver porque fica ligado muitas horas. Training é intenso mas curto por job.

---

## 6. Premissas que Mais Influenciam o Resultado

| # | Premissa | Valor Adotado | Impacto se Mudar |
|---|---|---|---|
| 1 | **Sem endpoint real-time** | Batch apenas | Se endpoint 24/7: **+$530/mês** (ml.m5.large) |
| 2 | Horas de notebook | 160h/mês (8h×20dias) | Se 12h/dia: +$4.66/mês |
| 3 | Número de training jobs | 5/mês | Se 15/mês: +$14.73/mês |
| 4 | Instância de training | ml.g4dn.xlarge | Se ml.g5.xlarge: +~$5/mês |
| 5 | Região | us-east-1 | Se sa-east-1: +20-30% em todos |
| 6 | Tamanho das imagens | 70 GB total S3 | Se 200 GB: +$3/mês |
| 7 | Spot Instances | Não usado | Se spot (-70%): -$5/mês em training |

---

## 7. Informações a Obter do Cliente para Refinar

| # | Informação Necessária | Por quê |
|---|---|---|
| 1 | Formato/tamanho real das imagens exportáveis | Define armazenamento S3 real |
| 2 | Se haverá demo com inferência real-time | Impacto brutal: +$530/mês se endpoint 24/7 |
| 3 | Preferência de região (LGPD, compliance) | Preços mudam significativamente |
| 4 | Quantidade de classes de classificação | Pode exigir mais epochs = mais horas GPU |
| 5 | Frequência de re-treinamento durante PoC | Mais jobs = mais custo |
| 6 | Se múltiplas modalidades (RX + mamografia) na PoC | Dobra dataset e treinamentos |
| 7 | Período exato da PoC (4, 6 ou 8 semanas?) | Define duração de cobrança |

---

## 8. Serviços Não Modelados (complementar)

| Item | Motivo | Impacto |
|---|---|---|
| EBS do Notebook (50 GB gp3) | A calculator inclui storage no próprio SageMaker | ~$4/mês (não significativo) |
| Data Transfer IN | Upload de imagens para S3 | **$0** (ingress é grátis) |
| Data Transfer OUT | Apenas download de resultados (~5 GB) | ~$0.45/mês |
| IAM / VPC | Sem custo direto | $0 |
| Free Tier (primeiros 250h ml.t3.medium) | Pode zerar custo notebook no mês 1 | -$9.31 no mês 1 |
| Créditos AWS (parceria Update) | Mencionado nas reuniões | Pode zerar 100% do custo |

---

## 9. Resumo Executivo

| Métrica | Valor |
|---|---|
| **Link da estimativa** | [Abrir na AWS Calculator](https://calculator.aws/#/estimate?id=a51eac021402ae76ca33fb2cb657081f7e8bc911) |
| **Custo mensal** | ~$22–$25 |
| **Custo total PoC (2 meses)** | ~$45–$50 |
| **Custo anual (projeção)** | ~$270–$300 |
| **Maior risco de custo** | Endpoint real-time (+$530/mês) |
| **Maior oportunidade** | Spot Training (-70% GPU) + Free Tier (-$9) |
| **Ordem de grandeza** | Inferior a US$50 para toda a PoC |
| **Região** | us-east-1 (PREMISSA) |
| **Serviços** | S3 + SageMaker (Notebook+Training+Batch) + CloudWatch |

---

*Estimativa gerada em 21/08/2026 via AWS Pricing Calculator MCP Server.*
*Preços: on-demand, us-east-1. Sujeitos a variação.*
