# Desafio MBA Engenharia de Software com IA - Full Cycle

## Descrição do Projeto

Este projeto implementa um sistema de **Retrieval-Augmented Generation (RAG)** para processamento e consulta de documentos PDF. Ele permite:

- **Ingestão de PDFs**: Carregar, dividir e armazenar embeddings de documentos PDF em um banco de dados vetorial (PostgreSQL com PGVector).
- **Busca semântica**: Realizar buscas por similaridade nos documentos usando embeddings gratuitos (HuggingFace).
- **Chatbot interativo**: Responder perguntas do usuário baseadas exclusivamente no conteúdo do PDF, via interface de linha de comando (CLI).

O projeto é didático e usa tecnologias gratuitas/locais para evitar custos, focando em engenharia de software com IA.

## Tecnologias Utilizadas

- **Python**: Linguagem principal.
- **LangChain**: Framework para chains de IA e processamento de documentos.
- **HuggingFace Embeddings**: Modelos de embeddings gratuitos e locais (sentence-transformers/all-MiniLM-L6-v2).
- **PostgreSQL + PGVector**: Banco vetorial para armazenamento e busca de embeddings.
- **Google Gemini**: Modelo de IA para geração de respostas (via API, requer chave gratuita).
- **Docker**: Para containerização do banco de dados (opcional, via docker-compose.yml).

## Pré-requisitos

- **Python 3.8+**: Instale via [python.org](https://www.python.org/).
- **PostgreSQL com PGVector**: Configure um banco local ou use Docker.
- **Chaves de API**:
  - `GOOGLE_API_KEY`: Para o modelo Gemini (obtenha gratuitamente em [Google AI Studio](https://makersuite.google.com/app/apikey)).
  - `HF_TOKEN`: Para HuggingFace (opcional, mas recomendado para downloads mais rápidos).
- **Arquivo PDF**: Coloque um PDF chamado `document.pdf` na raiz do projeto (ou ajuste `PDF_PATH` no `.env`).

## Como Executar

### 1. Clonagem e Configuração Inicial
```bash
git clone <url-do-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Configuração do Ambiente
- Copie o arquivo `.env` de exemplo e preencha as variáveis:
  ```
  GOOGLE_API_KEY=your_google_api_key
  HF_TOKEN=your_huggingface_token  # Opcional
  DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/rag
  PG_VECTOR_COLLECTION_NAME=langchain_challenge
  PDF_PATH=document.pdf
  ```
- Instale as dependências:
  ```bash
  pip install -r requirements.txt
  ```

### 3. Configuração do Banco de Dados
- Inicie o PostgreSQL com PGVector (usando Docker):
  ```bash
  docker-compose up -d
  ```
- Ou configure manualmente um banco PostgreSQL com extensão PGVector.

### 4. Ingestão do PDF
Execute o script de ingestão para processar e armazenar o PDF:
```bash
python src/ingest.py
```
- Este script carrega o PDF, divide em chunks, gera embeddings e armazena no banco vetorial.
- Logs detalhados são exibidos no console.

### 5. Execução do Chatbot
Após a ingestão, inicie o chatbot:
```bash
python src/chat.py
```
- O chatbot se apresenta e permite até 5 perguntas via CLI.
- Digite "sair" para encerrar.

## Como Usar

### Ingestão (`ingest.py`)
- **Propósito**: Processar um PDF e prepará-lo para buscas.
- **Execução**: Rode uma vez por PDF. O script valida variáveis de ambiente, carrega o documento, divide em trechos (chunks) de ~1000 caracteres com sobreposição, enriquece metadados e armazena embeddings no PostgreSQL.
- **Logs**: Acompanhe o progresso via console (INFO/ERROR).

### Busca (`search.py`)
- **Propósito**: Módulo interno para buscas semânticas.
- **Uso**: Chamado automaticamente pelo `chat.py`. Recebe uma pergunta, busca os top 5 trechos similares no banco e formata um prompt com contexto.

### Chatbot (`chat.py`)
- **Propósito**: Interface interativa para perguntas.
- **Fluxo**:
  1. Inicialização: O chatbot se apresenta.
  2. Loop de perguntas: Digite uma pergunta sobre o PDF.
  3. Resposta: Baseada apenas no contexto do PDF (RAG). Se não houver info, responde "Não tenho informações...".
- **Limitações**: Até 5 interações; respostas em português/inglês conforme o PDF.
- **Exemplo**:
  ```
  Chatbot: Olá, sou um assistente que responde perguntas sobre o PDF...
  User: Qual é o tema do documento?
  Chatbot: [Resposta baseada no PDF]
  ```

## Estrutura do Projeto
```
.
├── docker-compose.yml          # Configuração Docker para PostgreSQL
├── requirements.txt            # Dependências Python
├── .env                        # Variáveis de ambiente (não versionado)
├── src/
│   ├── ingest.py               # Script de ingestão de PDFs
│   ├── search.py               # Módulo de busca semântica
│   └── chat.py                 # Chatbot CLI
└── README.md                   # Esta documentação
```

## Solução Implementada

Esta solução atende ao desafio criando um pipeline completo de RAG:
- **Ingestão**: Processamento robusto com logs e tratamento de erros.
- **Busca**: Embeddings locais gratuitos para eficiência.
- **Chat**: Modelo de IA acessível, com respostas contextualizadas.

Para dúvidas ou melhorias, consulte os logs ou o código fonte.