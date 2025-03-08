# Relatório Técnico: Fine-Tuning do Modelo Llama 3.3 8B para Dados de Produtos da Amazon

## Seção 1: Introdução e Arquitetura do Modelo
### 1.1 Visão Geral do Projeto
Este relatório técnico documenta o processo de fine-tuning do modelo Llama 3.3 8B, quantizado para precisão de 4 bits, para tarefas especializadas relacionadas a dados de produtos da Amazon. O projeto demonstra como aproveitar modelos de linguagem de última geração com recursos computacionais limitados através de técnicas de quantização e fine-tuning eficiente em parâmetros.

### 1.2 Arquitetura do Modelo Llama 3.3
Llama 3.3 é a mais recente iteração da Meta AI na família Llama de grandes modelos de linguagem. A variante de 8B parâmetros oferece um equilíbrio entre eficiência computacional e desempenho. As principais características arquitetônicas incluem:
- Arquitetura baseada em Transformer com design apenas de decodificador
- Janela de contexto aprimorada em comparação com gerações anteriores
- Capacidades aprimoradas de seguir instruções
- Otimizado para uma ampla gama de tarefas de processamento de linguagem natural

### 1.3 Abordagem de Quantização
Para tornar o fine-tuning viável em hardware de nível consumidor, empregamos quantização de 4 bits:
- Reduz a pegada de memória em aproximadamente 4x em comparação com a precisão de 16 bits
- Mantém a maior parte do desempenho do modelo enquanto reduz significativamente os requisitos computacionais
- Permite treinamento em GPUs com VRAM limitada (por exemplo, NVIDIA T4)
- Utiliza a biblioteca bitsandbytes para implementação eficiente de quantização

### 1.4 Fine-Tuning Eficiente em Parâmetros (PEFT)
Implementamos a Adaptação de Baixo Posto (LoRA) para fine-tuning eficiente em parâmetros:
- Adiciona pequenas matrizes de decomposição de posto treináveis aos pesos existentes
- Reduz drasticamente o número de parâmetros treináveis (de bilhões para milhões)
- Preserva a maior parte do conhecimento pré-treinado enquanto se adapta a novas tarefas
- Permite treinamento mais rápido e requisitos de memória reduzidos

## Seção 2: Preparação do Dataset e Configuração de Treinamento
### 2.1 Dataset de Produtos da Amazon
O dataset consiste em informações de produtos da Amazon com foco em três atributos principais:
- ID do Produto: Identificador único para cada produto
- Título do Produto: O nome ou título do produto
- Conteúdo do Produto: Descrição detalhada do produto

Todas as outras colunas do dataset original foram removidas para focar nesses atributos essenciais. Essa abordagem simplificada permite que o modelo aprenda as relações entre identificadores de produtos, títulos e descrições sem ser distraído por informações estranhas.

### 2.2 Configuração do Ambiente
O projeto requer um conjunto específico de bibliotecas para permitir treinamento eficiente:
```python
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
!pip install --no-deps cut_cross_entropy unsloth_zoo
!pip install datasets
!pip install unsloth
```

Essas bibliotecas fornecem:
- bitsandbytes: Suporte para quantização de 4 bits
- accelerate: Capacidades de treinamento distribuído
- xformers: Mecanismos de atenção eficientes em memória
- peft: Técnicas de fine-tuning eficientes em parâmetros
- trl: Treinamento de modelos de aprendizado por reforço
- unsloth: Biblioteca de otimização para treinamento mais rápido de modelos Llama

### 2.3 Inicialização do Modelo
O modelo é inicializado com quantização de 4 bits usando a biblioteca Unsloth:
```python
from unsloth import FastLanguageModel, is_bfloat16_supported

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

### 2.4 Configuração LoRA
Os adaptadores LoRA são configurados com parâmetros específicos para equilibrar desempenho e eficiência:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

## Seção 3: Processo de Treinamento e Otimização
### 3.1 Configuração de Treinamento
O processo de treinamento é configurado usando a classe TrainingArguments da biblioteca transformers:
```python
from transformers import TrainingArguments
from trl import SFTTrainer

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=is_bfloat16_supported(),
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=False,
    report_to="none",
)
```

Os principais parâmetros de treinamento incluem:
- 3 épocas de treinamento
- Tamanho de lote de 4 com acumulação de gradiente de 4 etapas
- Taxa de aprendizado de 2e-4 com um cronograma constante
- Recorte de gradiente em 0,3
- Treinamento de precisão mista com BF16 onde suportado

### 3.2 Processamento e Formatação de Dados
O dataset é processado e formatado para corresponder à estrutura de entrada esperada para o modelo:
```python
def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["id"])):
        text = f"Abaixo está uma instrução que descreve uma tarefa, junto com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente a solicitação.\n\n### Instrução:\nDescreva o seguinte produto em detalhes.\n\n### Entrada:\n{examples['title'][i]}\n\n### Resposta:\n{examples['content'][i]}"
        output_texts.append(text)
    return output_texts
```

Esta função cria exemplos de instrução-ajuste que solicitam ao modelo para gerar descrições detalhadas de produtos com base nos títulos dos produtos.

### 3.3 Execução do Treinamento
O treinamento é executado usando o SFTTrainer da biblioteca TRL:
```python
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    packing=False,
    max_seq_length=None,
    dataset_text_field=None,
)

trainer_stats = trainer.train()
```

### 3.4 Técnicas de Otimização
Várias técnicas de otimização são empregadas para melhorar a eficiência do treinamento:
- Checkpoint de gradiente para reduzir o uso de memória
- Treinamento de precisão mista para acelerar os cálculos
- Acumulação de gradiente para simular tamanhos de lote maiores
- Kernels otimizados da Unsloth para treinamento mais rápido de modelos Llama
- Cronograma de taxa de aprendizado constante com aquecimento para estabilizar o treinamento

## Seção 4: Avaliação do Modelo e Inferência
### 4.1 Preparando o Modelo para Inferência
Após o treinamento, o modelo é convertido para o modo de inferência:
```python
FastLanguageModel.for_inference(model)
```

Esta etapa remove componentes específicos de treinamento e otimiza o modelo para tarefas de geração.

### 4.2 Exemplos de Inferência
O modelo pode ser usado para inferência com o seguinte código:
```python
inputs = tokenizer(
    "Abaixo está uma instrução que descreve uma tarefa, junto com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente a solicitação.\n\n### Instrução:\nDescreva o seguinte produto em detalhes.\n\n### Entrada:\nFones de Ouvido Bluetooth Sem Fio\n\n### Resposta:",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Este exemplo demonstra como gerar uma descrição detalhada de produto para "Fones de Ouvido Bluetooth Sem Fio" usando o modelo fine-tuned.

### 4.3 Carregando o Modelo Fine-Tuned
O modelo fine-tuned pode ser carregado para uso futuro:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "/content/drive/MyDrive/Colab Notebooks/data/trn_model",
  max_seq_length = 2048,
  dtype = None,
  load_in_4bit = True,
)
```

### 4.4 Aplicações e Trabalhos Futuros
O modelo fine-tuned tem várias aplicações potenciais:
- Geração automatizada de descrições de produtos
- Enriquecimento de conteúdo para plataformas de e-commerce
- Aprimoramento de resultados de busca através de melhor compreensão do produto
- Sistemas de recomendação baseados em descrições de produtos

Trabalhos futuros podem incluir:
- Fine-tuning em datasets de produtos maiores e mais diversos
- Exploração de diferentes formatos de prompt para melhor desempenho
- Implementação de métricas de avaliação específicas para qualidade de descrição de produtos
- Extensão do modelo para lidar com entradas multimodais (texto + imagens)
- Implantação do modelo em ambientes de produção com inferência otimizada

### 4.5 Conclusion
This project demonstrates the feasibility of fine-tuning large language models like Llama 3.3 8B for specialized tasks with limited computational resources. Through quantization and parameter-efficient fine-tuning techniques, we've created a model capable of generating detailed product descriptions from basic product information. The approach outlined in this report can be extended to other domains and use cases where specialized language generation capabilities are required.
