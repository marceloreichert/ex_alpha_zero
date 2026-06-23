# ExAlphaZero

Implementação do algoritmo **AlphaZero** para o jogo **Connect Four (Lig 4)** em Elixir, com interface web via Phoenix LiveView para treinamento e partidas contra a IA.

## Visão Geral

O projeto combina:

- **ResNet** — rede neural residual que avalia posições e sugere jogadas
- **MCTS** — Monte Carlo Tree Search guiado pela rede neural
- **Self-play** — o modelo treina jogando contra si mesmo, sem dados humanos
- **Phoenix LiveView** — interface em tempo real para disparar o treinamento, acompanhar o progresso e jogar contra o modelo treinado

O jogo padrão é Connect Four em um tabuleiro 6×7 com vitória em 4 peças consecutivas.

## Requisitos

- Elixir 1.15+
- Erlang/OTP 26+
- [EXLA](https://hexdocs.pm/exla) — backend XLA para aceleração numérica (compilado na primeira execução)

## Instalação

```bash
# Clone o repositório
git clone https://github.com/<seu-usuario>/ex_alpha_zero.git
cd ex_alpha_zero

# Instale dependências e compile os assets
mix setup
```

> A primeira compilação pode demorar alguns minutos porque o EXLA precisa baixar e compilar o backend XLA.

## Executando o Servidor

### macOS (recomendado)

Use o script `run.sh`, que define as variáveis de ambiente necessárias para o EXLA compilar sem erros no Apple Silicon / macOS:

```bash
chmod +x run.sh
./run.sh
```

### Outros sistemas

```bash
mix phx.server
```

Acesse [http://localhost:4000](http://localhost:4000).

### Variáveis de ambiente

| Variável | Valor padrão | Descrição |
|---|---|---|
| `EXLA_CPU_ONLY` | `true` | Desativa CUDA/ROCm e usa apenas CPU. Remova ou defina como `false` para usar GPU. |
| `CFLAGS` | `-Wno-error -Wno-invalid-specialization` | Silencia erros de compilação do XLA no macOS. |
| `CXXFLAGS` | `-Wno-error -Wno-invalid-specialization` | Idem para C++. |

Você pode definir essas variáveis no arquivo `.env` ou exportá-las manualmente no shell antes de iniciar o servidor.

## Fluxo de Uso

1. Acesse `http://localhost:4000`
2. Clique em **Treinar** para iniciar o treinamento — o progresso aparece em tempo real
3. Após o treinamento, selecione um dos modelos salvos e clique em **Carregar Modelo**
4. Jogue contra a IA clicando nas colunas do tabuleiro

## Configuração do Treinamento

Os hiperparâmetros ficam em `lib/ex_alpha_zero_web/live/game_live.ex`:

```elixir
@default_args %{
  num_iterations:          16,   # iterações do loop principal (self-play → treino → salvar)
  num_selfplay_iterations: 20,   # partidas de self-play por iteração
  num_epochs:              4,    # épocas de treino sobre a memória coletada
  batch_size:              64,   # tamanho do mini-batch
  temperature:             1.25, # temperatura para amostragem de ações no self-play
  learning_rate:           0.001,
  num_mcts_searches:       200,  # simulações MCTS por jogada
  dirichlet_epsilon:       0.25, # peso do ruído Dirichlet na raiz do MCTS
  dirichlet_alpha:         0.3,  # parâmetro alpha da distribuição Dirichlet
  action_size:             7,    # número de colunas (ações possíveis)
  c:                       2.0   # constante de exploração UCB
}
```

### Guia de ajuste dos parâmetros

| Parâmetro | Aumentar | Diminuir |
|---|---|---|
| `num_iterations` | Modelo mais forte, treino mais longo | Treino mais rápido |
| `num_selfplay_iterations` | Mais dados por iteração | Menor diversidade de partidas |
| `num_mcts_searches` | IA mais forte durante self-play e jogo | Muito mais lento |
| `temperature` | Mais exploração nas jogadas | IA escolhe sempre a melhor jogada |
| `dirichlet_epsilon` | Mais ruído na raiz (mais exploração) | IA segue mais o modelo atual |
| `learning_rate` | Aprendizado mais agressivo (instável) | Convergência mais lenta e estável |
| `batch_size` | Gradientes mais estáveis | Maior uso de memória |

### Arquitetura da ResNet

Configurada em `lib/ex_alpha_zero/training_server.ex`:

```elixir
ensure_resnet(game, num_res_blocks: 9, num_hidden: 128, learning_rate: ...)
```

- **`num_res_blocks`** — número de blocos residuais (mais blocos = mais capacidade, mais lento)
- **`num_hidden`** — canais ocultos em cada camada convolucional

## Modelos Salvos

Cada iteração salva um snapshot em `models/params_<N>.nx`. Esses arquivos são serializados pelo Nx e podem ser carregados pela interface web. Para usar um modelo em outros contextos:

```elixir
params = "models/params_16.nx" |> File.read!() |> Nx.deserialize()
```

## Estrutura do Projeto

```
lib/
├── ex_alpha_zero/
│   ├── alpha_zero.ex       # loop de self-play e treino
│   ├── game.ex             # regras do Connect Four
│   ├── mcts.ex             # Monte Carlo Tree Search
│   ├── resnet.ex           # rede neural residual (Axon + EXLA)
│   ├── training_server.ex  # GenServer que orquestra o treinamento
│   └── utils.ex            # temperatura, ruído Dirichlet, amostragem
└── ex_alpha_zero_web/
    └── live/game_live.ex   # interface LiveView
models/
    params_1.nx ... params_N.nx   # checkpoints salvos pelo treinamento
```

## Dependências Principais

| Biblioteca | Função |
|---|---|
| [Nx](https://hexdocs.pm/nx) | Tensores e computação numérica |
| [Axon](https://hexdocs.pm/axon) | Construção e execução da rede neural |
| [EXLA](https://hexdocs.pm/exla) | Backend XLA (compilação JIT para CPU/GPU) |
| [Polaris](https://hexdocs.pm/polaris) | Otimizador Adam |
| [Phoenix LiveView](https://hexdocs.pm/phoenix_live_view) | Interface web reativa |

## Desenvolvimento

```bash
# Rodar testes
mix test

# Verificar formatação, compilação e testes de uma vez
mix precommit
```
