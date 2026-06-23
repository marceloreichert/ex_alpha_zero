defmodule ExAlphaZeroWeb.GameLive do
  use ExAlphaZeroWeb, :live_view

  alias ExAlphaZero.{Game, MCTS, TrainingServer}

  @default_args %{
    num_iterations: 16,
    num_selfplay_iterations: 20,
    num_epochs: 4,
    batch_size: 64,
    temperature: 1.25,
    learning_rate: 0.001,
    num_mcts_searches: 200,
    dirichlet_epsilon: 0.25,
    dirichlet_alpha: 0.3,
    action_size: 7,
    c: 2.0
  }

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket), do: TrainingServer.subscribe()

    game = Game.new()
    board = Game.get_initial_board(game)
    models = TrainingServer.list_models()

    {:ok,
     assign(socket,
       game: game,
       board: board_to_list(board),
       nx_board: board,
       status: :setup,
       message: "Bem-vindo ao AlphaZero Elixir Connect Four!",
       model_loaded: false,
       available_models: models,
       training: false,
       training_progress: %{phase: nil, current: 0, total: 0},
       game_over: false,
       valid_moves: get_valid_moves_list(board)
     )}
  end

  @impl true
  def handle_event("train", _params, socket) do
    args = Map.put(@default_args, :action_size, socket.assigns.game.action_size)
    TrainingServer.start_training(args)
    {:noreply, assign(socket, training: true, status: :training, message: "Treinando modelo...")}
  end

  @impl true
  def handle_event("load_model", %{"iteration" => iteration_str}, socket) do
    iteration = String.to_integer(iteration_str)

    case TrainingServer.load_model(iteration) do
      :ok ->
        {:noreply,
         start_new_game(socket)
         |> assign(
           model_loaded: true,
           status: :playing,
           message: "Modelo #{iteration} carregado! Sua vez (🟡)."
         )}

      {:error, reason} ->
        {:noreply, assign(socket, message: "Erro ao carregar modelo: #{reason}")}
    end
  end

  @impl true
  def handle_event("drop", %{"col" => col_str}, socket) do
    if socket.assigns.status != :playing or socket.assigns.game_over do
      {:noreply, socket}
    else
      col = String.to_integer(col_str)
      valid = socket.assigns.valid_moves

      if Enum.at(valid, col) == 1 do
        pid = self()

        socket =
          socket
          |> apply_human_move(col)
          |> then(fn s ->
            if not s.assigns.game_over do
              assign(s, status: :thinking, message: "IA pensando...")
            else
              s
            end
          end)

        if not socket.assigns.game_over do
          Task.start(fn ->
            action =
              get_ai_action(
                socket.assigns.game,
                Map.put(@default_args, :action_size, socket.assigns.game.action_size),
                socket.assigns.nx_board
              )

            send(pid, {:ai_move, action})
          end)
        end

        {:noreply, socket}
      else
        {:noreply, socket}
      end
    end
  end

  @impl true
  def handle_event("new_game", _params, socket) do
    {:noreply, start_new_game(socket)}
  end

  @impl true
  def handle_event("refresh_models", _params, socket) do
    {:noreply, assign(socket, available_models: TrainingServer.list_models())}
  end

  @impl true
  def handle_event("back_to_setup", _params, socket) do
    {:noreply, assign(socket, status: :setup, available_models: TrainingServer.list_models())}
  end

  @impl true
  def handle_info({:ai_move, action}, socket) do
    {:noreply, apply_ai_move(socket, action)}
  end

  @impl true
  def handle_info({:training_event, event}, socket) do
    {:noreply, handle_training_event(socket, event)}
  end

  # ── rendering helpers ──────────────────────────────────────────────────────

  defp handle_training_event(socket, :training_completed) do
    models = TrainingServer.list_models()

    socket
    |> assign(
      training: false,
      status: :setup,
      message: "Treinamento concluído!",
      available_models: models,
      training_progress: %{phase: nil, current: 0, total: 0}
    )
  end

  defp handle_training_event(socket, {:training_started, total}) do
    assign(socket, training_progress: %{phase: :iterations, current: 0, total: total})
  end

  defp handle_training_event(socket, {:iteration_started, n, total}) do
    assign(socket,
      message: "Iteração #{n}/#{total}...",
      training_progress: %{phase: :iterations, current: n, total: total}
    )
  end

  defp handle_training_event(socket, {:selfplay_progress, n, total}) do
    assign(socket,
      message: "Self-play #{n}/#{total}...",
      training_progress: %{phase: :selfplay, current: n, total: total}
    )
  end

  defp handle_training_event(socket, {:epoch_progress, n, total}) do
    assign(socket,
      message: "Treinando epoch #{n}/#{total}...",
      training_progress: %{phase: :epoch, current: n, total: total}
    )
  end

  defp handle_training_event(socket, {:iteration_completed, n}) do
    assign(socket, message: "Iteração #{n} salva.")
  end

  defp handle_training_event(socket, {:training_error, reason}) do
    assign(socket,
      training: false,
      status: :setup,
      message: "Erro no treinamento: #{reason}"
    )
  end

  defp handle_training_event(socket, _), do: socket

  # ── game logic ─────────────────────────────────────────────────────────────

  defp apply_human_move(socket, col) do
    game = socket.assigns.game
    nx_board = socket.assigns.nx_board

    new_board = Game.get_next_board(nx_board, col, -1)
    {value, is_terminal} = Game.get_value_and_terminated(game, new_board, col)

    if is_terminal do
      msg = if value == 1, do: "Você venceu! 🎉", else: "Empate!"

      assign(socket,
        nx_board: new_board,
        board: board_to_list(new_board),
        valid_moves: get_valid_moves_list(new_board),
        game_over: true,
        status: :game_over,
        message: msg
      )
    else
      assign(socket,
        nx_board: new_board,
        board: board_to_list(new_board),
        valid_moves: get_valid_moves_list(new_board)
      )
    end
  end

  defp apply_ai_move(socket, action) do
    game = socket.assigns.game
    nx_board = socket.assigns.nx_board

    new_board = Game.get_next_board(nx_board, action, 1)
    {value, is_terminal} = Game.get_value_and_terminated(game, new_board, action)

    if is_terminal do
      msg = if value == 1, do: "IA venceu! 🤖", else: "Empate!"

      assign(socket,
        nx_board: new_board,
        board: board_to_list(new_board),
        valid_moves: get_valid_moves_list(new_board),
        game_over: true,
        status: :game_over,
        message: msg
      )
    else
      assign(socket,
        nx_board: new_board,
        board: board_to_list(new_board),
        valid_moves: get_valid_moves_list(new_board),
        status: :playing,
        message: "IA jogou coluna #{action}. Sua vez (🟡)."
      )
    end
  end

  defp start_new_game(socket) do
    game = socket.assigns.game
    board = Game.get_initial_board(game)

    assign(socket,
      nx_board: board,
      board: board_to_list(board),
      valid_moves: get_valid_moves_list(board),
      game_over: false,
      status: :playing,
      message: "Nova partida! Sua vez (🟡)."
    )
  end

  defp get_ai_action(game, args, board) do
    neutral_board = Game.change_perspective(board, 1)
    action_probs = MCTS.search(%{args: args, game: game}, neutral_board)
    action_probs |> Enum.with_index() |> Enum.max_by(fn {p, _} -> p end) |> elem(1)
  end

  defp board_to_list(board), do: Nx.to_list(board)

  defp get_valid_moves_list(board) do
    board |> Game.get_valid_moves() |> Nx.to_flat_list()
  end

  def progress_label(%{phase: :iterations}), do: "Iteração"
  def progress_label(%{phase: :selfplay}), do: "Self-play"
  def progress_label(%{phase: :epoch}), do: "Epoch"
  def progress_label(_), do: ""

  def progress_pct(%{current: 0, total: 0}), do: 0
  def progress_pct(%{current: c, total: t}), do: round(c / t * 100)
end
