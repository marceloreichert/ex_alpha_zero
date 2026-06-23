defmodule ExAlphaZero.TrainingServer do
  use GenServer

  alias ExAlphaZero.{AlphaZero, ResNet, Game}

  @pubsub ExAlphaZero.PubSub
  @topic  "training"

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  def start_training(args) do
    GenServer.cast(__MODULE__, {:start_training, args})
  end

  def load_model(iteration) do
    GenServer.call(__MODULE__, {:load_model, iteration}, 30_000)
  end

  def list_models do
    case File.ls("models") do
      {:ok, files} ->
        files
        |> Enum.filter(&String.starts_with?(&1, "params_"))
        |> Enum.map(fn f ->
          f
          |> String.replace("params_", "")
          |> String.replace(".nx", "")
          |> String.to_integer()
        end)
        |> Enum.sort()

      {:error, _} ->
        []
    end
  end

  def subscribe do
    Phoenix.PubSub.subscribe(@pubsub, @topic)
  end

  # ── GenServer callbacks ────────────────────────────────────────────────────

  @impl true
  def init(:ok) do
    {:ok, %{training: false, task: nil}}
  end

  @impl true
  def handle_cast({:start_training, args}, state) do
    if state.training do
      {:noreply, state}
    else
      game = Game.new()
      ensure_resnet(game, 9, 128, args.learning_rate)

      task = Task.async(fn -> run_training(game, args) end)
      broadcast({:training_started, args.num_iterations})

      {:noreply, %{state | training: true, task: task}}
    end
  end

  @impl true
  def handle_call({:load_model, iteration}, _from, state) do
    game = Game.new()
    ensure_resnet(game, 9, 128, 0.001)

    try do
      ResNet.load(iteration)
      {:reply, :ok, state}
    rescue
      e -> {:reply, {:error, Exception.message(e)}, state}
    end
  end

  @impl true
  def handle_info({ref, _result}, %{task: %{ref: ref}} = state) do
    Process.demonitor(ref, [:flush])
    broadcast(:training_completed)
    {:noreply, %{state | training: false, task: nil}}
  end

  def handle_info({:DOWN, ref, :process, _pid, reason}, %{task: %{ref: ref}} = state) do
    broadcast({:training_error, inspect(reason)})
    {:noreply, %{state | training: false, task: nil}}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # ── private ────────────────────────────────────────────────────────────────

  defp run_training(game, args) do
    Enum.each(1..args.num_iterations, fn iteration ->
      broadcast({:iteration_started, iteration, args.num_iterations})

      memory =
        Enum.flat_map(1..args.num_selfplay_iterations, fn i ->
          broadcast({:selfplay_progress, i, args.num_selfplay_iterations})
          AlphaZero.self_play(game, args)
        end)

      Enum.each(1..args.num_epochs, fn epoch ->
        broadcast({:epoch_progress, epoch, args.num_epochs})
        AlphaZero.train(memory, args)
      end)

      ResNet.save(iteration)
      broadcast({:iteration_completed, iteration})
    end)
  end

  defp ensure_resnet(game, num_res_blocks, num_hidden, learning_rate) do
    ResNet.stop()
    {:ok, _} = ResNet.start_link(game, num_res_blocks, num_hidden, learning_rate)
  end

  defp broadcast(event) do
    Phoenix.PubSub.broadcast(@pubsub, @topic, {:training_event, event})
  end
end
