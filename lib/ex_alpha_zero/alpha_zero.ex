defmodule ExAlphaZero.AlphaZero do
  alias ExAlphaZero.{Game, MCTS, ResNet, Utils}

  def self_play(game, args) do
    board  = Game.get_initial_board(game)
    player = 1
    do_self_play(game, args, board, player, [])
  end

  defp do_self_play(game, args, board, player, memory) do
    neutral_board = Game.change_perspective(board, player)
    action_probs  = MCTS.search(%{args: args, game: game}, neutral_board)

    memory = [{neutral_board, action_probs, player} | memory]

    valid_moves = Game.get_valid_moves(board) |> Nx.to_flat_list()

    action =
      action_probs
      |> Utils.apply_temperature(args.temperature)
      |> Utils.mask_and_normalize(valid_moves)
      |> Utils.sample_action(args.action_size)

    board = Game.get_next_board(board, action, player)

    {value, is_terminal} = Game.get_value_and_terminated(game, board, action)

    if is_terminal do
      build_return_memory(memory, value, player)
    else
      do_self_play(game, args, board, Game.get_opponent(player), memory)
    end
  end

  defp build_return_memory(memory, value, current_player) do
    Enum.map(memory, fn {hist_board, hist_probs, hist_player} ->
      outcome =
        if hist_player == current_player,
          do:   value,
          else: Game.get_opponent_value(value)

      {Game.get_encoded_board(hist_board), hist_probs, outcome}
    end)
  end

  def train(memory, args) do
    memory
    |> Enum.shuffle()
    |> Enum.chunk_every(args.batch_size)
    |> Enum.with_index(1)
    |> Enum.each(fn {batch, _i} ->
      {states, policy_targets, value_targets} =
        Enum.reduce(batch, {[], [], []}, fn {s, p, v}, {sa, pa, va} ->
          {[s | sa], [p | pa], [v | va]}
        end)
        |> then(fn {sa, pa, va} ->
          {Enum.reverse(sa), Enum.reverse(pa), Enum.reverse(va)}
        end)

      state_tensor =
        states
        |> Enum.map(&Nx.new_axis(&1, 0))
        |> Nx.concatenate(axis: 0)
        |> Nx.as_type(:f32)

      policy_tensor =
        policy_targets
        |> Enum.map(&Nx.tensor(&1, type: :f32))
        |> Nx.stack()

      value_tensor =
        value_targets
        |> Enum.map(&Nx.tensor([&1], type: :f32))
        |> Nx.stack()

      ResNet.train_step(state_tensor, policy_tensor, value_tensor)
    end)
  end
end
