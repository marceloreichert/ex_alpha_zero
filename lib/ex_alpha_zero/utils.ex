defmodule ExAlphaZero.Utils do
  def mask_and_normalize(policy, valid_moves) do
    masked = Enum.zip_with(policy, valid_moves, fn p, v -> p * v end)
    total  = Enum.sum(masked)

    if total > 0 do
      Enum.map(masked, fn p -> p / total end)
    else
      n_valid = Enum.sum(valid_moves)
      Enum.map(valid_moves, fn v -> v / n_valid end)
    end
  end

  def apply_temperature(action_probs, temperature) do
    probs = Enum.map(action_probs, fn p -> :math.pow(p, 1.0 / temperature) end)
    total = Enum.sum(probs)
    Enum.map(probs, fn p -> p / total end)
  end

  def sample_action(probs, action_size) do
    r = :rand.uniform()

    probs
    |> Enum.with_index()
    |> Enum.reduce_while(0.0, fn {p, i}, acc ->
      acc = acc + p
      if r <= acc, do: {:halt, i}, else: {:cont, acc}
    end)
    |> then(fn
      i when is_integer(i) -> i
      _                    -> action_size - 1
    end)
  end

  def add_dirichlet_noise(policy, args) do
    epsilon = args.dirichlet_epsilon
    noise   = ExAlphaZero.Dirichlet.sample(List.duplicate(args.dirichlet_alpha, args.action_size))

    Enum.zip_with(policy, noise, fn p, n ->
      (1 - epsilon) * p + epsilon * n
    end)
  end
end
