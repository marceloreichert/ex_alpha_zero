defmodule ExAlphaZero.Dirichlet do
  def sample(alphas) when is_list(alphas) do
    gammas = Enum.map(alphas, &sample_gamma/1)
    total  = Enum.sum(gammas)
    Enum.map(gammas, fn g -> g / total end)
  end

  defp sample_gamma(alpha) when alpha < 1 do
    u = :rand.uniform()
    sample_gamma(alpha + 1.0) * :math.pow(u, 1.0 / alpha)
  end

  defp sample_gamma(alpha) do
    d = alpha - 1.0 / 3.0
    c = 1.0 / :math.sqrt(9.0 * d)
    marsaglia_loop(d, c)
  end

  defp marsaglia_loop(d, c) do
    {x, v} = sample_v(c)
    u = :rand.uniform()

    cond do
      v <= 0.0 ->
        marsaglia_loop(d, c)

      u < 1.0 - 0.0331 * (x * x) * (x * x) ->
        d * v

      :math.log(u) < 0.5 * x * x + d * (1.0 - v + :math.log(v)) ->
        d * v

      true ->
        marsaglia_loop(d, c)
    end
  end

  defp sample_v(c) do
    x = sample_normal()
    v = 1.0 + c * x
    {x, v * v * v}
  end

  defp sample_normal do
    u1 = :rand.uniform()
    u2 = :rand.uniform()
    :math.sqrt(-2.0 * :math.log(u1)) * :math.cos(2.0 * :math.pi() * u2)
  end
end
