defmodule ExAlphaZero.ResNet do
  import Axon

  def start_link(game, num_res_blocks, num_hidden, learning_rate) do
    model = build(game, num_res_blocks, num_hidden)

    {init_fn, predict_fn} = Axon.build(model, mode: :train)
    params = init_fn.(Nx.template({1, 3, game.row_count, game.col_count}, :f32), Axon.ModelState.empty())

    {opt_init_fn, opt_update_fn} = Polaris.Optimizers.adam(learning_rate: learning_rate)
    opt_state = opt_init_fn.(params.data)

    {_init_fn_infer, predict_fn_infer} = Axon.build(model, mode: :inference)

    grad_fn = Nx.Defn.jit(
      fn params_data, state_tensor, policy_targets, value_targets ->
        Nx.Defn.value_and_grad(params_data, fn params_data ->
          %{prediction: {out_policy, out_value}} =
            predict_fn.(Axon.ModelState.new(params_data), %{"state" => state_tensor})

          policy_loss =
            out_policy
            |> Axon.Activations.log_softmax(axis: 1)
            |> Nx.multiply(policy_targets)
            |> Nx.sum(axes: [1])
            |> Nx.mean()
            |> Nx.negate()

          value_loss =
            out_value
            |> Nx.subtract(value_targets)
            |> Nx.pow(2)
            |> Nx.mean()

          Nx.add(policy_loss, value_loss)
        end)
      end,
      compiler: EXLA
    )

    Agent.start_link(fn ->
      %{
        model:            model,
        params:           params,
        predict_fn_infer: predict_fn_infer,
        grad_fn:          grad_fn,
        opt_update_fn:    opt_update_fn,
        opt_state:        opt_state,
        learning_rate:    learning_rate
      }
    end, name: __MODULE__)
  end

  def predict(input) do
    %{params: params, predict_fn_infer: predict_fn_infer} =
      Agent.get(__MODULE__, & &1)

    predict_fn_infer.(params, %{"state" => input})
  end

  def get_params,    do: Agent.get(__MODULE__, & &1.params)
  def get_opt_state, do: Agent.get(__MODULE__, & &1.opt_state)

  def set_mode(_mode), do: :ok

  def train_step(state_tensor, policy_targets, value_targets) do
    %{params: params, opt_state: opt_state, grad_fn: grad_fn, opt_update_fn: opt_update_fn} =
      Agent.get(__MODULE__, & &1)

    {loss, grads} = grad_fn.(params.data, state_tensor, policy_targets, value_targets)

    {new_params_data, new_opt_state} = opt_update_fn.(grads, opt_state, params.data)

    Agent.update(__MODULE__, fn state ->
      %{state | params: %{state.params | data: new_params_data}, opt_state: new_opt_state}
    end)

    Nx.to_number(loss)
  end

  def save(iteration) do
    params = get_params()
    File.mkdir_p!("models")
    File.write!("models/params_#{iteration}.nx", Nx.serialize(params))
  end

  def load(iteration) do
    params = "models/params_#{iteration}.nx" |> File.read!() |> Nx.deserialize()
    Agent.update(__MODULE__, fn state -> %{state | params: params} end)
  end

  def running? do
    case Process.whereis(__MODULE__) do
      nil -> false
      _   -> true
    end
  end

  def stop do
    case Process.whereis(__MODULE__) do
      nil -> :ok
      pid -> Agent.stop(pid)
    end
  end

  def build(game, num_res_blocks, num_hidden) do
    input = input("state", shape: {nil, 3, game.row_count, game.col_count})

    x = start_block(input, num_hidden)
    x = Enum.reduce(1..num_res_blocks, x, fn _i, x -> res_block(x, num_hidden) end)

    policy = policy_head(x, game)
    value  = value_head(x)

    Axon.container({policy, value})
  end

  defp start_block(x, num_hidden) do
    x
    |> conv(num_hidden, kernel_size: {3, 3}, padding: :same, use_bias: false)
    |> batch_norm()
    |> relu()
  end

  defp res_block(x, num_hidden) do
    residual = x

    x
    |> conv(num_hidden, kernel_size: {3, 3}, padding: :same, use_bias: false)
    |> batch_norm()
    |> relu()
    |> conv(num_hidden, kernel_size: {3, 3}, padding: :same, use_bias: false)
    |> batch_norm()
    |> add(residual)
    |> relu()
  end

  defp policy_head(x, game) do
    x
    |> conv(32, kernel_size: {3, 3}, padding: :same, use_bias: false)
    |> batch_norm()
    |> relu()
    |> flatten()
    |> dense(game.action_size)
  end

  defp value_head(x) do
    x
    |> conv(3, kernel_size: {3, 3}, padding: :same, use_bias: false)
    |> batch_norm()
    |> relu()
    |> flatten()
    |> dense(1)
    |> tanh()
  end
end
