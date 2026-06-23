defmodule ExAlphaZero.MCTS do
  alias ExAlphaZero.{Game, Utils, ResNet}

  defmodule Node do
    defstruct [
      :id, :game, :args, :board, :parent_id, :action_taken,
      prior: 0,
      visit_count: 0,
      value_sum: 0,
      children_ids: []
    ]
  end

  defmodule Tree do
    use Agent

    def start_link, do: Agent.start_link(fn -> {%{}, 0} end)
    def stop(tree),  do: Agent.stop(tree)

    def put_node(tree, node) do
      Agent.get_and_update(tree, fn {nodes, next_id} ->
        node = %{node | id: next_id}
        {next_id, {Map.put(nodes, next_id, node), next_id + 1}}
      end)
    end

    def get_node(tree, id) do
      Agent.get(tree, fn {nodes, _} -> Map.fetch!(nodes, id) end)
    end

    def update_node(tree, id, fun) do
      Agent.update(tree, fn {nodes, next_id} ->
        {Map.update!(nodes, id, fun), next_id}
      end)
    end

    def add_child(tree, parent_id, child_id) do
      update_node(tree, parent_id, fn node ->
        %{node | children_ids: node.children_ids ++ [child_id]}
      end)
    end
  end

  def search(%{args: args, game: game}, board) do
    {:ok, tree} = Tree.start_link()

    root_id =
      Tree.put_node(tree, %Node{
        game:         game,
        args:         args,
        board:        board,
        parent_id:    nil,
        action_taken: nil,
        visit_count:  1
      })

    {policy, _} = predict(board)

    valid_moves =
      board
      |> Game.get_valid_moves()
      |> Nx.to_flat_list()

    policy =
      policy
      |> Utils.add_dirichlet_noise(args)
      |> Utils.mask_and_normalize(valid_moves)

    expand(tree, root_id, policy)

    Enum.each(1..args.num_mcts_searches, fn _ ->
      node_id = select_leaf(tree, root_id)
      node    = Tree.get_node(tree, node_id)

      {value, is_terminal} =
        Game.get_value_and_terminated(game, node.board, node.action_taken)

      value = Game.get_opponent_value(value)

      value =
        if not is_terminal do
          {policy, value} = predict(node.board)

          valid_moves =
            node.board
            |> Game.get_valid_moves()
            |> Nx.to_flat_list()

          policy = Utils.mask_and_normalize(policy, valid_moves)

          expand(tree, node_id, policy)

          value |> Nx.squeeze() |> Nx.to_number()
        else
          value
        end

      backpropagate(tree, node_id, value)
    end)

    root  = Tree.get_node(tree, root_id)
    probs = List.duplicate(0.0, args.action_size)

    probs =
      Enum.reduce(root.children_ids, probs, fn child_id, acc ->
        child = Tree.get_node(tree, child_id)
        List.replace_at(acc, child.action_taken, child.visit_count * 1.0)
      end)

    total  = Enum.sum(probs)
    result = Enum.map(probs, fn v -> v / total end)

    Tree.stop(tree)
    result
  end

  defp expand(tree, node_id, policy) do
    node        = Tree.get_node(tree, node_id)
    valid_moves = Game.get_valid_moves(node.board) |> Nx.to_flat_list()

    policy
    |> Enum.with_index()
    |> Enum.zip(valid_moves)
    |> Enum.each(fn {{prob, action}, valid} ->
      if prob > 0 and valid == 1 do
        child_board =
          node.board
          |> Game.get_next_board(action, 1)
          |> Game.change_perspective(-1)

        child_id =
          Tree.put_node(tree, %Node{
            game:         node.game,
            args:         node.args,
            board:        child_board,
            parent_id:    node_id,
            action_taken: action,
            prior:        prob
          })

        Tree.add_child(tree, node_id, child_id)
      end
    end)
  end

  defp select_leaf(tree, node_id) do
    node = Tree.get_node(tree, node_id)

    if node.children_ids == [] do
      node_id
    else
      best_child_id =
        Enum.max_by(node.children_ids, fn child_id ->
          child = Tree.get_node(tree, child_id)
          get_ucb(node, child)
        end)

      select_leaf(tree, best_child_id)
    end
  end

  defp backpropagate(tree, node_id, value) do
    node = Tree.get_node(tree, node_id)

    Tree.update_node(tree, node_id, fn n ->
      %{n | value_sum: n.value_sum + value, visit_count: n.visit_count + 1}
    end)

    if node.parent_id != nil do
      backpropagate(tree, node.parent_id, Game.get_opponent_value(value))
    end
  end

  defp get_ucb(parent, child) do
    q_value =
      if child.visit_count == 0 do
        0
      else
        1 - (child.value_sum / child.visit_count + 1) / 2
      end

    q_value +
      parent.args.c *
      (:math.sqrt(parent.visit_count) / (child.visit_count + 1)) *
      child.prior
  end

  defp predict(board) do
    {policy, value} =
      board
      |> Game.get_encoded_board()
      |> Nx.new_axis(0)
      |> ResNet.predict()

    policy =
      policy
      |> Axon.Activations.softmax(axis: 1)
      |> Nx.squeeze(axes: [0])
      |> Nx.to_flat_list()

    {policy, value}
  end
end
