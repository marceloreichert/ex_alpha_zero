defmodule ExAlphaZero.Game do
  defstruct [:row_count, :col_count, :action_size, :in_a_row]

  def new(opts \\ []) do
    row_count = Keyword.get(opts, :row_count, 6)
    col_count = Keyword.get(opts, :col_count, 7)

    %__MODULE__{
      row_count:   row_count,
      col_count:   col_count,
      action_size: Keyword.get(opts, :action_size, col_count),
      in_a_row:    Keyword.get(opts, :in_a_row, 4)
    }
  end

  def get_initial_board(%__MODULE__{row_count: rows, col_count: cols}) do
    Nx.broadcast(Nx.tensor(0, type: :s32), {rows, cols})
  end

  def get_next_board(board, action, player) do
    col = board[[.., action]]

    empty_rows =
      col
      |> Nx.equal(0)
      |> Nx.to_flat_list()
      |> Enum.with_index()
      |> Enum.filter(fn {val, _i} -> val == 1 end)
      |> Enum.map(fn {_val, i} -> i end)

    case empty_rows do
      [] ->
        board

      rows ->
        row     = Enum.max(rows)
        indices = Nx.tensor([[row, action]])
        updates = Nx.tensor([player], type: :s32)
        Nx.indexed_put(board, indices, updates)
    end
  end

  def get_valid_moves(board) do
    case Nx.shape(board) do
      {_rows, _cols} ->
        board[0]
        |> Nx.equal(0)
        |> Nx.as_type(:u8)

      {_batch, _rows, _cols} ->
        board[[.., 0, ..]]
        |> Nx.equal(0)
        |> Nx.as_type(:u8)
    end
  end

  def check_win(_game, _board, nil), do: false

  def check_win(%__MODULE__{row_count: row_count, col_count: col_count, in_a_row: in_a_row}, board, action) do
    col = board[[.., action]] |> Nx.to_flat_list()

    occupied =
      col
      |> Enum.with_index()
      |> Enum.filter(fn {val, _i} -> val != 0 end)
      |> Enum.map(fn {_val, i} -> i end)

    case occupied do
      [] ->
        false

      rows ->
        row        = Enum.min(rows)
        player     = board[row][action] |> Nx.to_number()
        board_list = Nx.to_list(board)

        count = fn offset_row, offset_col ->
          Enum.reduce_while(1..(in_a_row - 1), 0, fn i, _acc ->
            r = row + offset_row * i
            c = action + offset_col * i

            if r < 0 or r >= row_count or c < 0 or c >= col_count do
              {:halt, i - 1}
            else
              val = board_list |> Enum.at(r) |> Enum.at(c)
              if val != player, do: {:halt, i - 1}, else: {:cont, i}
            end
          end)
        end

        count.(1,  0) >= in_a_row - 1 or
        count.(0,  1) + count.(0,  -1) >= in_a_row - 1 or
        count.(1,  1) + count.(-1, -1) >= in_a_row - 1 or
        count.(1, -1) + count.(-1,  1) >= in_a_row - 1
    end
  end

  def get_value_and_terminated(%__MODULE__{} = game, board, action) do
    if check_win(game, board, action) do
      {1, true}
    else
      valid_sum =
        board
        |> get_valid_moves()
        |> Nx.sum()
        |> Nx.to_number()

      if valid_sum == 0, do: {0, true}, else: {0, false}
    end
  end

  def change_perspective(board, player) do
    Nx.multiply(board, Nx.tensor(player, type: :s32))
  end

  def get_encoded_board(board) do
    planes =
      [-1, 0, 1]
      |> Enum.map(fn player ->
        Nx.equal(board, Nx.tensor(player, type: :s32))
      end)
      |> Nx.stack()
      |> Nx.as_type(:f32)

    case Nx.shape(board) do
      {_rows, _cols}         -> planes
      {_batch, _rows, _cols} -> Nx.transpose(planes, axes: [1, 0, 2, 3])
    end
  end

  def get_opponent(player),      do: -player
  def get_opponent_value(value), do: -value
end
