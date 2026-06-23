defmodule ExAlphaZeroWeb.PageController do
  use ExAlphaZeroWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
