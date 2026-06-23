#!/bin/sh
# Inicia o servidor Phoenix com os flags necessários para o EXLA no Mac
export EXLA_CPU_ONLY=true
export CFLAGS="-Wno-error -Wno-invalid-specialization"
export CXXFLAGS="-Wno-error -Wno-invalid-specialization"

exec mix phx.server
