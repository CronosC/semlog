#!/usr/bin/env bash
# Builds run configs one at a time, each in its own memory-capped systemd scope.
# Errors are handled per config, so a failed build does not stop the rest.
# It works in the current directory, because the paths inside a run config
# resolve against it and a worktree lacks the gitignored datasets/.

MEM=${SEMLOG_MEM:-6G}
DRY_RUN=0
INTERRUPTED=0
SIGNAL_STATUS=130
CHILD=
WOKEN=0
FAILED=0
RESULTS=()
CAPPED=(systemd-run --user --scope -q -p "MemoryMax=$MEM" -p MemorySwapMax=0)

usage() {
  cat <<EOF
usage: scripts/build_runs.sh [-n] CONFIG...

Run it from the repository root. Builds each CONFIG in turn, one at a time, under
systemd-run --user --scope -p MemoryMax=\$SEMLOG_MEM (now $MEM).
A CONFIG is a run name such as p2p, read as datasets/runs/p2p.toml, or a
path to a TOML file. A declare_file that does not exist is mined first.

Each run writes NAME.build.log and NAME.build.exit into the output
directory of its config, then a summary line per config is printed.
The exit code is 1 when any config failed. Ctrl-C ends the loop after
the current config. TERM and HUP stop the current build, write its
NAME.build.log and NAME.build.exit, and end the loop.

  -n  print the resolved configuration of each CONFIG and build nothing
  -h  print this help
EOF
}

resolve_config() {
  local arg=$1 path
  if [[ "$arg" != */* && "$arg" != *.toml ]]; then
    path="$PWD/datasets/runs/$arg.toml"
  else
    path=$(realpath -m -- "$arg")
  fi
  if [[ ! -f "$path" ]]; then
    echo "Config not found: $path. Name a file in datasets/runs/ or pass a TOML path." >&2
    return 1
  fi
  printf '%s\n' "$path"
}

setting() {
  local resolved=$1 key=$2
  sed -n "s/^$key = \"\(.*\)\"\$/\1/p" <<<"$resolved"
}

replace_file() {
  local src=$1 dst=$2 mode
  mode=$(printf '%o' "$((8#666 & ~8#$(umask)))")
  chmod "$mode" "$src"
  mv -f -- "$src" "$dst"
}

on_signal() {
  INTERRUPTED=1
  WOKEN=1
  SIGNAL_STATUS=$2
  if [[ "$1" != INT && -n "$CHILD" ]]; then
    kill -s "$1" "$CHILD" 2>/dev/null
  fi
}

# Runs a command in the background with its output appended to the file $1,
# so that a trapped signal interrupts the wait and reaches the command.
# Returns the exit code of the command.
run_logged() {
  local log=$1 status code
  shift
  ( trap - INT QUIT; exec "$@" ) >>"$log" 2>&1 &
  CHILD=$!
  WOKEN=0
  wait "$CHILD"
  status=$?
  # A wait cut short by a trapped signal leaves the command running.
  while [[ "$WOKEN" -eq 1 ]]; do
    WOKEN=0
    wait "$CHILD" 2>/dev/null
    code=$?
    [[ "$code" -eq 127 ]] || status=$code
  done
  CHILD=
  return "$status"
}

record() {
  local name=$1 status=$2 seconds=$3 log=$4
  RESULTS+=("$(printf '%-28s %5s %8s  %s' "$name" "$status" "${seconds}s" "$log")")
  [[ "$status" -eq 0 ]] || FAILED=1
  printf '%s\n' "${RESULTS[-1]}"
}

build_one() {
  local config=$1 name resolved output declare_file log tmp exit_tmp status start
  name=$(basename "$config" .toml)
  start=$SECONDS
  if ! resolved=$(uv run semlog build --config "$config" --dry-run 2>&1); then
    printf '%s\n' "$resolved" >&2
    record "$name" 2 "$((SECONDS - start))" "-"
    return
  fi
  output=$(setting "$resolved" output)
  output=${output:-.}
  declare_file=$(setting "$resolved" declare_file)
  log="$output/$name.build.log"
  if ! mkdir -p -- "$output" 2>/dev/null ||
    ! tmp=$(mktemp "$output/.$name.build.log.XXXXXX" 2>/dev/null); then
    echo "Cannot write to $output, the output directory of $config. Make it writable or change output." >&2
    record "$name" 2 "$((SECONDS - start))" "-"
    return
  fi
  status=0
  if [[ -n "$declare_file" && ! -f "$declare_file" && "$INTERRUPTED" -eq 0 ]]; then
    run_logged "$tmp" "${CAPPED[@]}" uv run semlog mine-declare --config "$config"
    status=$?
  fi
  if [[ "$status" -eq 0 && "$INTERRUPTED" -eq 0 ]]; then
    run_logged "$tmp" "${CAPPED[@]}" uv run semlog build --config "$config"
    status=$?
  fi
  [[ "$status" -ne 0 || "$INTERRUPTED" -eq 0 ]] || status=$SIGNAL_STATUS
  replace_file "$tmp" "$log"
  if ! exit_tmp=$(mktemp "$output/.$name.build.exit.XXXXXX" 2>/dev/null); then
    echo "Cannot write $output/$name.build.exit. Make $output writable." >&2
  else
    printf '%s\n' "$status" >"$exit_tmp"
    replace_file "$exit_tmp" "$output/$name.build.exit"
  fi
  record "$name" "$status" "$((SECONDS - start))" "$log"
  [[ "$status" -ne 130 ]] || INTERRUPTED=1
}

dry_run_one() {
  local config=$1
  echo "=== $config"
  uv run semlog build --config "$config" --dry-run || FAILED=1
}

summary() {
  local line
  echo
  printf '%-28s %5s %8s  %s\n' NAME EXIT TIME LOG
  for line in "${RESULTS[@]}"; do
    printf '%s\n' "$line"
  done
}

main() {
  local opt arg config configs=()
  while getopts "nh" opt; do
    case "$opt" in
    n) DRY_RUN=1 ;;
    h) usage; return 0 ;;
    *) usage >&2; return 2 ;;
    esac
  done
  shift "$((OPTIND - 1))"
  if [[ $# -eq 0 ]]; then
    usage >&2
    return 2
  fi
  for arg in "$@"; do
    config=$(resolve_config "$arg") || return 2
    configs+=("$config")
  done
  trap 'on_signal INT 130' INT
  trap 'on_signal TERM 143' TERM
  trap 'on_signal HUP 129' HUP
  for config in "${configs[@]}"; do
    [[ "$INTERRUPTED" -eq 0 ]] || break
    if [[ "$DRY_RUN" -eq 1 ]]; then
      dry_run_one "$config"
    else
      build_one "$config"
    fi
  done
  [[ "$DRY_RUN" -eq 1 ]] || summary
  [[ "$INTERRUPTED" -eq 0 ]] || return "$SIGNAL_STATUS"
  return "$FAILED"
}

main "$@"
