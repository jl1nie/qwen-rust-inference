// Both spreads follow the same two retained positions and one rejected position.
#let state-history(show-rejected: false) = {
  let rejected = if show-rejected {
    ([3 検証のみ], [0, 1, 2], [$mat(7; 0)$])
  } else {
    ()
  }
  table(
    columns: (1.35fr, 1fr, 1fr),
    align: (left, center, center),
    table.header([処理した位置数], [有効なKVの位置], [DeltaNetの状態]),
    [1], [0], [$mat(2; 0)$],
    [2], [0, 1], [$mat(4; 0)$],
    ..rejected,
  )
}
