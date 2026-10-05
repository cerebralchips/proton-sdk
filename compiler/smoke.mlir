module @proton_smoke {
  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.mulf %a, %b : tensor<4xf32>
    %1 = arith.addf %0, %a : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
