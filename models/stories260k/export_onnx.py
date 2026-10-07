#!/usr/bin/env python3
"""Export the pinned Stories260K decoder as standard, explicit ONNX equations.

The existing NumPy decoder is the independent execution oracle. Only its pinned
checkpoint reader is shared; no reference step outputs are embedded in the graph.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, helper as H, numpy_helper as NH

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stories import Model, CONTEXT


class Graph:
    def __init__(self):
        self.nodes, self.constants = [], []
        self.serial = 0

    def const(self, value, dtype=np.float32, name=None):
        if name is None:
            name = f'c{self.serial}'
            self.serial += 1
        self.constants.append(NH.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(self, kind, *args, **attrs):
        name = f'v{self.serial}'
        self.serial += 1
        self.nodes.append(H.make_node(kind, list(args), [name], name=name, **attrs))
        return name

    def reshape(self, x, shape):
        return self.op('Reshape', x, self.const(shape, np.int64))

    def gather(self, x, indices, axis=0):
        return self.op('Gather', x, self.const(indices, np.int64), axis=axis)

    def norm(self, x, weight):
        square = self.op('Mul', x, x)
        mean = self.op('ReduceMean', square, axes=[-1], keepdims=1)
        denom = self.op('Sqrt', self.op('Add', mean, self.const(1e-5)))
        return self.op('Mul', self.op('Div', x, denom), self.const(weight))

    def quantize_activation(self, x):
        amax = self.op('ReduceMax', self.op('Abs', x), axes=[-1], keepdims=1)
        raw_scale = self.op('Div', amax, self.const(127))
        scale = self.op('Where', self.op('Equal', raw_scale, self.const(0)),
                        self.const(1), raw_scale)
        scaled = self.op('Div', x, scale)
        # SDK contract is ties away from zero, not ONNX Round's ties to even.
        magnitude = self.op('Floor', self.op('Add', self.op('Abs', scaled), self.const(.5)))
        rounded = self.op('Mul', self.op('Sign', scaled), magnitude)
        return self.op('Cast', rounded, to=T.INT8), scale

    def linear(self, x, weight, quantized, name, trace=None):
        if not quantized:
            return self.op('MatMul', x, self.const(weight.T, name=name))
        # Independently implement per-output-channel weight quantization rather
        # than calling the reference quantizer under test.
        scale = np.max(np.abs(weight), axis=1) / np.float32(127)
        scale = np.where(scale == 0, np.float32(1), scale).astype(np.float32)
        values = weight / scale[:, None]
        packed = (np.sign(values) * np.floor(np.abs(values) + np.float32(.5))).astype(np.int8)
        qx, sx = self.quantize_activation(x)
        qw = self.const(packed.T, np.int8, name=name)
        # Standard ONNX INT32 MatMul after signed extension preserves exact
        # W8A8 products/sums on both importers. No float-dequantized GEMM.
        acc = self.op('MatMul', self.op('Cast', qx, to=T.INT32), self.op('Cast', qw, to=T.INT32))
        if trace is not None:
            trace.update(activation=qx, activation_scale=sx, accumulator=acc)
        fp = self.op('Cast', acc, to=T.FLOAT)
        return self.op('Mul', self.op('Mul', fp, sx), self.const(scale, name=name+'_scale'))

    def finish(self, inputs, outputs, name):
        graph = H.make_graph(self.nodes, name, inputs, outputs, self.constants)
        model = H.make_model(graph, producer_name='proton-sdk', opset_imports=[H.make_opsetid('', 17)])
        model.ir_version = 9
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        onnx.checker.check_model(model, full_check=True)
        return model


def export(model_dir, output, quantized):
    m = Model(model_dir)
    g = Graph()
    mode = 'w8a8' if quantized else 'fp32'
    hs = m.d // m.nh
    x = g.op('Gather', g.const(m.embedding, name='embedding'), 'token', axis=0)
    positions = g.const(np.arange(CONTEXT).reshape(CONTEXT, 1), np.int64)
    update_mask = g.op('Equal', positions, 'position')
    valid = g.reshape(g.op('LessOrEqual', positions, 'position'), [1, CONTEXT])
    # Fixed 64-position RoPE tables computed with the same FP32 definition as
    # checkpoint inference; this avoids implementation-dependent trig lowering.
    freq = np.power(np.float32(10000), np.arange(0, hs, 2, dtype=np.float32)/np.float32(hs))
    angles = np.arange(CONTEXT, dtype=np.float32)[:, None] / freq
    cos = g.op('Gather', g.const(np.cos(angles), name='rope_cos'), 'position', axis=0)
    sin = g.op('Gather', g.const(np.sin(angles), name='rope_sin'), 'position', axis=0)

    def rope(z, heads):
        z = g.reshape(z, [heads, hs])
        even, odd = g.gather(z, list(range(0, hs, 2)), 1), g.gather(z, list(range(1, hs, 2)), 1)
        a = g.op('Sub', g.op('Mul', even, cos), g.op('Mul', odd, sin))
        b = g.op('Add', g.op('Mul', even, sin), g.op('Mul', odd, cos))
        pair = g.op('Concat', g.reshape(a, [heads, hs//2, 1]),
                    g.reshape(b, [heads, hs//2, 1]), axis=2)
        return g.reshape(pair, [heads, hs])

    keys, values = [], []
    for layer in range(m.layers):
        def linear(x, kind):
            return g.linear(x, m.w[kind][layer], quantized, f'weight_{kind}_{layer}')
        xb = g.norm(x, m.norm_att[layer])
        q = rope(linear(xb, 'q'), m.nh)
        k = g.reshape(rope(linear(xb, 'k'), m.nkv), [1, m.kv])
        v = linear(xb, 'v')
        kcache = g.op('Where', update_mask, k, g.gather('keys', layer))
        vcache = g.op('Where', update_mask, v, g.gather('values', layer))
        keys.append(g.reshape(kcache, [1, CONTEXT, m.kv]))
        values.append(g.reshape(vcache, [1, CONTEXT, m.kv]))
        def heads(cache):
            split = g.reshape(cache, [CONTEXT, m.nkv, hs])
            split = g.op('Transpose', split, perm=[1, 0, 2])
            return g.gather(split, [h//(m.nh//m.nkv) for h in range(m.nh)])
        scores = g.op('MatMul', heads(kcache), g.reshape(q, [m.nh, hs, 1]))
        scores = g.op('Div', g.reshape(scores, [m.nh, CONTEXT]), g.const(np.sqrt(np.float32(hs))))
        scores = g.op('Where', valid, scores, g.const(-np.inf))
        probability = g.op('Softmax', scores, axis=-1)
        weighted = g.op('MatMul', g.reshape(probability, [m.nh, 1, CONTEXT]), heads(vcache))
        x = g.op('Add', x, linear(g.reshape(weighted, [1, m.d]), 'o'))
        xb = g.norm(x, m.norm_ffn[layer])
        up, gate = linear(xb, 'up'), linear(xb, 'gate')
        silu = g.op('Div', up, g.op('Add', g.const(1), g.op('Exp', g.op('Neg', up))))
        x = g.op('Add', x, linear(g.op('Mul', silu, gate), 'down'))
    logits = g.linear(g.norm(x, m.norm_final), m.w['cls'], quantized, 'weight_cls')
    for value, name in [(g.reshape(logits, [m.vocab]), 'logits'),
                        (g.op('Concat', *keys, axis=0), 'new_keys'),
                        (g.op('Concat', *values, axis=0), 'new_values')]:
        g.nodes.append(H.make_node('Identity', [value], [name]))
    inputs = [H.make_tensor_value_info('token', T.INT64, [1]),
              H.make_tensor_value_info('position', T.INT64, [1])]
    inputs += [H.make_tensor_value_info(n, T.FLOAT, [m.layers, CONTEXT, m.kv]) for n in ['keys', 'values']]
    outputs = [H.make_tensor_value_info('logits', T.FLOAT, [m.vocab])]
    outputs += [H.make_tensor_value_info(n, T.FLOAT, [m.layers, CONTEXT, m.kv]) for n in ['new_keys', 'new_values']]
    result = g.finish(inputs, outputs, f'stories260k_{mode}')
    H.set_model_props(result, {'precision': mode, 'context': str(CONTEXT),
        'checkpoint_sha256': hashlib.sha256((Path(model_dir)/'stories260K.bin').read_bytes()).hexdigest(),
        'quantization': 'symmetric W8A8; per-channel weights; dynamic per-row activations; ties away from zero' if quantized else 'FP32',
        'input_contract': 'token in [0,512); position in [0,64); finite FP32 caches; sequential positions for generation'})
    onnx.save(result, output)
    return {'precision': mode, 'nodes': len(result.graph.node), 'bytes': Path(output).stat().st_size,
            'sha256': hashlib.sha256(Path(output).read_bytes()).hexdigest(),
            'int8_initializers': sum(t.data_type == T.INT8 for t in result.graph.initializer)}


def export_quantization_probe(output):
    """Exercise zero scales, rounding ties and a 172-element reduction tail."""
    g = Graph()
    # Exactly representable integer-valued weights; per-channel scale is one.
    weight = ((np.arange(5*172).reshape(5, 172)*37) % 255 - 127).astype(np.float32)
    weight[:, 0] = 127
    trace = {}
    result = g.linear('input', weight, True, 'probe_weight', trace)
    specs = [('activation', T.INT8, [3,172]), ('activation_scale', T.FLOAT, [3,1]),
             ('accumulator', T.INT32, [3,5]), ('result', T.FLOAT, [3,5])]
    trace['result'] = result
    for name, _, _ in specs:
        g.nodes.append(H.make_node('Identity', [trace[name]], [name]))
    model = g.finish([H.make_tensor_value_info('input', T.FLOAT, [3,172])],
                     [H.make_tensor_value_info(n, t, shape) for n,t,shape in specs], 'quantization_probe')
    onnx.save(model, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {mode: export(args.model_dir, args.output_dir/f'stories260k.{mode}.onnx', mode=='w8a8')
              for mode in ['fp32', 'w8a8']}
    export_quantization_probe(args.output_dir/'quantization_probe.onnx')
    print(json.dumps(record, indent=2))
