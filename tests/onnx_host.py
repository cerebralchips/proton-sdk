#!/usr/bin/env python3
"""Compare all decoder outputs against the independent SDK NumPy reference."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import iree.runtime as ireert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'models'))
from stories import Model, CONTEXT

ATOL, RTOL = 1e-4, 1e-5
NAMES = ['logits', 'keys', 'values']
SHAPES = [(512,), (5, CONTEXT, 32), (5, CONTEXT, 32)]


def compare(actual, expected, label):
    if len(actual) != 3 or len(expected) != 3:
        raise ValueError(f'{label}: missing output')
    metrics = {}
    for name, shape, got, want in zip(NAMES, SHAPES, actual, expected):
        got, want = np.asarray(got), np.asarray(want)
        if got.shape != shape or want.shape != shape or got.dtype != np.float32:
            raise ValueError(f'{label}/{name}: shape/dtype mismatch: {got.shape}/{got.dtype}')
        if not np.isfinite(got).all() or not np.isfinite(want).all():
            raise ValueError(f'{label}/{name}: nonfinite output')
        error = np.abs(got-want)
        limit = ATOL + RTOL*np.abs(want)
        if np.any(error > limit):
            idx = np.unravel_index(np.argmax(error/limit), shape)
            raise ValueError(f'{label}/{name}: mismatch at {idx}: {got[idx]} vs {want[idx]}; max abs {error.max()}')
        metrics[name] = {'elements': int(got.size), 'max_abs_error': float(error.max())}
    if int(np.argmax(actual[0])) != int(np.argmax(expected[0])):
        raise ValueError(f'{label}: greedy token mismatch')
    return metrics


def host_copy(value):
    # Pinned IREE 3.11's MappedMemory.asarray/to_host path leaks keep-alive
    # references with this NumPy version. local-sync outputs are host-mappable;
    # the buffer protocol gives an owned copy without that binding path.
    return np.frombuffer(value._buffer_view.map(), dtype=value.dtype).reshape(value.shape).copy()


def check_inputs(token, position, keys, values):
    if not 0 <= token < 512 or not 0 <= position < CONTEXT:
        raise ValueError('token/position outside the exported graph contract')
    for x in (keys, values):
        if x.shape != SHAPES[1] or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('invalid KV-cache input')


class Backend:
    def __init__(self, directory, mode, kind):
        self.kind = kind
        self.function = f'stories260k_{mode}'
        prefix = directory/f'stories260k.{mode}'
        if kind == 'onnxruntime':
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            self.session = ort.InferenceSession(str(prefix)+'.onnx', sess_options=options,
                                                providers=['CPUExecutionProvider'])
        else:
            self.module = ireert.load_vm_flatbuffer(Path(str(prefix)+'.vmfb').read_bytes(), driver='local-sync')

    def __call__(self, token, position, keys, values):
        check_inputs(token, position, keys, values)
        args = [np.array([token], np.int64), np.array([position], np.int64), keys, values]
        if self.kind == 'onnxruntime':
            outputs = self.session.run(None, dict(zip(['token', 'position', 'keys', 'values'], args)))
        else:
            outputs = [host_copy(x) for x in self.module[self.function](*args)]
        return outputs


def reference_step(model, token, position, quantized):
    logits = model.step(token, position, quantized)
    return [logits.copy(), model.keys.copy(), model.values.copy()]


def quantization_probe(directory):
    x = np.zeros((3,172), np.float32)
    # Row 0: zero input. Row 1: exact half ties and extrema. Row 2: neighboring floats.
    x[1,:8] = [.5,-.5,1.5,-1.5,126.5,-126.5,127,-127]
    x[2,:7] = [np.nextafter(np.float32(.5),np.float32(0)),
               np.nextafter(np.float32(.5),np.float32(1)),
               np.nextafter(np.float32(-.5),np.float32(0)),
               np.nextafter(np.float32(-.5),np.float32(-1)), 127,-127,2.5]
    # Python scalar FP32 operations reproduce the SDK's rounding expression.
    # No exporter or reference quantizer is invoked to generate this oracle.
    q = np.array([[int(np.sign(a))*int(np.floor(np.float32(abs(a)+np.float32(.5))))
                   for a in row] for row in x], np.int8)
    scales = np.ones((3,1), np.float32)
    w = np.array([[(i*172+j)*37 % 255 - 127 for j in range(172)] for i in range(5)], np.int8)
    w[:,0] = 127
    acc = np.array([[sum(int(a)*int(b) for a,b in zip(row,col)) for col in w] for row in q], np.int32)
    expected = [q,scales,acc,acc.astype(np.float32)]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(directory/'quantization_probe.onnx'), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    module = ireert.load_vm_flatbuffer((directory/'quantization_probe.vmfb').read_bytes(), driver='local-sync')
    outputs = {'onnxruntime':session.run(None, {'input':x}),
               'iree':[host_copy(y) for y in module.quantization_probe(x)]}
    for backend, values in outputs.items():
        for name, actual, want in zip(['activation','scale','accumulator','rescaling'],values,expected):
            if actual.dtype != want.dtype or actual.shape != want.shape or not np.array_equal(actual, want):
                raise ValueError(f'quantization probe {backend}/{name}: exact comparison failed')
    print('Quantization probe: PASS exact INT8 activations, INT32 accumulators and rescaling', flush=True)
    return {'result':'PASS', 'backends':list(outputs), 'activation_elements':int(q.size),
            'accumulator_elements':int(acc.size), 'reduction_size':172,
            'coverage':['zero scale guard','positive/negative half ties','neighboring FP32 values',
                        'signed INT8 extrema','INT32 accumulation','matrix tile tails'],
            'comparison':'exact equality'}


def validate(model_dir, directory, corrupt_output=None):
    results, arrays = {}, {}
    probe = quantization_probe(directory)
    for mode in ['fp32', 'w8a8']:
        quantized = mode == 'w8a8'
        graph = onnx.load(directory/f'stories260k.{mode}.onnx')
        onnx.checker.check_model(graph, full_check=True)
        if quantized:
            assert sum(t.data_type == onnx.TensorProto.INT8 for t in graph.graph.initializer) == 36
        results[mode] = {}
        for kind in ['onnxruntime', 'iree']:
            backend = Backend(directory, mode, kind)
            ref = Model(model_dir)
            keys, values = np.zeros(SHAPES[1], np.float32), np.zeros(SHAPES[2], np.float32)
            token = 1
            cases, tokens, logits = [], [], []
            started = time.monotonic()
            # Feed each backend's own caches back, exercising accumulated drift.
            for position in range(16):
                expected = reference_step(ref, token, position, quantized)
                actual = backend(token, position, keys, values)
                if corrupt_output and position == 0:
                    print(f'Deliberate corruption: {corrupt_output}', flush=True)
                    actual[NAMES.index(corrupt_output)].flat[-1] += np.float32(1)
                cases.append({'case': f'generation-{position}', **compare(actual, expected, f'{mode}/{kind}/{position}')})
                for name, got, want in zip(NAMES, actual, expected):
                    arrays[f'{mode}_{kind}_step{position}_{name}'] = got
                    arrays[f'{mode}_{kind}_step{position}_reference_{name}'] = want
                token = int(np.argmax(actual[0]))
                keys, values = actual[1:]
                tokens.append(token)
                logits.append(actual[0])
            arrays[f'{mode}_{kind}_logits'] = np.stack(logits)
            arrays[f'{mode}_{kind}_keys'] = keys
            arrays[f'{mode}_{kind}_values'] = values
            # Seeded arbitrary caches at first, interior and last legal positions.
            # Future slots are nonzero: masking and preservation must both work.
            rng = np.random.default_rng(260)
            for position in [0, 1, 31, 63]:
                keys = rng.normal(0, .1, SHAPES[1]).astype(np.float32)
                values = rng.normal(0, .1, SHAPES[2]).astype(np.float32)
                token = [0, 511, 42, 1][[0, 1, 31, 63].index(position)]
                ref.keys, ref.values = keys.copy(), values.copy()
                expected = reference_step(ref, token, position, quantized)
                actual = backend(token, position, keys, values)
                cases.append({'case': f'boundary-{position}', **compare(actual, expected, f'{mode}/{kind}/boundary-{position}')})
                for name, got, want in zip(NAMES, actual, expected):
                    arrays[f'{mode}_{kind}_boundary{position}_{name}'] = got
                    arrays[f'{mode}_{kind}_boundary{position}_reference_{name}'] = want
                keep = np.arange(CONTEXT) != position
                if not np.array_equal(actual[1][:, keep], keys[:, keep]) or not np.array_equal(actual[2][:, keep], values[:, keep]):
                    raise ValueError('Non-current cache slots were modified')
            results[mode][kind] = {'result':'PASS', 'steps':16, 'tokens':tokens,
                'text':ref.decode(tokens), 'cases':cases,
                'execution_seconds':round(time.monotonic()-started, 6)}
            print(f'{mode}/{kind}: PASS {len(cases)} cases; 16 generated tokens: {ref.decode(tokens)}', flush=True)
    # Quantization error is a separate measurement, not an implementation error.
    fp, quant = Model(model_dir), Model(model_dir)
    token = 1
    fp_logits, q_logits = [], []
    for pos in range(16):
        f = fp.step(token, pos, False)
        q = quant.step(token, pos, True)
        fp_logits.append(f); q_logits.append(q)
        token = int(np.argmax(f))
    error = np.stack(q_logits)-np.stack(fp_logits)
    quantization = {'teacher_forced_steps':16, 'max_abs_logit_error':float(np.max(np.abs(error))),
        'rms_logit_error':float(np.sqrt(np.mean(error*error))),
        'top1_agreement':float(np.mean(np.argmax(q_logits,axis=1)==np.argmax(fp_logits,axis=1)))}
    # The same acceptance function must reject corrupted logits AND cache data.
    expected = [x.copy() for x in actual]  # Mutate actual last-step outputs, not a synthetic stand-in.
    negative = []
    for name, output in zip(NAMES, range(3)):
        bad = [x.copy() for x in expected]
        bad[output].flat[-1] += np.float32(1)
        try:
            compare(bad, expected, 'deliberate-corruption')
        except ValueError:
            negative.append(name+' corruption rejected')
        else:
            raise AssertionError('Corrupted output accepted')
    for token, position in [(-1,0),(512,0),(1,-1),(1,64)]:
        try:
            check_inputs(token, position, expected[1], expected[2])
        except ValueError:
            negative.append(f'invalid token={token},position={position} rejected')
        else:
            raise AssertionError('Invalid input accepted')
    np.savez(directory/'outputs.npz', **arrays)
    return {'result':'PASS', 'tolerance':{'atol':ATOL, 'rtol':RTOL}, 'runs':results,
            'quantization_error':quantization, 'negative_checks':negative, 'quantization_probe':probe}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--corrupt-output', choices=NAMES)
    args = parser.parse_args()
    record = validate(args.model_dir, args.artifacts, args.corrupt_output)
    (args.artifacts/'host-results.json').write_text(json.dumps(record, indent=2)+'\n')
    print('FULL OUTPUT COMPARISON PASS (host only)', flush=True)
