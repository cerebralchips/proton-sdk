#!/usr/bin/env python3
"""Independent NumPy checkpoint reader, reference decoder and INT8 exporter.

Checkpoint layout follows llama2.c (MIT); math here uses NumPy arrays, not target
kernel code. Native FP32 llama2.c is an additional independent reference.
"""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np

MODEL_SHA = 'b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696'
TOKEN_SHA = '037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312'
CONTEXT = 64

def round_away(x):
    return (np.sign(x) * np.floor(np.abs(x) + np.float32(.5))).astype(np.int8)

def quantize(w):
    scale = np.max(np.abs(w), axis=-1, keepdims=True) / np.float32(127)
    scale = np.where(scale == 0, np.float32(1), scale).astype(np.float32)
    return round_away(w / scale), scale[..., 0]

class Model:
    def __init__(self, directory):
        self.directory=Path(directory)
        raw=(self.directory/'stories260K.bin').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==MODEL_SHA, 'Checkpoint hash mismatch'
        d,h,l,nh,nkv,v,seq=struct.unpack_from('<7i',raw)
        assert (d,h,l,nh,nkv,v,seq)==(64,172,5,8,4,512,512), 'Unsupported model configuration'
        self.d,self.h,self.layers,self.nh,self.nkv,self.vocab=d,h,l,nh,nkv,v
        self.kv=d*nkv//nh
        a=np.frombuffer(raw, dtype='<f4',offset=28)
        p=0
        def take(shape):
            nonlocal p
            count=int(np.prod(shape)); result=a[p:p+count].reshape(shape).copy(); p+=count
            assert np.isfinite(result).all()
            return result
        self.embedding=take((v,d)); self.norm_att=take((l,d))
        self.w={}
        for name,shape in [('q',(l,d,d)),('k',(l,self.kv,d)),('v',(l,self.kv,d)),('o',(l,d,d))]:
            self.w[name]=take(shape)
        self.norm_ffn=take((l,d))
        for name,shape in [('up',(l,h,d)),('down',(l,d,h)),('gate',(l,h,d))]:
            self.w[name]=take(shape)
        self.norm_final=take((d,)); take((seq,d//nh))
        assert p==len(a), (p,len(a))
        self.w['cls']=self.embedding
        self.qw={name:quantize(w) for name,w in self.w.items()}
        raw=(self.directory/'tok512.bin').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==TOKEN_SHA, 'Tokenizer hash mismatch'
        p=4; self.tokens=[]
        for i in range(v):
            score,n=struct.unpack_from('<fi',raw,p); p+=8
            self.tokens.append(raw[p:p+n]); p+=n
        assert p==len(raw)
        self.reset()

    def reset(self):
        self.keys=np.zeros((self.layers,CONTEXT,self.kv),np.float32)
        self.values=np.zeros_like(self.keys)

    def linear(self,x,name,layer,quantized):
        if not quantized:
            w=self.w[name] if layer is None else self.w[name][layer]
            return (w @ x).astype(np.float32)
        w,scale=self.qw[name]
        if layer is not None: w,scale=w[layer],scale[layer]
        qx,sx=quantize(x)
        acc=w.astype(np.int32) @ qx.astype(np.int32)
        return ((acc.astype(np.float32)*sx)*scale).astype(np.float32)

    @staticmethod
    def norm(x,w):
        return w * (x / np.sqrt(np.mean(x*x,dtype=np.float32)+np.float32(1e-5)))

    def step(self,token,pos,quantized=True):
        assert 0<=token<self.vocab and 0<=pos<CONTEXT
        x=self.embedding[token].copy(); hs=self.d//self.nh
        angle=np.float32(pos)/np.power(np.float32(10000),np.arange(0,hs,2,dtype=np.float32)/np.float32(hs))
        cos=np.cos(angle); sin=np.sin(angle)
        for l in range(self.layers):
            xb=self.norm(x,self.norm_att[l])
            q=self.linear(xb,'q',l,quantized).reshape(self.nh,hs)
            k=self.linear(xb,'k',l,quantized).reshape(self.nkv,hs)
            self.values[l,pos]=self.linear(xb,'v',l,quantized)
            for z in [q,k]:
                even=z[:,0::2].copy(); odd=z[:,1::2].copy()
                z[:,0::2]=even*cos-odd*sin; z[:,1::2]=even*sin+odd*cos
            self.keys[l,pos]=k.reshape(-1)
            out=np.zeros((self.nh,hs),np.float32)
            for head in range(self.nh):
                kh=head//(self.nh//self.nkv)
                ks=self.keys[l,:pos+1,kh*hs:(kh+1)*hs]
                vs=self.values[l,:pos+1,kh*hs:(kh+1)*hs]
                score=(ks@q[head])/np.sqrt(np.float32(hs))
                prob=np.exp(score-np.max(score)); prob/=np.sum(prob,dtype=np.float32)
                out[head]=prob@vs
            x+=self.linear(out.reshape(-1),'o',l,quantized)
            xb=self.norm(x,self.norm_ffn[l])
            up=self.linear(xb,'up',l,quantized); gate=self.linear(xb,'gate',l,quantized)
            hidden=(up/(np.float32(1)+np.exp(-up)))*gate
            x+=self.linear(hidden,'down',l,quantized)
        return self.linear(self.norm(x,self.norm_final),'cls',None,quantized)

    def generate(self,steps=16,quantized=True):
        self.reset(); token=1; logits=[]; tokens=[]
        for pos in range(steps):
            y=self.step(token,pos,quantized); token=int(np.argmax(y)); logits.append(y); tokens.append(token)
        return tokens,np.array(logits)

    def decode(self,tokens):
        return b''.join(self.tokens[t] for t in tokens).decode('utf8',errors='replace').lstrip()

def export(model,dest):
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    fp_tokens,fp=model.generate(16,False)
    q_tokens,q=model.generate(16,True)
    # Teacher-force the floating reference token path to separate numerical
    # differences from autoregressive divergence.
    model.reset(); tq=[]
    for pos,t in enumerate([1]+fp_tokens[:-1]): tq.append(model.step(t,pos,True))
    tq=np.array(tq)
    np.savez(dest/'reference.npz',fp_logits=fp,q_logits=q,q_tokens=q_tokens,fp_tokens=fp_tokens)
    metrics={'fp_tokens':fp_tokens,'quantized_tokens':q_tokens,'fp_text':model.decode(fp_tokens),
      'quantized_text':model.decode(q_tokens),'teacher_forced_max_abs_logit_error':float(np.max(np.abs(tq-fp))),
      'teacher_forced_rms_logit_error':float(np.sqrt(np.mean((tq-fp)**2))),
      'teacher_forced_top1_agreement':float(np.mean(np.argmax(tq,axis=1)==fp_tokens)),
      'context':CONTEXT,'steps':16,'weight_quantization':'symmetric INT8 per output channel',
      'activation_quantization':'dynamic symmetric INT8 per input row; ties away from zero'}
    (dest/'reference.json').write_text(json.dumps(metrics,indent=2)+'\n')
    # Emit exact hexadecimal float constants; target never reparses decimal weights.
    c=['#include "model_data.h"']; header=['#pragma once','#include <stdint.h>',
      'typedef struct { const int8_t *packed; const float *scale; unsigned n,k; } proton_weight_t;']
    def array(name,a,ctype):
        a=np.asarray(a).reshape(-1)
        values=[float(x).hex()+'f' for x in a] if ctype=='float' else [str(int(x)) for x in a]
        header.append(f'extern const {ctype} {name}[{len(a)}];')
        c.append(f'const {ctype} {name}[{len(a)}] __attribute__((aligned(16)))={{'+','.join(values)+'};')
    for name,a in [('embedding',model.embedding),('norm_att',model.norm_att),('norm_ffn',model.norm_ffn),('norm_final',model.norm_final)]:
        array(name,a,'float')
    rows=[]; metadata=[]
    for l in range(model.layers):
        for name in ['q','k','v','o','up','gate','down']:
            w,scale=model.qw[name]; rows.append((f'{name}{l}',w[l],scale[l]))
    w,scale=model.qw['cls']; rows.append(('cls',w,scale))
    for name,w,scale in rows:
        n,k=w.shape; padded=np.zeros(((n+3)//4*4,(k+15)//16*16),np.int8); padded[:n,:k]=w
        packed=padded.reshape(-1,4,padded.shape[1]//16,16).transpose(0,2,1,3).copy()
        array('w_'+name,packed,'int8_t'); array('s_'+name,scale,'float')
        metadata.append('{w_'+name+',s_'+name+f',{n},{k}'+'}')
    header.append(f'extern const proton_weight_t weights[{len(rows)}];')
    c.append('const proton_weight_t weights[]={'+','.join(metadata)+'};')
    array('expected_tokens',q_tokens,'int32_t'); array('expected_logits',q,'float')
    pieces=[model.tokens[t] for t in q_tokens]
    # Tokenizer is deployed separately from model math, with all vocabulary pieces.
    offsets=[0]; data=b''
    for piece in model.tokens: data+=piece+b'\0'; offsets.append(len(data))
    array('token_offsets',offsets,'uint32_t'); array('token_bytes',list(data),'uint8_t')
    (dest/'model_data.h').write_text('\n'.join(header)+'\n')
    (dest/'model_data.c').write_text('\n'.join(c)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    import sys
    export(Model(sys.argv[1]),sys.argv[2])
