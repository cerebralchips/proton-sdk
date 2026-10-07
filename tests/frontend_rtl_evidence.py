"""Strict full-output oracle for a single imported decoder invocation on RTL."""
import re
import numpy as np

SHAPES={'logits':(512,),'keys':(5,64,32),'values':(5,64,32)}


def check(text,reference,precision):
    if any(s in text for s in ['FAIL','TRAP','Simulation timeout','%Error']):
        raise ValueError('RTL failure/timeout marker')
    if text.count('*** SUCCESS ***')!=1 or text.count(f'RESULT: PASS frontend_{precision} ')!=1:
        raise ValueError('Missing unique RTL completion')
    if text.count(f'FRONTEND START precision={precision} steps=1')!=1 or text.count('FRONTEND INVOKE steps=1')!=1:
        raise ValueError('Expected exactly one decoder invocation')
    cycles=re.findall(r'Executed cycles:\s*(\d+)',text)
    invoke=re.findall(rf'FRONTEND steps=1 precision={precision} invoke_cycles=(\d+)',text)
    tokens=re.findall(r'TOKEN pos=(\d+) id=(\d+)',text)
    if len(cycles)!=1 or int(cycles[0])<=0 or len(invoke)!=1 or not 0<int(invoke[0])<int(cycles[0]):
        raise ValueError('Missing/invalid RTL cycle evidence')
    if len(tokens)!=1 or tokens[0][0]!='0': raise ValueError('Expected exactly one output token')
    outputs={name:np.empty(int(np.prod(shape)),np.float32) for name,shape in SHAPES.items()}
    cursors={name:0 for name in SHAPES}
    for line in text.splitlines():
        if not line.startswith(('DATA ','ZERO ')): continue
        fields=line.split()
        if len(fields)<4: raise ValueError('Truncated tensor record')
        kind,name,index,count=fields[:4]
        index,count=int(index),int(count)
        if name not in outputs or index!=cursors[name] or count<1 or index+count>outputs[name].size:
            raise ValueError('Tensor coverage gap/overlap/range')
        if kind=='ZERO':
            if len(fields)!=4: raise ValueError('Malformed zero record')
            values=np.zeros(count,np.float32)
        else:
            if len(fields)!=4+count or any(not re.fullmatch('[0-9a-f]{8}',b) for b in fields[4:]):
                raise ValueError('Malformed tensor bits')
            values=np.array([int(b,16) for b in fields[4:]],np.uint32).view(np.float32)
        outputs[name][index:index+count]=values
        cursors[name]+=count
    expected=np.load(reference)
    metrics={}
    for name,shape in SHAPES.items():
        if cursors[name]!=outputs[name].size: raise ValueError('Incomplete tensor: '+name)
        actual=outputs[name].reshape(shape)
        want=expected[name]
        if want.shape!=shape or want.dtype!=np.float32: raise ValueError('Unexpected reference layout')
        if not np.isfinite(actual).all() or not np.isfinite(want).all(): raise ValueError('Nonfinite tensor')
        err=np.abs(actual-want)
        if np.any(err>1e-4+1e-5*np.abs(want)): raise ValueError('Tensor mismatch: '+name)
        markers=re.findall(rf'TENSOR PASS name={name} elements=(\d+) max_error_bits=([0-9a-f]+)',text)
        if len(markers)!=1 or int(markers[0][0])!=actual.size: raise ValueError('Missing target tensor self-check')
        reported=np.array([int(markers[0][1],16)],np.uint32).view(np.float32)[0]
        if not np.isfinite(reported) or reported!=np.max(err): raise ValueError('Target/host error evidence differs')
        metrics[name]={'elements':int(actual.size),'max_abs_error':float(err.max())}
    token=int(tokens[0][1])
    if token!=int(np.argmax(expected['logits'])) or token!=int(np.argmax(outputs['logits'])):
        raise ValueError('Greedy token mismatch')
    if 'MEMORY_GUARD PASS bounded_heap exhaustion_rejected stack_canary' not in text:
        raise ValueError('Missing memory guard evidence')
    heap=re.findall(rf'RESULT: PASS frontend_{precision} heap_peak=(\d+)',text)
    if len(heap)!=1: raise ValueError('Missing heap evidence')
    real=re.findall(r'^real ([0-9.]+)$',text,re.M)
    return {'result':'PASS','execution':'CVA6/Ara RTL via Verilator; scalar CPU code generation',
        'rtl_cycles':int(cycles[0]),'invoke_cycles':int(invoke[0]),'token':token,
        'outputs':metrics,'compared_elements':sum(x['elements'] for x in metrics.values()),
        'heap_peak_bytes':int(heap[0]),'simulator_real_seconds':float(real[-1]) if real else None}


def negative_checks(text,reference,precision):
    mutations={'timeout':text+'\nSimulation timeout\n',
      'missing_success':text.replace('*** SUCCESS ***',''),
      'second_token':text+'\nTOKEN pos=1 id=2\n',
      'missing_cache':re.sub(r'^(DATA|ZERO) keys .*\n','',text,count=1,flags=re.M)}
    for name in SHAPES:
        mutations['corrupt_'+name]=re.sub(rf'^(DATA {name} \d+ \d+ )([0-9a-f]{{8}})',
              r'\g<1>7fc00000',text,count=1,flags=re.M)
    for label,bad in mutations.items():
        if bad==text: raise ValueError('Negative mutation not applied: '+label)
        try: check(bad,reference,precision)
        except ValueError: continue
        raise ValueError('Invalid evidence accepted: '+label)
    return {'result':'PASS','rejected':list(mutations),'additional_rtl_invocations':0}
