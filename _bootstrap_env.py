import os
import io
import sys
for _env in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env, '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('JOBLIB_TEMP_FOLDER', os.environ.get('TEMP', '/tmp'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

def _patch_stdio_utf8():
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name)
        try:
            encoding = (getattr(stream, 'encoding', '') or 'ascii').lower()
            if 'utf' not in encoding:
                buffer = getattr(stream, 'buffer', None)
                if buffer is not None:
                    new_stream = io.TextIOWrapper(buffer, encoding='utf-8', errors='replace', line_buffering=True)
                    setattr(sys, stream_name, new_stream)
        except Exception:
            pass
_patch_stdio_utf8()
del _patch_stdio_utf8
