# flash-attn-one-cuda

## Cloud Session Startup

### 1. Clone and Enter Repo
```bash
git clone https://github.com/ayush-os/flash-attn-one-cuda.git
cd flash-attn-one-cuda
```

### 2. Run Setup Script

```bash
source setup_env.sh
```

### 3. Run Verification

```Bash
uv run python test_flash_attn.py
```
### Recompile Kernel
If you change .cu or .cpp files, run
```Bash
uv pip install -e .
```

---