# flash-attn-one-cuda

# Create the environment

python3 -m venv venv

# Activate it

source venv/bin/activate

# Install the essentials

pip install torch numpy ninja

---

---

---

---

### STEP 1: BASELINE: 155.466 ms

### STEP 2: AFTER PARALLELIZING OUTER LOOP OVER BLOCKS: 22.979 ms (6.77x faster than baseline)

### STEP 3: RESOLVED BANK CONFLICTS: 9.386 ms (2.5x faster than step 2)

### STEP 4: RESOLVED UNCOALESCING: 9.360 (virtually no gain)

### STEP 5: VECTORIZATION: 6.318 (1.49x faster than step 4)

tried moving Q into regs but way too much register pressure caused minor slowdown

### STEP 6: CLEANUP (removing l and m arrays, etc): 5.304 (1.2x faster than step 5)

### STEP 7: FP32 to FP16 - 6.226 ms (seems like i lost)
## virtually no gain but necessary step for tensor core compatability