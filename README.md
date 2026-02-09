# flash-attn-one-cuda

# Create the environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install the essentials
pip install torch numpy ninja
***
***
***
***
### STEP 1: BASELINE: 155.466 ms

### STEP 2: AFTER PARALLELIZING OUTER LOOP OVER BLOCKS: 22.979 ms (6.77x improvement over baseline)