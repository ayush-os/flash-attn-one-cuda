# flash-attn-one-cuda

# Create the environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install the essentials
pip install torch numpy ninja


### BASELINE: 155.466 ms

### AFTER PARALLELIZING OUTER LOOP OVER BLOCKS: 22.979 ms