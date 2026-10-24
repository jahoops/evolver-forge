import random
import time

print("Starting simple Forge test...")
for gen in range(1, 6):
    value = random.uniform(-10, 10)
    score = -abs(value - 3.1415)
    print(f"Generation {gen}: x={value:.3f}, score={score:.3f}")
    time.sleep(0.2)
print("Finished experiment.")
