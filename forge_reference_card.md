# ⚙️ FORGE CO-EVOLUTION VISUAL REFERENCE

### 🧩 One-Line Overview  
**Neural Network (NN)** learns by experience → **Rules** learn by reasoning → **Coevolution Loop** balances both.

---

## 📈 Key Visuals & Interpretation

### 1. Metrics Split Plot  
🔵 NN Accuracy | 🟢 Rule Fidelity | 🟠 Disagreement | 🟣 Consistency Loss  
- Accuracy ↑ + Fidelity ↑ → Healthy  
- Accuracy ↑ + Fidelity ↓ → Rules are lagging  
- Both flat too early → Stagnation  
- Consistency Loss ↓ → Good convergence  

### 2. Rule–NN Consistency Plot  
- Tight diagonal → NN & Rules align  
- Broad scatter → Rules too coarse  
- Double bands → Multiple strategies emerging  

### 3. Causal Graph  
🟩 = Positive effect | 🟥 = Negative effect  
- Green core arrows stable → system identifies key factors  
- Red noise arrows stay peripheral → robustness  

### 4. Robustness Plot  
Curves under noise.feature, noise.label, noise.target  
- Gradual declines = robust  
- Sharp drops = over-fit to clean data  

### 5. Generation Metrics Table  
Watch for:
- `nn_accuracy` trending upward  
- `rule_fidelity` tracking accuracy  
- `rule_mutations` stable (4–8)  
- `consistency_loss` and `avg_disagreement` declining  

### 6. Canon Rules  
- Short, simple thresholds → clean generalization  
- Deep nested branches → complexity may be high  
- Repeating rules → maybe mutation diversity is low  

---

## 📋 Quick Commands  
- Run session:  
  ```bash
  poetry run python -m projects.coevolve_base.coevolve --generations 5
