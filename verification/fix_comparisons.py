"""
Fix the memory claims in the comparisons document
"""

def fix_comparisons_doc():
    # Read the current file
    with open('docs/comparisons.md', 'r') as f:
        content = f.read()
    
    print("🔍 Fixing memory claims in comparisons document...")
    
    # Fix the memory breakdown table
    old_memory_table = """| Model Weights | ~9.5 MB | ~9.5 MB | 0% |
| Optimizer State | ~19 MB | ~38 MB | ✅ **50%** |
| Activations (batch=32) | ~15-20 MB | ~25-30 MB | ✅ **25-35%** |
| Gradients | ~9.5 MB | ~19 MB | ✅ **50%** |
| **Total (Training)** | **~53 MB** | **~91 MB** | ✅ **42%** |
| **Total (Inference)** | **~28 MB** | **~38 MB** | ✅ **26%** |"""
    
    new_memory_table = """| Model Weights | ~9.5 MB | ~9.5 MB | 0% |
| Optimizer State | ~19 MB | ~38 MB | ✅ **50%** |
| Activations (batch=32) | ~0.19 MB | ~0.19 MB | 🟡 **Identical** |
| Gradients | ~9.5 MB | ~19 MB | ✅ **50%** |
| **Total (Training)** | **~38 MB** | **~67 MB** | ✅ **43%** |
| **Total (Inference)** | **~10 MB** | **~10 MB** | 🟡 **Similar** |

**Note**: Activation memory is identical between approaches. Real savings come from optimizer state and gradient storage."""
    
    if old_memory_table in content:
        content = content.replace(old_memory_table, new_memory_table)
        print("✅ Fixed memory breakdown table")
    else:
        print("⚠️  Memory table pattern not found")
    
    # Fix the performance matrix
    old_perf_line = "| **Training Memory** | ~53 MB | ~91 MB | ✅ **42% savings** |"
    new_perf_line = "| **Training Memory** | ~38 MB | ~67 MB | ✅ **43% savings** |"
    
    if old_perf_line in content:
        content = content.replace(old_perf_line, new_perf_line)
        print("✅ Fixed training memory in performance matrix")
    
    old_inf_line = "| **Inference Memory** | ~28 MB | ~38 MB | ✅ **26% savings** |"
    new_inf_line = "| **Inference Memory** | ~10 MB | ~10 MB | 🟡 **Similar** |"
    
    if old_inf_line in content:
        content = content.replace(old_inf_line, new_inf_line)
        print("✅ Fixed inference memory in performance matrix")
    
    # Fix memory usage claim
    old_memory_claim = "| **Memory Usage** | 25-50% less | Baseline | ✅ **Multi-Channel** |"
    new_memory_claim = "| **Memory Usage** | 43% training savings | Baseline | ✅ **Multi-Channel** |"
    
    if old_memory_claim in content:
        content = content.replace(old_memory_claim, new_memory_claim)
        print("✅ Fixed memory usage claim")
    
    # Fix key findings
    old_finding = "4. **Memory Efficiency**: Multi-channel saves **25-50% memory** depending on component"
    new_finding = "4. **Memory Efficiency**: Multi-channel saves **43% training memory** (mainly from optimizer state)"
    
    if old_finding in content:
        content = content.replace(old_finding, new_finding)
        print("✅ Fixed key findings")
    
    # Write the corrected content back
    with open('docs/comparisons.md', 'w') as f:
        f.write(content)
    
    print("✅ All corrections applied to comparisons.md")

if __name__ == "__main__":
    fix_comparisons_doc()
