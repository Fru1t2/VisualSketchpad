import builtins

def debug_add(self, other):
    if other is None and isinstance(self, int):
        import traceback
        import sys
        print("\n" + "!"*60)
        print("🚨 [CRITICAL DEBUG] int + None 발생 감지!")
        traceback.print_stack(file=sys.stdout)
        print("!"*60 + "\n")
        raise TypeError("STOP HERE: Found the culprit!")
    return original_add(self, other)


from main import run_agent

# run a example for vision tasks. save the execution trace to outputs/blink_spatial
run_agent("../tasks/blink_spatial/processed/val_Spatial_Relation_1", "../outputs/blink_spatial", task_type="vision")