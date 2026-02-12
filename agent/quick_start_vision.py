from main import run_agent
import sys
import traceback

def force_traceback(type, value, tb):
    print("\n" + "="*80)
    print("boom")
    traceback.print_exception(type, value, tb)
    print("="*80 + "\n")

sys.excepthook = force_traceback
# run a example for vision tasks. save the execution trace to outputs/blink_spatial
run_agent("../tasks/blink_spatial/processed/val_Spatial_Relation_1", "../outputs/blink_spatial", task_type="vision")