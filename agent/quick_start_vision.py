from main import run_agent
import logging

# AutoGen이 내부적으로 사용하는 로거의 에러 출력을 가로챕니다.
def force_crash_on_error():
    import autogen
    # 에러를 그냥 출력만 하고 넘어가는 지점을 찾아서 강제로 멈추게 함
    original_log_error = logging.Logger.error
    def new_log_error(self, msg, *args, **kwargs):
        if "unsupported operand type" in str(msg):
            import traceback
            traceback.print_stack() # 여기서 터지기까지의 경로를 강제로 다 출력
        original_log_error(self, msg, *args, **kwargs)
    logging.Logger.error = new_log_error

force_crash_on_error()
# run a example for vision tasks. save the execution trace to outputs/blink_spatial
run_agent("../tasks/blink_spatial/processed/val_Spatial_Relation_1", "../outputs/blink_spatial", task_type="vision")