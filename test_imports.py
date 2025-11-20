import importlib
import traceback

modules = ['interface', 'json_interpreter.trainer']
for mname in modules:
    try:
        m = importlib.import_module(mname)
        print(f'IMPORT_OK: {mname} ->', getattr(m, '__file__', '<built-in>'))
        if mname == 'interface':
            print('HAS_CLASS_TensorSuggestLiteUI =', hasattr(m, 'TensorSuggestLiteUI'))
        if mname == 'json_interpreter.trainer':
            print('HAS_train_from_json =', hasattr(m, 'train_from_json'))
            print('HAS_convert_to_tflite =', hasattr(m, 'convert_to_tflite'))
    except Exception as e:
        print(f'IMPORT_FAIL: {mname}', e)
        traceback.print_exc()

