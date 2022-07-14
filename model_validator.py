"""
tflite-benchmark
-------------------
Measuring latency, accuracy, compatibilites of target runtime for given tflite models
by comparing output tensors inferenced by both; reference(host PC) and target.
"""
from __future__ import print_function
from logging import raiseExceptions
import os
import subprocess
import shlex
import re
import argparse
import numpy as np
import tensorflow as tf


GLOBAL_SETTING = {
    'iteration' : 1,
    'rtol' : 1e-5,
    'atol' : 1e-5,
    'target' : 'heaan',
}


def convert_to_list(x):
    """Upcasting single object to list"""
    if not isinstance(x, list):
        x = [x]
    return x


def run_tflite_on_host(tflite_file, inputs):
    """Actual processing of tflite model inference on Host device.

    Args:
        tflite_file (string): path to tflite model
        inputs (list of numpy objects): raw inputs to be transformed to input tensors.
        use_target (str, optional): Defaults to '--use_npu=true'.

    Returns:
        outputs : numpy object lists containing data of output tensors.
    """
    with open(tflite_file, 'rb') as f:
        model_buf = f.read()
    inputs = convert_to_list(inputs)

    runtime = tf.lite.Interpreter(model_content=model_buf)
    input_details = runtime.get_input_details()
    output_details = runtime.get_output_details()

    for i, input_detail in enumerate(input_details):
        runtime.resize_tensor_input(input_detail['index'], inputs[i].shape)
    runtime.allocate_tensors()

    assert len(inputs) == len(input_details)

    for i, input_detail in enumerate(input_details):
        runtime.set_tensor(input_detail['index'], inputs[i])
    runtime.invoke()

    print(runtime.get_tensor(input_details[i]['index']))

    outputs = []
    for _, output_detail in enumerate(output_details):
        shape = output_detail['shape']
        outputs.append(np.reshape(runtime.get_tensor(output_detail['index']), shape))
    return outputs


def run_tflite_on_heaan(tflite_file, inputs):
    """Actual processing of tflite model inference on HEaaN framework.

    Args:
        tflite_file (string): path to tflite model
        inputs (list of numpy objects): raw inputs to be transformed to input tensors.
        use_target (str, optional): Defaults to '--use_npu=true'.

    Returns:
        outputs : numpy object lists containing data of output tensors.
    """
    with open(tflite_file, 'rb') as f:
        model_buf = f.read()
    inputs = convert_to_list(inputs)

    outputs = []
    # Do something like below.
#    import pyHeaan.numpy as henp
#    import pyHeaan.runtime as hert
#    from pyHeaan.securitySpec import HeaanPresets
#
#    or python interface accessing target runtime over RPC/gRPC?
#
#    runtime = hert.runtime(
#                workload = {
#                    'callable': False,
#                    'type': 'tflite',
#                    'binary': model_buf,
#                },
#                specification = HeaanPresets('Venti'),
#                backend_type = None)
#
#    with runtime.verified_storage() as vs:
#        try:
#            vs.createt('')
#        except Exception as exc:
#            raise RuntimeError(f'Failed to create secure storage') from exc
#        vs.hide()
#
#    secure_inputs = runtime.encrypt(henp.asanyarray(inputs))
#    scrambled_outputs = runtime.execute(input = privacy, )
#
#    outputs = henp.toarray(runtime.decrypt(scrambled_outputs))
    return outputs


def run_tflite_on_android(tflite_file, inputs, use_target='--use_npu=true'):
    """Actual processing of tflite model inference on Android target.

    Args:
        tflite_file (string): path to tflite model
        inputs (list of numpy objects): raw inputs to be transformed to input tensors.
        use_target (str, optional): Defaults to '--use_npu=true'.

    Returns:
        outputs : numpy object lists containing data of output tensors.
    """
    with open(tflite_file, 'rb') as f:
        model_buf = f.read()
    inputs = convert_to_list(inputs)

    runtime = tf.lite.Interpreter(model_content=model_buf)
    input_details = runtime.get_input_details()
    output_details = runtime.get_output_details()

    root = '/data/local/tmp'
    input_layer = ''
    input_layer_shape = ''
    tails = ''
    try:
        r = subprocess.check_call(f'adb push {tflite_file} {root}', shell=True)
    except Exception as exc:
        raise RuntimeError(f'Failed to push tflite model {tflite_file}') from exc
    for i, input_detail in enumerate(input_details):
        name = input_detail['name']
        shape = input_detail['shape'].tolist()
        name = name.replace(':','_')
        shape = ','.join([str(_) for _ in shape])
        with open(f'.invals{i}', 'w').encoding('UTF8') as f:
            np.array(inputs[i]).tofile(f)
        try:
            r = subprocess.check_call(f'adb push .invals{i} {root}', shell=True)
        except Exception as exc:
            raise RuntimeError(f'Failed to push input tensor file .invals{i}') from exc
        tails += f'{name}:{root}/.invals{i},'
        input_layer += f'{name},'
        input_layer_shape += f'{shape}:'

    input_layer = input_layer.rstrip(',')
    input_layer_shape = input_layer_shape.rstrip(':')
    tails = tails.rstrip(',')
    split_file_path = tflite_file.split('/')
    file_name = split_file_path[-1]

    command = f'adb shell {root}/benchmark_model --graph={root}/{file_name} \
        --num_runs=1 --min_secs=0 --max_secs=0 {use_target} --input_layer={input_layer} \
        --input_layer_shape={input_layer_shape} --input_layer_value_files={tails} \
        --save_outputs_in_file={root}/.outvals'

    try:
        subprocess.check_output(
            shlex.split('adb root'), stderr=subprocess.STDOUT
        ).decode('utf-8')
        subprocess.check_call(command, shell=True)
        subprocess.check_call(f'adb pull {root}/.outvals .', shell=True)
    except Exception as exc:
        raise RuntimeError(f'Failed to run {root}/benchmark_model') from exc

    outputs = []
    for i, output_detail in enumerate(output_details):
        shape = output_detail['shape']
        for _, shape_ in enumerate(shape):
            read_size *= int(shape_)

        with open('.outvals', 'rb') as f:
            f.seek(read_size * i)
            outbuf = f.read(read_size)

        outputs.append(np.reshape(np.frombuffer(outbuf, output_detail['dtype'], shape)))
    return outputs


def probe_adb_device():
    """For android targets, make sure if ADB is supported and the target is connected to Host PC.
    Returns:
        dev_id : unique device ID granted by ADB runtime.
    """
    try:
        log = subprocess.check_output(
                shlex.split('adb devices'),
                stderr=subprocess.STDOUT).decode('utf-8')
    except Exception as exc:
        raise RuntimeError('Check first if adb works well in your host.') from exc
    rex = re.compile('(?P<dev_id>[A-Z|0-9]+)\s+[a-z]+', re.DOTALL)
    dev_lists = list(rex.finditer(log))
    if len(dev_lists) > 1 and not os.getenv('ANDROID_SERIAL'):
        raise RuntimeError(
                'Multiple devices seem to be connected to your host, \
                 which is not desirable for this test scheme. \
                 Remain only one device connected, then try again.')
    if len(dev_lists) == 1 and log.find('unauthorized') != -1:
        raise RuntimeError(
                'Make sure if \'USB debugging\' in \'developer options\' \
                is allowed first, then try again.')
    if len(dev_lists) == 0:
        raise RuntimeError('Make sure if your device is configured to enable \
                \'developer options\' first, then try again.')
    return dev_lists[0].group('dev_id')


def compare_output(from_host, from_target, metric='Strict', k=0):
    """Version of np.testing.assert_allclose with `atol` and `rtol` fields set
    in reasonable defaults.

    Arguments `from_host` and `from_target` are not interchangeable, since the function
    compares the `abs(actual-desired)` with `atol+rtol*abs(desired)`.  Since we
    often allow `desired` to be close to zero, we generally want non-zero `atol`.
    """
    for i in enumerate(from_host):
        o_host = np.asanyarray(from_host[i])
        o_targ = np.asanyarray(from_target[i])

        assert o_host.shape == o_targ.shape

        if metric is None or len(o_host.shape) > 2 :
            metric = 'Strict'

        if metric == 'Strict':
            np.testing.assert_allclose(
                o_host, o_targ,
                rtol=GLOBAL_SETTING['rtol'],
                atol=GLOBAL_SETTING['atol'],
                verbose=True
            )
        elif metric == 'TopK':
            assert o_host.size > k

            if len(o_host.shape) == 1:
                topk_host = (-o_host).argsort()[:k]
                topk_targ = (-o_targ).argsort()[:k]
            else:
                topk_host = (-o_host).argsort()[:,:k]
                topk_targ = (-o_targ).argsort()[:,:k]
            np.testing.assert_almost_equal(np.sort(topk_host), np.sort(topk_targ))


class ModelValidator:
    """Decorator for tflite model validation"""
    def __init__(self, f):
        self.inner_f = f
        if GLOBAL_SETTING['target'] == 'android':
            self.target_id = probe_adb_device()

    def __call__(self, *args, **kwargs):
        tflite_file, inputs = self.inner_f(self)

        x = run_tflite_on_host(tflite_file, inputs)
        if GLOBAL_SETTING['target'] == 'heaan':
            y = run_tflite_on_heaan(tflite_file, inputs)
        elif GLOBAL_SETTING['target'] == 'andriod':
            y = run_tflite_on_android(tflite_file, inputs)
        else:
            raise Exception('Not yet supported')

        for _ in range(GLOBAL_SETTING['iteration']):
            compare_output(
                x,
                y,
                kwargs.get('metric'),
                kwargs.get('k'),
            )
        print("###### test result : Pass #####")


class TestRecipes:
    """Add test case each per a single tflite mode here with @ModelValidator"""
    def __init__(self):
        return

    @ModelValidator
    def do_dummy(self):
        """
        Responsible to return the **tuple** in that
        - **tflite file path**
        - **inputs** composed of numpy arrays for the given operation.
        must be given by the author.
        Returns:
            file_path : directory to tflite model file.
            inputs object : can be both a list of numpy objects or a single numpy object.
        """
        input_a = np.random.uniform(size=(2, 2)).astype('float32')
        input_b = np.random.uniform(size=(2, 2)).astype('float32')
        file_path = os.path.join(os.path.dirname(__file__), 'models/', 'add_fp16.tflite')
        inputs = [input_a, input_b]
        return file_path, inputs

    @ModelValidator
    def do_mul(self):
        """model containing tf.mul node only"""
        input_a = np.random.uniform(size=(2, 2), low=0, high=10).astype('int8')
        input_b = np.random.uniform(size=(2, 2), low=0, high=10).astype('int8')
        file_path = os.path.join(os.path.dirname(__file__), 'models/', 'mul_int8.tflite')
        return file_path, [input_a, input_b]

    @ModelValidator
    def do_argmax(self):
        """model containing tf.argmax node only"""
        inputs = np.random.uniform(size=(1, 720, 1080, 3)).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'argmax.tflite'), inputs

    @ModelValidator
    def do_relu(self):
        """model containing tf.relu node only"""
        inputs = np.random.uniform(size=(32), low=-1., high=1.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'relu.tflite'), inputs

    @ModelValidator
    def do_relu6(self):
        """model containing tf.relu6 node only"""
        inputs = np.random.uniform(size=(32), low=-1., high=1.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'relu6.tflite'), inputs

    @ModelValidator
    def do_elu(self):
        """Elu"""
        inputs = np.random.uniform(size=(32), low=-1., high=1.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'elu.tflite'), inputs

    @ModelValidator
    def do_prelu(self):
        """Prelu"""
        inputs = np.random.uniform(size=(32), low=-1., high=1.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'prelu.tflite'), inputs

    @ModelValidator
    def do_tanh(self):
        """Tanh"""
        inputs = np.random.uniform(size=(32), low=-1., high=1.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'tanh.tflite'), inputs

    @ModelValidator
    def do_dense(self):
        """Dense"""
        inputs = np.random.uniform(size=(32), low=-3., high=3.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'dense.tflite'), inputs

    @ModelValidator
    def do_depthwiseconv2d(self):
        """DepthwiseConv2D"""
        inputs = np.random.uniform(size=(1, 32, 32, 32), low=-3., high=3.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'depthwise_conv2d.tflite'), inputs

    @ModelValidator
    def do_transpose(self):
        """Transpose"""
        inputs = np.random.uniform(size=(3, 2), low=-10., high=10.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'transpose.tflite'), inputs

    @ModelValidator
    def do_conv2d(self):
        """Conv2D"""
        inputs = np.random.uniform(size=(1, 512, 512, 3), low=0., high=16.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'conv2d.tflite'), inputs

    @ModelValidator
    def do_maxpool2d(self):
        """MaxPool2D"""
        inputs = np.random.uniform(size=(1, 32, 32, 1), low=0., high=16.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'maxpool2d.tflite'), inputs

    @ModelValidator
    def do_pad(self):
        """Pad"""
        inputs = np.random.uniform(size=(1, 3, 3, 1), low=0., high=5.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'pad.tflite'), inputs

    @ModelValidator
    def do_densenet(self):
        """DenseNet E2E model"""
        inputs = np.random.uniform(size=(1, 224, 224, 3), low=0., high=255.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'densenet.tflite'), inputs

    @ModelValidator
    def do_inception(self):
        """Inception V3 E2E model"""
        inputs = np.random.uniform(size=(1, 299, 299, 3), low=0., high=255.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'inception_v3.tflite'), inputs

    @ModelValidator
    def do_mobilenet(self):
        """Mobilenet V2 E2E model"""
        inputs = np.random.uniform(size=(1, 224, 224, 3), low=0., high=255.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'mobilenet_v2.tflite'), inputs

    @ModelValidator
    def do_split(self):
        """Split"""
        inputs = np.random.uniform(size=(1, 720, 4, 3), low=-128., high=127.).astype('float32')
        return os.path.join(os.path.dirname(__file__), 'models/', 'split_fp32.tflite'), inputs

    @ModelValidator
    def do_mobilebert(self):
        """MobileBERT E2E model"""
        input1 = np.random.uniform(size=(1, 384), low=0, high=255).astype('int32')
        input2 = np.random.uniform(size=(1, 384), low=0, high=255).astype('int32')
        input3 = np.random.uniform(size=(1, 384), low=0, high=255).astype('int32')
        file_path = os.path.join(os.path.dirname(__file__), 'models/', 'mobilebert.tflite')
        inputs = [input1, input2, input3]
        return file_path, inputs


def parse_args():
    """
    Returns:
        argument list
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', action='store',
        choices=[
            'all',
            'split',
            'pad',
            'mul',
            'conv2d',
            'maxpool2d',
            'reshape',
            'depthwiseconv2d',
            'dense',
            'argmax',
            'relu',
            'relu6',
            'elu',
            'prelu',
            'tanh',
            'densenet',
            'inception',
            'mobilenet',
            'mobilebert'],
        default='all', help="test model"
    )
    parser.add_argument(
        '--target', action='store',
        choices = [
            'heaan',
            'android',
            'ios',
            'raspberrypi'],
        default='heaan', help='target specified for benchmark'
    )
    args = parser.parse_args()
    return args


test_func = {
    'split' : (lambda : TestRecipes().do_split()),
    'pad' : (lambda : TestRecipes().do_pad()),
    'conv2d' : (lambda : TestRecipes().do_conv2d()),
    'mul' : (lambda : TestRecipes().do_mul()),
    'maxpool2d' : (lambda : TestRecipes().do_maxpool2d()),
    'transpose' : (lambda : TestRecipes().do_transpose()),
    'depthwiseconv2d' : (lambda : TestRecipes().do_depthwiseconv2d()),
    'dense' : (lambda : TestRecipes().do_dense()),
    'argmax' : (lambda : TestRecipes().do_argmax()),
    'relu' : (lambda : TestRecipes().do_relu()),
    'relu6' : (lambda : TestRecipes().do_relu6()),
    'prelu' : (lambda : TestRecipes().do_prelu()),
    'tanh' : (lambda : TestRecipes().do_tanh()),
    'densenet' : (lambda : TestRecipes().do_densenet()),
    'inception' : (lambda : TestRecipes().do_inception()),
    'mobilenet' : (lambda : TestRecipes().do_mobilenet()),
    'mobilebert' : (lambda : TestRecipes().do_mobilebert()),
}


def main(args):
    """ Batch test for tflite models

    Args:
        model (string) optional(default 'all')
            : valid tflite model, try `python3 ModelValidator.py --help`.
        target (string) optional(default 'heaan')
            : benchmarking target, try `python3 ModelValidator.py --help`.
    """
    model  = args.model
    GLOBAL_SETTING['target'] = args.target
    if model == 'all':
        for func in test_func.values():
            func()
    else:
        test_func[model]()


if __name__ == "__main__":
    main(parse_args())
    