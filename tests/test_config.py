import yaml

from delta.ml import model_parser

def test_model_parser():
    test_str = '''
    params:
        v1 : 10
    layers:
    - Flatten:
        input_shape: in_shape
    - Dense:
        units: v1
        activation : relu
    - Dense:
        units: out_shape
        activation : softmax
    '''

    input_shape = (17, 17, 8)
    output_shape = 3
    params_exposed = { 'out_shape' : output_shape, 'in_shape' : input_shape}
    model = model_parser.model_from_dict(yaml.safe_load(test_str), params_exposed)()
    model.compile(optimizer='adam', loss='mse')

    assert model.input_shape[1:] == input_shape
    assert model.output_shape[1] == output_shape
    assert len(model.layers) == 3
