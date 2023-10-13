import pytest
from NICEClass import NICEClass

@pytest.fixture
def brain():
    return NICEClass()

def test_generate_id(brain):
    id_ = brain.GenerateID(0.5)
    assert isinstance(id_, str)
    assert len(id_) == brain.RULES['node_space']

def test_random_weight(brain):
    weight = brain.RandomWeight()
    assert brain.RULES['weights_space'][0] <= weight <= brain.RULES['weights_space'][1]

def test_add_node(brain):
    my_brain = brain.Brain()
    original_nodes = len(my_brain['nodes'])
    brain.AddNode(my_brain)
    assert len(my_brain['nodes']) == original_nodes + 1

# def test_add_node(brain):
#     original_nodes = len(brain.Brain['nodes'])
#     brain.AddNode(brain.Brain)
#     assert len(brain.Brain['nodes']) == original_nodes + 1

