import os
import rfim2d
from rfim2d import general_use


def test_load_data():
    data1 = general_use.load_svA()
    data2 = general_use.load_hvdMdh()

def test_generate_colors():
    num = 3
    colors = general_use.generate_colors(num)
    assert colors.shape==(num,3)
