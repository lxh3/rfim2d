from rfim2d import save_and_load


def test_load_data():
    data1 = save_and_load.load_svA()
    print(len(data1))
    data2 = save_and_load.load_hvdMdh()
    print(len(data2))
