import torch
from welford_torch import Welford


def test_init():
    a = torch.tensor([[0]])
    w = Welford(a)
    assert w.count == 1
    assert torch.allclose(w.mean, torch.tensor([0], dtype=torch.float32))
    assert torch.all(torch.isnan(w.var_s))
    assert torch.allclose(w.var_p, torch.tensor([0], dtype=torch.float32))

    a = torch.tensor([[0], [1]])
    w = Welford(a)
    assert w.count == 2
    assert torch.allclose(w.mean, torch.tensor([0.5]))
    assert torch.allclose(w.var_s, torch.tensor([0.5]))
    assert torch.allclose(w.var_p, torch.tensor([0.25]))

    a = torch.tensor([[0, 100], [1, 110], [2, 120], [3, 130], [4, 140]])
    w = Welford(a)
    assert w.count == 5
    assert torch.allclose(w.mean, torch.tensor([2, 120], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([2.5, 250], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([2, 200], dtype=torch.float32))

    a = torch.arange(60).reshape(3, 4, 5)
    w = Welford(a)
    assert w.mean.shape == w.var_s.shape == w.var_p.shape == (4, 5)
    a = a.to(torch.float32)
    assert torch.allclose(w.mean, torch.mean(a, axis=0))
    assert torch.allclose(w.var_s, torch.var(a, axis=0, unbiased=True))
    assert torch.allclose(w.var_p, torch.var(a, axis=0, unbiased=False))


def test_add():
    w = Welford()
    w.add(torch.tensor([0, 100]))
    assert torch.allclose(w.mean, torch.tensor([0, 100], dtype=torch.float32))
    assert torch.all(torch.isnan(w.var_s))
    assert torch.allclose(w.var_p, torch.tensor([0, 0], dtype=torch.float32))

    w.add(torch.tensor([1, 110]))
    assert torch.allclose(w.mean, torch.tensor([0.5, 105], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([0.5, 50], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([0.25, 25], dtype=torch.float32))

    w = Welford()
    w.add(torch.tensor([[0, 100, 1000], [2, 220, 2200]]))
    w.add(torch.tensor([[1, 110, 1100], [2, 220, 2200]]))
    assert torch.allclose(w.mean, torch.tensor([[0.5, 105, 1050], [2, 220, 2200]], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([[0.5, 50, 5000], [0, 0, 0]], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([[0.25, 25, 2500], [0, 0, 0]], dtype=torch.float32))


def test_add_all():
    w = Welford()
    a = torch.tensor([[0, 100], [1, 110], [2, 120], [3, 130], [4, 140]])
    w.add_all(a)
    assert w.count == 5
    assert torch.allclose(w.mean, torch.tensor([2, 120], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([2.5, 250], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([2, 200], dtype=torch.float32))

    w = Welford()
    a = torch.tensor([[0, 100]])
    w.add_all(a)
    assert w.count == 1
    assert torch.allclose(w.mean, torch.tensor([0, 100], dtype=torch.float32))
    assert torch.all(torch.isnan(w.var_s))
    assert torch.allclose(w.var_p, torch.tensor([0, 0], dtype=torch.float32))

    a = torch.tensor([[1, 110], [2, 120], [3, 130], [4, 140]])
    w.add_all(a)
    assert w.count == 5
    assert torch.allclose(w.mean, torch.tensor([2, 120], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([2.5, 250], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([2, 200], dtype=torch.float32))

    w = Welford()
    a = torch.tensor([[[0, 100, 1000], [2, 220, 2200]], [[1, 110, 1100], [2, 220, 2200]]])
    w.add_all(a)
    assert w.count == 2
    assert torch.allclose(w.mean, torch.tensor([[0.5, 105, 1050], [2, 220, 2200]], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([[0.5, 50, 5000], [0, 0, 0]], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([[0.25, 25, 2500], [0, 0, 0]], dtype=torch.float32))


def test_rollback():
    a = torch.tensor([[0, 100]])
    w = Welford(a)

    a = torch.tensor([1, 110])
    w.add(a)
    assert torch.allclose(w.mean, torch.tensor([0.5, 105], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([0.5, 50], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([0.25, 25], dtype=torch.float32))

    w.rollback()
    a = torch.tensor([2, 120])
    w.add(a)
    assert w.count == 2
    assert torch.allclose(w.mean, torch.tensor([1, 110], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([2, 200], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([1, 100], dtype=torch.float32))

    a = torch.tensor([[2, 120], [3, 130]])
    w.add_all(a)
    w.rollback()
    assert w.count == 2
    assert torch.allclose(w.mean, torch.tensor([1, 110], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([2, 200], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([1, 100], dtype=torch.float32))

    w = Welford()
    w.add(torch.tensor([[0, 100, 1000], [2, 220, 2200]]))
    w.add(torch.tensor([[1, 110, 1100], [2, 220, 2200]]))
    w.rollback()
    w.add(torch.tensor([[2, 120, 1200], [2, 220, 2200]]))
    assert torch.allclose(w.mean, torch.tensor([[1.0, 110, 1100], [2, 220, 2200]], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([[2.0, 200, 20000], [0, 0, 0]], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([[1, 100, 10000], [0, 0, 0]], dtype=torch.float32))


def test_merge():
    a = torch.tensor([[0, 100], [1, 110], [2, 120], [3, 130], [4, 140]])
    wa = Welford(a)
    b = torch.tensor([[5, 150], [6, 160], [7, 170]])
    wb = Welford(b)
    wa.merge(wb)
    assert wa.count == 8
    assert torch.allclose(wa.mean, torch.tensor([3.5, 135], dtype=torch.float32))
    assert torch.allclose(wa.var_s, torch.tensor([6, 600], dtype=torch.float32))
    assert torch.allclose(wa.var_p, torch.tensor([5.25, 525], dtype=torch.float32))

    wa.rollback()
    c = torch.tensor([[5, 150], [6, 160], [7, 170], [8, 180], [9, 190], [10, 200]])
    wc = Welford(c)
    wa.merge(wc)
    assert wa.count == 11
    assert torch.allclose(wa.mean, torch.tensor([5, 150], dtype=torch.float32))
    assert torch.allclose(wa.var_s, torch.tensor([11, 1100], dtype=torch.float32))
    assert torch.allclose(wa.var_p, torch.tensor([10, 1000], dtype=torch.float32))

    a = torch.tensor([[0, 100]])
    wa = Welford(a)
    b = torch.tensor([[1, 110]])
    wb = Welford(b)
    wa.merge(wb)
    assert wa.count == 2
    assert torch.allclose(wa.mean, torch.tensor([0.5, 105], dtype=torch.float32))
    assert torch.allclose(wa.var_s, torch.tensor([0.5, 50], dtype=torch.float32))
    assert torch.allclose(wa.var_p, torch.tensor([0.25, 25], dtype=torch.float32))

    wa = Welford(torch.tensor([[[0, 100, 1000], [2, 220, 2200]]]))
    wb = Welford(torch.tensor([[[1, 110, 1100], [2, 220, 2200]]]))
    wa.merge(wb)
    assert wa.count == 2
    assert torch.allclose(wa.mean, torch.tensor([[0.5, 105, 1050], [2, 220, 2200]], dtype=torch.float32))
    assert torch.allclose(wa.var_s, torch.tensor([[0.5, 50, 5000], [0, 0, 0]], dtype=torch.float32))
    assert torch.allclose(wa.var_p, torch.tensor([[0.25, 25, 2500], [0, 0, 0]], dtype=torch.float32))

def test_all():
    tests = [
        test_init,
        test_add,
        test_add_all,
        test_rollback,
        test_merge,
    ]
    for test in tests:
        try:
            print(f"# Running test {test.__name__}...")
            test()
            print(f"Test {test.__name__} passed")
        except AssertionError as e:
            print(f"Test {test.__name__} failed: {e}")

if __name__ == "__main__":
    test_all()
