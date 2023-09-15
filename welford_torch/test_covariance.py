# Source: https://carstenschelp.github.io/2019/05/12/Online_Covariance_Algorithm_002.html

import copy
import traceback

import torch
import numpy as np

from covariance_torch import OnlineCovariance

# tools for testing
def create_correlated_dataset(n, mu, dependency, scale):
    latent = torch.randn(n, dependency.shape[0])
    dependent = latent @ dependency
    scaled = dependent * torch.tensor(scale)
    scaled_with_offset = scaled + torch.tensor(mu)
    return scaled_with_offset

def torch_to_np(tensor):
    return tensor.cpu().numpy()

def calculate_conventional(data):
    mean = torch.mean(data, dim=0)
    # Using numpy for covariance and correlation coefficient calculations
    cov = np.cov(torch_to_np(data), rowvar=False)
    corrcoef = np.corrcoef(torch_to_np(data), rowvar=False)

    return mean, cov, corrcoef

# Test variances
def test_init():
    a = torch.tensor([[0]])
    w = OnlineCovariance(a)
    assert w.count == 1
    assert torch.allclose(w.mean, torch.tensor([0], dtype=torch.float32))
    assert torch.all(torch.isnan(w.var_s))
    assert torch.allclose(w.var_p, torch.tensor([0], dtype=torch.float32))

    a = torch.tensor([[0], [1]])
    w = OnlineCovariance(a)
    assert w.count == 2
    assert torch.allclose(w.mean, torch.tensor([0.5]))
    assert torch.allclose(w.var_s, torch.tensor([0.5]))
    assert torch.allclose(w.var_p, torch.tensor([0.25]))

    a = torch.tensor([[0, 100], [1, 110], [2, 120], [3, 130], [4, 140]])
    w = OnlineCovariance(a)
    assert w.count == 5
    assert torch.allclose(w.mean, torch.tensor([2, 120], dtype=torch.float32))
    assert torch.allclose(w.var_s, torch.tensor([2.5, 250], dtype=torch.float32))
    assert torch.allclose(w.var_p, torch.tensor([2, 200], dtype=torch.float32))

    a = torch.arange(60).reshape(3, 4, 5)
    w = OnlineCovariance(a)
    assert w.mean.shape == w.var_s.shape == w.var_p.shape == (4, 5)
    a = a.to(torch.float32)
    assert torch.allclose(w.mean, torch.mean(a, axis=0))
    assert torch.allclose(w.var_s, torch.var(a, axis=0, unbiased=True))
    assert torch.allclose(w.var_p, torch.var(a, axis=0, unbiased=False))

#tests
def test_add():
    "Demonstrate OnlineCovariance.add(observation)"

    # COVARIANCE MATRIX shape [3] -> [3, 3]
    data = create_correlated_dataset(
        10000, (2.2, 4.4, 1.5), torch.tensor([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3)
    )

    # CONVENTIONAL COVARIANCE MATRIX
    conventional_mean, conventional_cov, conventional_corrcoef = calculate_conventional(data)

    # ONLINE COVARIANCE MATRIX
    ocov = OnlineCovariance()
    for observation in data:
        ocov.add(observation)

    assert torch.allclose(conventional_mean, ocov.mean), \
        "Mean should be the same with both approaches."

    assert np.allclose(conventional_cov, torch_to_np(ocov.cov), atol=1e-3), \
        "Covariance-matrix should be the same with both approaches."

    assert np.allclose(conventional_corrcoef, torch_to_np(ocov.corrcoef)), \
        "Pearson-Correlationcoefficient-matrix should be the same with both approaches."

    # ONLINE COVARIANCE MATRICES shape [2, 3] -> [3, 3]
    data_2 = create_correlated_dataset(
        10000, (6.6, 1.1, 2.5), torch.tensor([[0.9, 0.2, 0.3],[0.3, 0.1, 0.2],[0.2,0.3,0.5]]), (2, 7, 5)
    )

    # CONVENTIONAL COVARIANCE
    conventional_mean_2, conventional_cov_2, conventional_corrcoef_2 = calculate_conventional(data_2)

    # ONLINE COVARIANCE MATRIX
    zipped_data = []
    for obs_1, obs_2 in zip(data, data_2):
        zipped_data.append(torch.stack([obs_1, obs_2]))
    zipped_data = torch.stack(zipped_data)

    ocov = OnlineCovariance()
    for observation in zipped_data:
        ocov.add(observation)

    assert torch.allclose(conventional_mean, ocov.mean[0]), \
        "Mean should be the same with both approaches."
    assert np.allclose(conventional_cov, torch_to_np(ocov.cov[0]), atol=1e-3), \
        "Covariance-matrix should be the same with both approaches."
    assert np.allclose(conventional_corrcoef, torch_to_np(ocov.corrcoef[0])), \
        "Pearson-Correlationcoefficient-matrix should be the same with both approaches."

    assert torch.allclose(conventional_mean_2, ocov.mean[1]), \
        "Mean should be the same with both approaches."
    assert np.allclose(conventional_cov_2, torch_to_np(ocov.cov[1]), atol=1e-3), \
        "Covariance-matrix should be the same with both approaches."
    assert np.allclose(conventional_corrcoef_2, torch_to_np(ocov.corrcoef[1])), \
        "Pearson-Correlationcoefficient-matrix should be the same with both approaches."


def test_merge():
    "Demonstrate OnlineCovariance.merge()"

    data_part1 = create_correlated_dataset(500, (2.2, 4.4, 1.5), torch.tensor([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3))
    data_part2 = create_correlated_dataset(1000, (5, 6, 2), torch.tensor([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3))
    ocov_part1 = OnlineCovariance()
    ocov_part2 = OnlineCovariance()
    ocov_both = OnlineCovariance()

    for row in data_part1:
        ocov_part1.add(row)
        ocov_both.add(row)

    for row in data_part2:
        ocov_part2.add(row)
        ocov_both.add(row)

    ocov_merged = copy.deepcopy(ocov_part1).merge(ocov_part2)

    assert ocov_both.count == ocov_merged.count, \
        "Count of ocov_both and ocov_merged should be the same."

    assert torch.allclose(ocov_both.mean, ocov_merged.mean), \
        "Mean of ocov_both and ocov_merged should be the same."

    assert np.allclose(torch_to_np(ocov_both.cov), torch_to_np(ocov_merged.cov)), \
        "Covarance-matrix of ocov_both and ocov_merged should be the same."

    assert np.allclose(torch_to_np(ocov_both.corrcoef), torch_to_np(ocov_merged.corrcoef)), \
        "Pearson-Correlationcoefficient-matrix of ocov_both and ocov_merged should be the same."

def test_whitening():
    # COVARIANCE MATRIX shape [3] -> [3, 3]
    data = create_correlated_dataset(
        10000, (2.2, 4.4, 1.5), torch.tensor([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3)
    )

    ocov = OnlineCovariance(data)

    # EIGENVALUES
    eig_vec, eig_val = ocov.eig_vec, ocov.eig_val

    # WHITENING
    whit, whit_inv = ocov.whit, ocov.whit_inv

    assert not torch.allclose(whit, whit_inv, atol=1e-3), \
        "Whitening matrix should not be the inverse of itself in this dataset."
    assert torch.allclose( (whit @ whit_inv), ocov.identity, atol=1e-3), \
        "Whitening matrix by it's inverse should be the identity matrix."

    assert not torch.allclose( ocov.cov, ocov.identity, atol=1e-3), \
        "Covariance matrix for the given dataset should not be the Identity matrix."
    assert torch.allclose( whit @ ocov.cov @ whit.transpose(-2, -1), ocov.identity, atol=1e-3), \
        "Whitening matrix should transform the covariance matrix to the Identity matrix."

    # Calculate for new whitened dataset
    whitened_data = data @ whit.transpose(-2, -1)
    whitened_ocov = OnlineCovariance(whitened_data)

    assert torch.allclose(whitened_ocov.cov, whitened_ocov.identity, atol=1e-3), \
        "Whitening data should lead Covariance Matrix to look like the Identity matrix in this dataset."

def test_all():
    tests = [
        test_init,
        test_add,
        test_merge,
        test_whitening,
    ]
    for test in tests:
        try:
            print(f"# Running test {test.__name__}...")
            test()
            print(f"Test {test.__name__} passed")
        except AssertionError as e:
            print(f"Test {test.__name__} failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_all()
