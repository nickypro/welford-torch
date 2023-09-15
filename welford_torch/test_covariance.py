# Source: https://carstenschelp.github.io/2019/05/12/Online_Covariance_Algorithm_002.html

import torch
import numpy as np
from covariance_torch import OnlineCovariance
import traceback
import copy

# tools for testing
def create_correlated_dataset(n, mu, dependency, scale):
    latent = torch.randn(n, dependency.shape[0])
    dependent = latent @ dependency
    scaled = dependent * torch.tensor(scale)
    scaled_with_offset = scaled + torch.tensor(mu)
    return scaled_with_offset

def torch_to_np(tensor):
    return tensor.cpu().numpy()

#tests
def test_add():
    "Demonstrate OnlineCovariance.add(observation)"

    data = create_correlated_dataset(
        10000, (2.2, 4.4, 1.5), torch.tensor([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3)
    )

    # ORIGINAL COVARIANCE MATRIX
    conventional_mean = torch.mean(data, dim=0)
    # Using numpy for covariance and correlation coefficient calculations
    conventional_cov = np.cov(torch_to_np(data), rowvar=False)
    conventional_corrcoef = np.corrcoef(torch_to_np(data), rowvar=False)

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

def test_all():
    tests = [
        test_add,
        test_merge,
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
