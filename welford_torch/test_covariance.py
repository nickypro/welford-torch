# Source: https://carstenschelp.github.io/2019/05/12/Online_Covariance_Algorithm_002.html

# Demonstrate OnlineCovariance.add(observation)
# Create an interdependent dataset
import numpy as np
from covariance_torch import OnlineCovariance

def create_correlated_dataset(n, mu, dependency, scale):
    latent = np.random.randn(n, dependency.shape[0])
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    return scaled_with_offset

def test_add():
    "Demonstrate OnlineCovariance.add(observation)"

    # Create an interdependent dataset
    data = create_correlated_dataset(
        10000, (2.2, 4.4, 1.5), np.array([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3)
    )

    # Calculate mean and covariance conventionally with numpy
    conventional_mean = np.mean(data, axis=0)
    conventional_cov = np.cov(data, rowvar=False)
    conventional_corrcoef = np.corrcoef(data, rowvar=False)

    # Same calculations with OnlineCovariance
    ocov = OnlineCovariance(data.shape[1])
    for observation in data:
        ocov.add(observation)


    # Assert that both ways yield the same result
    assert np.isclose(conventional_mean, ocov.mean).all(), \
    """
    Mean should be the same with both approaches.
    """

    assert np.isclose(conventional_cov, ocov.cov, atol=1e-3).all(), \
    """
    Covariance-matrix should be the same with both approaches.
    """

    assert np.isclose(conventional_corrcoef, ocov.corrcoef).all(), \
    """
    Pearson-Correlationcoefficient-matrix should be the same with both approaches.
    """
    # Note the absolute tolerance-parameter (“atol=1e.-3”) in the second assert.
    # It requires further investigation to see which approach is actually more exact and which one is faster.


def test_merge():
    "Demonstrate OnlineCovariance.merge()"
    # create two differently correllated datasets
    # (again, three dimensions)
    data_part1 = create_correlated_dataset( \
        500, (2.2, 4.4, 1.5), np.array([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3))
    data_part2 = create_correlated_dataset( \
        1000, (5, 6, 2), np.array([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3))
    ocov_part1 = OnlineCovariance(3)
    ocov_part2 = OnlineCovariance(3)
    ocov_both = OnlineCovariance(3)

    # "grow" online-covariances for part 1 and 2 separately but also
    # put all observations into the OnlineCovariance object for both.

    for row in data_part1:
        ocov_part1.add(row)
        ocov_both.add(row)

    for row in data_part2:
        ocov_part2.add(row)
        ocov_both.add(row)

    ocov_merged = ocov_part1.merge(ocov_part2)

    # Assert that count, mean and cov of both grown and merged objects are the same
    assert ocov_both.count == ocov_merged.count, \
    """
    Count of ocov_both and ocov_merged should be the same.
    """

    assert np.isclose(ocov_both.mean, ocov_merged.mean).all(), \
    """
    Mean of ocov_both and ocov_merged should be the same.
    """

    assert np.isclose(ocov_both.cov, ocov_merged.cov).all(), \
    """
    Covarance-matrix of ocov_both and ocov_merged should be the same.
    """

    assert np.isclose(ocov_both.corrcoef, ocov_merged.corrcoef).all(), \
    """
    Pearson-Correlationcoefficient-matrix of ocov_both and ocov_merged should be the same.
    """

def test_all():
    tests = [
        #test_init,
        test_add,
        #test_add_all,
        #test_rollback,
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
