import numpy as np

from monet_stats.correlation_metrics import kendalltau, spearmanr


def test_reproduce():
    obs = np.random.rand(10, 2, 2)
    mod = np.random.rand(10, 2, 2)
    obs[0, 0, 0] = np.nan
    mod[1, 0, 1] = np.nan

    print("Testing kendalltau with axis=0")
    res = kendalltau(obs, mod, axis=0)
    print(f"Result shape: {res.shape}")
    assert res.shape == (2, 2)

    print("Testing spearmanr with axis=0")
    res_s = spearmanr(obs, mod, axis=0)
    print(f"Result shape: {res_s.shape}")
    assert res_s.shape == (2, 2)


if __name__ == "__main__":
    test_reproduce()
