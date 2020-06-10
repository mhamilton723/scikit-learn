import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, KDTree
from annoy import AnnoyIndex
import seaborn as sns
import pandas as pd


def get_data(name, dim, train_size, test_size):
    if name == "gaussian":
        X = np.random.normal(
            loc=0, scale=1, size=(train_size + test_size, dim)).astype(np.float32)
        X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    elif name == "low_dim_gaussian":
        intrinsic_d = 10
        X_low_d = np.random.normal(
            loc=0, scale=1, size=(train_size + test_size, intrinsic_d)).astype(np.float32)
        W = np.random.normal(loc=0, scale=1, size=(intrinsic_d, dim)).astype(np.float32)
        X = np.matmul(X_low_d, W)
        X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    else:
        raise ValueError("unknown dataset name: {}".format(name))
    return X[0:train_size], X[train_size:train_size + test_size]


train, test = get_data("low_dim_gaussian", 100, 100000, 100)
sklearn_ball_tree = BallTree(train, metric='euclidean')
sklearn_kd_tree = KDTree(train, metric='euclidean')

annoy_index = AnnoyIndex(100, 'euclidean')
for i, v in enumerate(test):
    annoy_index.add_item(i, v)
annoy_index.build(10)  # 10 trees


def bf_np(query, k):
    indicies = np.argsort(np.sqrt(np.sum((train - np.expand_dims(query, 0)) ** 2, axis=1)))[:k]
    return indicies


def annoy(query, k):
    return annoy_index.get_nns_by_vector(query, k, search_k=-1, include_distances=True)


def bt_skl(query, k):
    return sklearn_ball_tree.query(np.expand_dims(query, 0), k=k, return_distance=False)


def kd_skl(query, k):
    return sklearn_kd_tree.query(np.expand_dims(query, 0), k=k, return_distance=False)


n = train.shape[0]
#mask = np.concatenate([np.ones(100), np.zeros(n - 100)]).astype(bool)
mask = np.ones(n).astype(bool)
sklearn_ball_tree.update_node_filter(mask)

def cbt_skl(query, k, mask=mask):
    return sklearn_ball_tree.conditional_query(np.expand_dims(query, 0),
                                               mask,
                                               k=k, return_distance=False, compute_node_filter=False)

def cbf_np(query, k, mask=mask):
    selected_indicies = np.arange(0, train.shape[0])[mask]
    indicies = np.argsort(np.sqrt(np.sum((train[mask] - np.expand_dims(query, 0)) ** 2, axis=1)))[:k]
    return selected_indicies[indicies]

test_k = True
if test_k:
    for i in tqdm(range(10)):
        query = test[i]
        for k in [1, 2]:
            i_bt_c = cbt_skl(query, k)
            i_bf_c = cbf_np(query, k)

            i_bt = bt_skl(query, k)
            i_kd = kd_skl(query, k)
            i_bf = bf_np(query, k)
            assert (i_bf_c == i_bt_c).all()
            assert (i_bf == i_bt).all()
            assert (i_bf == i_kd).all()


test_masking = True
if test_masking:
    for i in tqdm(range(10)):
        mask2 = np.random.random(n) > .01
        sklearn_ball_tree.update_node_filter(mask2)
        query = test[i]
        for k in [1, 2]:
            i_bt_c = cbt_skl(query, k, mask=mask2)
            i_bf_c = cbf_np(query, k, mask=mask2)
            assert (i_bf_c == i_bt_c).all()


test_perf = True
results = {"method": [], "time": []}


def time_method(func, k, results, name):
    for i in tqdm(range(test.shape[0])):
        query = test[i]
        start = time.time_ns()
        func(query, k)
        results["time"].append((time.time_ns() - start) / 1e6)
        results["method"].append(name)


if test_perf:
    for k in [1]:  # , 2]:
        print("-------------")
        time_method(cbf_np, k, results, "cbf_np")
        time_method(cbt_skl, k, results, "cbt_skl")
        time_method(bf_np, k, results, "bf_np")
        time_method(bt_skl, k, results, "bt_skl")
        time_method(kd_skl, k, results, "kd_skl")
        # time_method(annoy, k, results, "annoy")

    df = pd.DataFrame(results)
    # df.to_csv("results/benchmark_k_{}.csv".format(k))

    ax = sns.boxplot(x="method", y="time", data=df)
    ax.set_yscale("log")
    plt.show()
