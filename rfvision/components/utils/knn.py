import torch

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    assert len(x.shape) == len(y.shape) == 2
    m, n = x.size(0), y.size(0)
    xx = (x**2).sum(1, keepdim=True).expand(m, n)
    yy = (y**2).sum(1, keepdim=True).expand(n, m).T
    dist_mat = xx + yy - 2 * x.matmul(y.T)
    return dist_mat

def knn_search(x, y, k=1):
    assert k > 0, 'k cannot less than 0'
    dist_mat = euclidean_dist(x, y)
    index = dist_mat.argsort(dim=-1)
    index = torch.where(index < k)[1]
    index = index.reshape(-1, k)
    return dist_mat, index

if __name__ == '__main__':
    x = torch.rand((500000, 3))
    y = torch.rand((500, 3))
    dist_mat, index = knn_search(x, y, 5)

