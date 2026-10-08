import torch
import numpy as np
from collections import OrderedDict

import lietorch

import os.path as osp

from lietorch import SE3

import src.geom.projective_ops as pops
from scipy.spatial.transform import Rotation

def compute_distance_matrix_flow(poses, disps, intrinsics):
    """ compute flow magnitude between all pairs of frames """
    if not isinstance(poses, SE3):
        poses = torch.from_numpy(poses).float().cuda()[None]
        poses = SE3(poses).inv()

        disps = torch.from_numpy(disps).float().cuda()[None]
        intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N = poses.shape[1]

    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1).cuda()
    jj = jj.reshape(-1).cuda()

    MAX_FLOW = 100.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        flow1, val1 = pops.induced_flow(poses, disps, intrinsics, ii[i:i + s], jj[i:i + s])
        flow2, val2 = pops.induced_flow(poses, disps, intrinsics, jj[i:i + s], ii[i:i + s])

        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)

        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.7] = np.inf

        i1 = ii[i:i + s].cpu().numpy()
        j1 = jj[i:i + s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix


def compute_distance_matrix_flow2(poses, disps, intrinsics, beta=0.4):
    """ compute flow magnitude between all pairs of frames """
    N = poses.shape[1]

    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)

    MAX_FLOW = 128.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        flow1a, val1a = pops.induced_flow(poses, disps, intrinsics, ii[i:i + s], jj[i:i + s], tonly=True)
        flow1b, val1b = pops.induced_flow(poses, disps, intrinsics, ii[i:i + s], jj[i:i + s])
        flow2a, val2a = pops.induced_flow(poses, disps, intrinsics, jj[i:i + s], ii[i:i + s], tonly=True)
        flow2b, val2b = pops.induced_flow(poses, disps, intrinsics, ii[i:i + s], jj[i:i + s])

        flow1 = flow1a + beta * flow1b
        val1 = val1a * val2b

        flow2 = flow2a + beta * flow2b
        val2 = val2a * val2b

        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)

        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.8] = np.inf

        i1 = ii[i:i + s].cpu().numpy()
        j1 = jj[i:i + s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix

def graph_to_edge_list(graph):
    ii, jj, kk = [], [], []
    for s, u in enumerate(graph):
        for v in graph[u]:
            ii.append(u)
            jj.append(v)
            kk.append(s)

    ii = torch.as_tensor(ii)
    jj = torch.as_tensor(jj)
    kk = torch.as_tensor(kk)
    return ii, jj, kk

def keyframe_indicies(graph):
    return torch.as_tensor([u for u in graph])

def meshgrid(m, n, device='cuda'):
    ii, jj = torch.meshgrid(torch.arange(m), torch.arange(n))
    return ii.reshape(-1).to(device), jj.reshape(-1).to(device)

def neighbourhood_graph(n, r):
    ii, jj = meshgrid(n, n)
    d = (ii - jj).abs()
    keep = (d >= 1) & (d <= r)
    return ii[keep], jj[keep]


def build_frame_graph(poses, disps, intrinsics, num=16, thresh=24.0, r=2):
    """ construct a frame graph between co-visible frames """
    N = poses.shape[1]
    poses = poses[0].cpu().numpy()
    disps = disps[0][:,3::8,3::8].cpu().numpy()
    intrinsics = intrinsics[0].cpu().numpy() / 8.0
    d = compute_distance_matrix_flow(poses, disps, intrinsics)

    count = 0
    graph = OrderedDict()
    
    for i in range(N):
        graph[i] = []
        d[i,i] = np.inf
        for j in range(i-r, i+r+1):
            if 0 <= j < N and i != j:
                graph[i].append(j)
                d[i,j] = np.inf
                count += 1

    while count < num:
        ix = np.argmin(d)
        i, j = ix // N, ix % N

        if d[i,j] < thresh:
            graph[i].append(j)
            d[i,j] = np.inf
            count += 1
        else:
            break
    
    return graph



def build_frame_graph_v2(poses, disps, intrinsics, num=16, thresh=24.0, r=2):
    """ construct a frame graph between co-visible frames """
    N = poses.shape[1]
    # poses = poses[0].cpu().numpy()
    # disps = disps[0].cpu().numpy()
    # intrinsics = intrinsics[0].cpu().numpy()
    d = compute_distance_matrix_flow2(poses, disps, intrinsics)

    # import matplotlib.pyplot as plt
    # plt.imshow(d)
    # plt.show()

    count = 0
    graph = OrderedDict()
    
    for i in range(N):
        graph[i] = []
        d[i,i] = np.inf
        for j in range(i-r, i+r+1):
            if 0 <= j < N and i != j:
                graph[i].append(j)
                d[i,j] = np.inf
                count += 1

    while 1:
        ix = np.argmin(d)
        i, j = ix // N, ix % N

        if d[i,j] < thresh:
            graph[i].append(j)

            for i1 in range(i-1, i+2):
                for j1 in range(j-1, j+2):
                    if 0 <= i1 < N and 0 <= j1 < N:
                        d[i1, j1] = np.inf

            count += 1
        else:
            break
    
    return graph

