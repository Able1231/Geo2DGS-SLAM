
import torch
import lietorch
from lietorch import SE3
from src.entities.factor_graph import FactorGraph
from tqdm import tqdm
import cv2
import torch.nn.functional as F

class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses 
        mainly inherited from DROID-SLAM
    """
    def __init__(self, cfg, net, video, device='cuda:0'):
        
        self.config = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.count = 0
        self.video = video
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image)

    def __fill(self, timestamps, images, depths, intrinsics):
        """ fill operator """
        tt = torch.tensor(timestamps, device=self.device)
        images = torch.stack(images, dim=0)
        if depths is not None:
            depths = torch.stack(depths, dim=0)
        intrinsics = torch.stack(intrinsics, 0)
        inputs = images.to(self.device)

        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(timestamps)

        ts = self.video.tstamp[:N]
        Ps = SE3(self.video.poses[:N])

        # found the location of current timestamp in keyframe queue
        t0 = torch.tensor([ts[ts<=t].shape[0] - 1 for t in timestamps])
        t1 = torch.where(t0 < N-1, t0+1, t0)

        # time interval between nearby keyframes
        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(dim=-1)
        w = v * (tt - ts[t0]).unsqueeze(dim=-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        # temporally put the non-keyframe at the end of keyframe queue
        self.video.counter.value += M
        self.video[N:N+M] = (tt, images[:, 0], Gs.data, depths, depths, intrinsics / 8.0, fmap)

        graph = FactorGraph(self.video, self.update)
        # build edge between current frame and nearby keyframes for optimization
        graph.add_factors(t0.cuda(), torch.arange(N, N+M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N+M).cuda())

        for _ in range(12):
            graph.update(N, N+M, motion_only=True)

        Gs = SE3(self.video.poses[N:N+M].clone())
        self.video.counter.value -= M

        return [Gs]

    @torch.no_grad()
    def __call__(self, image_stream):
        """ fill in poses of non-keyframe images. """

        # store all camera poses
        pose_list = []

        timestamps = []
        images = []
        depths = []
        intrinsics = []

        print("Filling full trajectory ...")
        intrinsic = torch.as_tensor(
            [self.config['cam']['fx'], self.config['cam']['fy'], self.config['cam']['cx'], self.config['cam']['cy']])

        W_edge = 16
        H_edge = 8
        
        W_out_with_edge = 320 + W_edge * 2
        H_out_with_edge = 240 + H_edge * 2
        outsize = (H_out_with_edge, W_out_with_edge)
        intrinsic[0] *= W_out_with_edge / self.config['cam']['W']
        intrinsic[1] *= H_out_with_edge / self.config['cam']['H']
        intrinsic[2] *= W_out_with_edge / self.config['cam']['W']
        intrinsic[3] *= H_out_with_edge / self.config['cam']['H']
        
        
        intrinsic[2] -= W_edge
        intrinsic[3] -= H_edge
        
        for (tstamp, image, depth, _)  in tqdm(image_stream):
            image = cv2.resize(image, (W_out_with_edge, H_out_with_edge))
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # [3, 384, 512]
            image = image.unsqueeze(0)  # [1, 3, 384, 512]

            depth = torch.from_numpy(depth).float()
            depth = F.interpolate(depth[None, None], outsize, mode='nearest')[0, 0]
                
            image = image[:, :, :, W_edge:-W_edge]
            depth = depth[:, W_edge:-W_edge]
            image = image[:, :, H_edge:-H_edge, :]
            depth = depth[H_edge:-H_edge, :]
            
            timestamps.append(tstamp)
            images.append(image)
            depths.append(depth)
            intrinsics.append(intrinsic)

            if len(timestamps) == 16:
                pose_list += self.__fill(timestamps, images, depths, intrinsics)
                timestamps, images, intrinsics, depths = [], [], [], []

        if len(timestamps) > 0:
            pose_list += self.__fill(timestamps, images, depths, intrinsics)

        # stitch pose segments together
        return lietorch.cat(pose_list, dim=0)