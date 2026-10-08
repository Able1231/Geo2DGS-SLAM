import torch
import lietorch
import numpy as np
from lietorch import SE3
from src.entities.factor_graph import FactorGraph
from src.entities.backend import Backend as LoopClosing

class DroidFrontend:
    def __init__(self, net, video, cfg):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=75)  # , upsample=args.upsample)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 50
        self.iters1 = 4*2
        self.iters2 = 2*2

        self.warmup = cfg["tracking"]["warmup"]
        self.beta = cfg["tracking"]["beta"]
        self.frontend_nms = cfg["tracking"]["frontend"]["nms"]
        self.keyframe_thresh = cfg["tracking"]["frontend"]["keyframe_thresh"]
        self.frontend_window = cfg["tracking"]["frontend"]["window"]
        self.frontend_thresh = cfg["tracking"]["frontend"]["thresh"]
        self.frontend_radius = cfg["tracking"]["frontend"]["radius"]

        self.enable_loop = cfg['tracking']['frontend']['enable_loop']
        self.loop_closing = LoopClosing(net, video, cfg)

        self.max_consecutive_drop_of_keyframes = (cfg['tracking']['max_age']/self.iters1)//3
        self.num_keyframes_dropped = 0

        self.update_status = False

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1 - 5, max(self.t1 - self.frontend_window, 0),
                                         rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh,
                                         beta=self.beta, remove=True)

        self.video.disps[self.t1 - 1] = torch.where(self.video.disps_sens[self.t1 - 1] > 0,
                                                    self.video.disps_sens[self.t1 - 1], self.video.disps[self.t1 - 1])

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True,use_mask=True)

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d = self.video.distance([self.t1 - 2], [self.t1 - 1], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 1)
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            cur_t = self.video.counter.value
            self.num_keyframes_dropped = 0
            if self.enable_loop and cur_t > self.frontend_window:
                n_kf, n_edge = self.loop_closing.loop_ba(t_start=0, t_end=cur_t, steps=self.iters2,
                                                         motion_only=False, local_graph=self.graph,
                                                         enable_wq=True)
                if n_edge == 0:
                    for itr in range(self.iters2):
                        self.graph.update(t0=None, t1=None, use_inactive=True,use_mask=True)
                self.last_loop_t = cur_t
            else:
                for itr in range(self.iters2):
                    self.graph.update(t0=None, t1=None, use_inactive=True,use_mask=True)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = self.video.disps[self.t1 - 1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True
        torch.cuda.empty_cache()

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True, use_mask=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1 - 4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1 - 1].clone()
        self.last_disp = self.video.disps[self.t1 - 1].clone()
        self.last_time = self.video.tstamp[self.t1 - 1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

    def __call__(self):
        """ main update """

        self.update_status = False

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            self.update_status = True

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
            self.update_status = True

        return self.update_status


