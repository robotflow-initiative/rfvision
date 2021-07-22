from rfvision.models.builder import build_detector, POSE_ESTIMATORS, build_pose_estimators
import numpy as np
from ..detectors import BaseDetector

@POSE_ESTIMATORS.register_module()
class DenseFusion(BaseDetector):
    def __init__(self,
                 train_cfg,
                 test_cfg,
                 estimator=None,
                 refiner=None,
                 refine=False,
                 iteration=2,
                 init_cfg=None):
        super(DenseFusion, self).__init__(init_cfg)

        self.estimator = build_pose_estimators(estimator)
        self.refiner = build_pose_estimators(refiner)

        self.refine = refine
        self.iteration = iteration
        self.best_test = np.Inf
        self.decay_margin = 0.016
        self.refine_margin = 0.013
        self.decay = False

    def forward(self, return_loss=True, **input):
        if return_loss:
            return self.forward_train(**input)
        else:
            return self.forward_test(**input)

    def forward_train(self, **input):
        pred_dict, loss_result, new_points, new_target = self.estimator(**input, return_loss=True)
        if self.refine:
            for ite in range(0, self.iteration):
                pred_dict, loss_result, new_points, new_target = self.refiner(new_points, new_target, pred_dict, return_loss=True, **input)
                loss_result['dis'].backward()
            return loss_result
        else:
            return loss_result

    def forward_test(self, **input):
        pred_dict, _, new_points, new_target = self.estimator(**input, return_loss=False)
        for ite in range(0, self.iteration):
            pred_dict, _, new_points, new_target = self.refiner(pred_dict, return_loss=False)

        return pred_dict
    
    def simple_test(self):
        pass
    
    def aug_test(self):
        pass
    
    def extract_feat(self):
        pass
