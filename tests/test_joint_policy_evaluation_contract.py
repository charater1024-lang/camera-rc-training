"""Synthetic CPU evaluation only: no training, downloads or checkpoints."""
import torch
from camera_rc.joint_policy.training import evaluate_joint_policy, INPUT_KEYS


class Sequence:
    rows=[{"origin_id":"heldout","condition":"synthetic"}]
    def __len__(self): return 1
    def __getitem__(self,index):
        return {**{k:torch.zeros(1) for k in INPUT_KEYS},"state":torch.tensor([1.,0.]),
                "future_dt_s":torch.tensor([1.,2.,3.]),"targets":torch.tensor([[1.,0.,1.],[2.,0.,1.],[3.,0.,1.]])}


class FixedPolicy(torch.nn.Module):
    def forward(self,**kwargs):
        return torch.tensor([[[10.,0.,1.],[20.,0.,1.],[30.,0.,1.]]])


def test_optional_vehicle_constraints_report_violation_without_claiming_closed_loop():
    limits=dict(max_speed_mps=2.,max_accel_mps2=1.,max_decel_mps2=1.,max_curvature_per_m=.5,max_lateral_accel_mps2=.5)
    report=evaluate_joint_policy(FixedPolicy(),Sequence(),device="cpu",vehicle_limits=limits)
    assert report["physical_constraints_checked"]
    assert report["model"]["physical_constraint_violation"]==1.
    assert report["open_loop_only"] and not report["collision_rate_measured"]
    assert not evaluate_joint_policy(FixedPolicy(),Sequence(),device="cpu")["physical_constraints_checked"]
