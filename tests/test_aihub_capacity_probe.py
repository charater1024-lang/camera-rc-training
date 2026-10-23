import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location("capacity_probe",Path(__file__).parents[1]/"scripts/probe_aihub_capacity.py")
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def test_headroom_counts_remaining_allocator_growth():
    assert m.budget_ready({"free_mib":7168,"utilization_percent":35},0,2,80)
    assert not m.budget_ready({"free_mib":7167,"utilization_percent":35},0,2,80)
    assert m.budget_ready({"free_mib":4096,"utilization_percent":99},3*m.GIB,2,100)
    assert not m.budget_ready({"free_mib":4095,"utilization_percent":5},3*m.GIB,2,80)
    assert not m.budget_ready({"free_mib":12000,"utilization_percent":81},0,2,80)


def test_cap_is_applied_before_any_tensor_allocation():
    calls=[]
    cuda=SimpleNamespace(memory_allocated=lambda device:0,memory_reserved=lambda device:0,
                         get_device_properties=lambda device:SimpleNamespace(total_memory=16*m.GIB),
                         set_per_process_memory_fraction=lambda fraction,device:calls.append((fraction,device)))
    assert m.apply_allocator_cap(SimpleNamespace(cuda=cuda),0)==16*m.GIB
    assert calls==[(5/16,0)]
    cuda.memory_allocated=lambda device:1
    with pytest.raises(RuntimeError,match="fresh process"):
        m.apply_allocator_cap(SimpleNamespace(cuda=cuda),0)
    assert len(calls)==1


@pytest.mark.parametrize("args",[(2,4,16,2,80,30),(4,1,16,2,80,30),(4,9,16,2,80,30),
                                 (4,4,8,2,80,30),(4,4,16,1.5,80,30),(4,4,16,2,96,30),
                                 (4,4,16,2,80,61),(4,4,16,float('nan'),80,30)])
def test_invalid_bounds(args):
    with pytest.raises(ValueError):m.validate_options(*args)


def test_allowed_shared_gpu_probe_bounds():
    m.validate_options(8,4,16,2,80,30)


@pytest.mark.parametrize("mask",[np.array([[0,2]],np.uint8),np.array([[255]],np.uint8),
                                 np.array([[[0]]],np.uint8),np.array([[0.]],np.float32)])
def test_invalid_mask_mapping(mask):
    with pytest.raises(ValueError):m.validate_mask_values(mask)


def test_road_nonroad_ignore_mapping():
    m.validate_mask_values(np.array([[0,1,255]],np.uint8))


def rows():
    return [{"dataset_id":f"aihub_{ds}","sample_id":f"frame{i}","image":f"{ds}/image{i}.jpg",
             "split":"train","source_split":"Training","label_space_id":"road_surface_v1","task":"segmentation"}
            for ds in (13,14,15) for i in range(8)]


def test_sample_selection_is_balanced_deterministic_and_train_only():
    selected=m.select_rows(rows())
    assert selected==m.select_rows(list(reversed(rows())))
    assert len(selected)==16
    assert {r['dataset_id'] for r in selected}=={'aihub_13','aihub_14','aihub_15'}
    bad=rows();bad[0]['source_split']='Validation'
    with pytest.raises(ValueError,match='training split'):m.select_rows(bad)
